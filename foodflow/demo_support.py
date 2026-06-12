from __future__ import annotations

import inspect
from collections import Counter
from typing import Callable

import pandas as pd

from .data import PreparedData
from .rerank import estimate_user_merchant_eta, fairness_scores, haversine_km, supply_score_for_merchant
from .rider_sim import assign_order


def streamlit_image_width_kwargs(image_func: Callable) -> dict[str, object]:
    """Return width kwargs compatible with both old and new Streamlit image APIs."""
    params = inspect.signature(image_func).parameters
    if "use_container_width" in params:
        return {"use_container_width": True}
    if "use_column_width" in params:
        return {"use_column_width": True}
    return {}


def clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def merchant_display_name(merchant: pd.Series) -> str:
    raw_name = str(merchant.get("wm_poi_name", "") or "").strip()
    merchant_id = str(merchant.get("wm_poi_id", "unknown"))
    if not raw_name or len(raw_name) > 28 or raw_name.count("-") >= 2:
        return f"商家 #{merchant_id}"
    return raw_name


def merchant_category(merchant: pd.Series) -> str:
    for col in ["primary_first_tag_name", "primary_first_tag_id", "primary_second_tag_name"]:
        value = str(merchant.get(col, "") or "").strip()
        if value and value.lower() != "nan":
            return value
    return "unknown"


def demo_user_cases(users: pd.DataFrame) -> dict[str, str]:
    cases: dict[str, str] = {}

    def add(label: str, user_id: object) -> None:
        user_id_str = str(user_id)
        if user_id_str and user_id_str.lower() != "nan" and user_id_str not in cases.values():
            cases[label] = user_id_str

    user_ids = set(users["user_id"].astype(str))
    if "8" in user_ids:
        add("默认案例：用户 8", "8")
    elif len(users):
        add("默认案例：首个用户", users.iloc[0]["user_id"])

    if "history_orders" in users.columns and len(users):
        history = pd.to_numeric(users["history_orders"], errors="coerce").fillna(0)
        add("复购活跃型", users.loc[history.idxmax(), "user_id"])

    price_col = "avg_order_price" if "avg_order_price" in users.columns else "avg_pay_amt"
    if price_col in users.columns and len(users):
        prices = pd.to_numeric(users[price_col], errors="coerce")
        valid = users[prices.notna()].copy()
        if len(valid):
            valid_prices = pd.to_numeric(valid[price_col], errors="coerce")
            add("高消费型", valid.loc[valid_prices.idxmax(), "user_id"])
            add("价格敏感型", valid.loc[valid_prices.idxmin(), "user_id"])

    for _, row in users.head(20).iterrows():
        if len(cases) >= 5:
            break
        add(f"备选用户 {row['user_id']}", row["user_id"])

    return cases


def user_category_profile(data: PreparedData, user_id: str, top_n: int = 6) -> pd.DataFrame:
    history = data.orders_train[data.orders_train["user_id"].astype(str) == str(user_id)]
    if history.empty:
        return pd.DataFrame(columns=["category", "orders", "share"])
    category_cols = [
        col
        for col in ["primary_first_tag_name", "primary_first_tag_id", "primary_second_tag_name", "primary_second_tag_id"]
        if col in data.merchants.columns
    ]
    if not category_cols:
        return pd.DataFrame(columns=["category", "orders", "share"])
    merged = history.merge(
        data.merchants[["wm_poi_id"] + category_cols],
        on="wm_poi_id",
        how="left",
    )
    category = merged[category_cols[0]]
    for col in category_cols[1:]:
        category = category.fillna(merged[col])
    category = category.astype(str)
    counts = category.value_counts().head(top_n).rename_axis("category").reset_index(name="orders")
    counts["share"] = counts["orders"] / max(float(counts["orders"].sum()), 1.0)
    return counts


def build_recommendation_frame(data: PreparedData, model, user_id: str, recs: list[str], period: str) -> pd.DataFrame:
    merchants = data.merchants.set_index("wm_poi_id", drop=False)
    users = data.users.set_index("user_id", drop=False)
    user_row = users.loc[user_id]
    fairness_lookup = getattr(model, "fair", fairness_scores(data.merchants))
    history = data.orders_train[data.orders_train["user_id"].astype(str) == str(user_id)]
    repeat_counts = Counter(history["wm_poi_id"].astype(str))
    category_profile = user_category_profile(data, user_id, top_n=20)
    category_counts = dict(zip(category_profile["category"].astype(str), category_profile["orders"].astype(int)))

    rows = []
    for rank, merchant_id in enumerate(recs, start=1):
        merchant = merchants.loc[merchant_id]
        if hasattr(model, "component_scores"):
            components = model.component_scores(user_id, merchant_id, period)
            user_weight = float(getattr(model, "user_weight", 1.0))
            fairness_weight = float(getattr(model, "fairness_weight", 0.0))
            eta_weight = float(getattr(model, "eta_weight", 0.0))
            supply_weight = float(getattr(model, "supply_weight", 0.0))
        else:
            user_score = float(model.user_score(user_id, merchant_id, period))
            eta = estimate_user_merchant_eta(user_row, merchant, period)
            components = {
                "user_score": user_score,
                "merchant_fairness": float(fairness_lookup.get(str(merchant_id), 0.0)),
                "eta_minutes": float(eta),
                "eta_score": float(1.0 - min(eta / 70.0, 1.0)),
                "supply_score": float(supply_score_for_merchant(merchant)),
                "final_score": user_score,
            }
            user_weight = 1.0
            fairness_weight = 0.0
            eta_weight = 0.0
            supply_weight = 0.0
        category = merchant_category(merchant)
        repeat = int(repeat_counts.get(str(merchant_id), 0))
        user_price = float(user_row.get("avg_order_price", user_row.get("avg_pay_amt", 35)) or 35)
        merchant_price = float(merchant.get("avg_order_price", 35) or 35)
        distance_km = haversine_km(
            float(user_row.get("lng", 116.40)),
            float(user_row.get("lat", 39.92)),
            float(merchant.get("lng", 116.40)),
            float(merchant.get("lat", 39.92)),
        )
        price_fit = 1.0 - min(abs(user_price - merchant_price) / max(user_price, merchant_price, 1.0), 1.0)

        reasons = []
        if repeat:
            reasons.append(f"复购 {repeat} 次")
        if category_counts.get(category, 0):
            reasons.append(f"偏好品类 {category}")
        if price_fit >= 0.75:
            reasons.append("价格匹配")
        if components["eta_minutes"] <= 45:
            reasons.append("履约较快")
        if components["merchant_fairness"] >= 0.65:
            reasons.append("曝光补偿")
        if not reasons:
            reasons.append("综合得分靠前")

        rows.append(
            {
                "rank": rank,
                "merchant_id": str(merchant_id),
                "merchant_name": merchant_display_name(merchant),
                "category": category,
                "repeat_orders": repeat,
                "distance_km": float(distance_km),
                "avg_price": merchant_price,
                "poi_score": float(merchant.get("poi_score", 0) or 0),
                "order_count": int(float(merchant.get("order_count", 0) or 0)),
                "final_score": float(components["final_score"]),
                "user_score": float(components["user_score"]),
                "fairness": float(components["merchant_fairness"]),
                "eta_minutes": float(components["eta_minutes"]),
                "eta_score": float(components["eta_score"]),
                "supply": float(components["supply_score"]),
                "user_contrib": float(user_weight * components["user_score"]),
                "fairness_contrib": float(fairness_weight * components["merchant_fairness"]),
                "eta_contrib": float(eta_weight * components["eta_score"]),
                "supply_contrib": float(supply_weight * components["supply_score"]),
                "reason": " / ".join(reasons),
                "lng": float(merchant.get("lng", 116.40)),
                "lat": float(merchant.get("lat", 39.92)),
            }
        )
    return pd.DataFrame(rows)


def build_rider_policy_frame(
    user_row: pd.Series,
    merchant_row: pd.Series,
    riders: pd.DataFrame,
    period: str,
    current_time: int = 0,
) -> pd.DataFrame:
    labels = [
        ("nearest", "最近骑手"),
        ("min_eta", "最小 ETA"),
        ("load_aware", "负载感知"),
    ]
    rows = []
    for policy, label in labels:
        rider_id, eta = assign_order(user_row, merchant_row, riders, policy, period, current_time)
        rider = riders[riders["rider_id"].astype(str) == str(rider_id)]
        rows.append(
            {
                "policy": label,
                "policy_key": policy,
                "rider_id": rider_id,
                "eta": float(eta),
                "load": int(rider.iloc[0]["load"]) if not rider.empty else 0,
                "reliability": float(rider.iloc[0]["reliability"]) if not rider.empty else 0.0,
                "lng": float(rider.iloc[0]["lng"]) if not rider.empty else None,
                "lat": float(rider.iloc[0]["lat"]) if not rider.empty else None,
            }
        )
    return pd.DataFrame(rows)
