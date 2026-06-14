from __future__ import annotations

import inspect
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

from .data import PreparedData
from .rerank import estimate_user_merchant_eta, fairness_scores, haversine_km, supply_score_for_merchant
from .rider_sim import assign_order, estimate_order_eta, generate_riders, rider_score, update_rider_after_assignment


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


def rider_candidate_reason(row: pd.Series) -> str:
    reasons = []
    if float(row["eta"]) <= 35:
        reasons.append("ETA 短")
    if float(row["pickup_distance_km"]) <= 2.5:
        reasons.append("取餐近")
    if int(row["load"]) == 0:
        reasons.append("低负载")
    if float(row["reliability"]) >= 0.92:
        reasons.append("可靠性高")
    return " / ".join(reasons[:3]) if reasons else "综合得分靠前"


def build_rider_candidate_frame(
    user_row: pd.Series,
    merchant_row: pd.Series,
    riders: pd.DataFrame,
    period: str,
    current_time: int = 0,
    top_n: int = 8,
) -> pd.DataFrame:
    columns = [
        "rank",
        "rider_id",
        "score",
        "eta",
        "eta_score",
        "pickup_distance_km",
        "load",
        "load_score",
        "reliability",
        "available_at",
        "lng",
        "lat",
        "reason",
    ]
    if riders.empty or top_n <= 0:
        return pd.DataFrame(columns=columns)

    candidates = riders.copy()
    candidates["rider_id"] = candidates["rider_id"].astype(str)
    candidates["lng"] = pd.to_numeric(candidates["lng"], errors="coerce").fillna(116.40)
    candidates["lat"] = pd.to_numeric(candidates["lat"], errors="coerce").fillna(39.92)
    candidates["load"] = pd.to_numeric(candidates["load"], errors="coerce").fillna(0).astype(int)
    candidates["available_at"] = pd.to_numeric(candidates["available_at"], errors="coerce").fillna(0).astype(int)
    candidates["reliability"] = pd.to_numeric(candidates["reliability"], errors="coerce").fillna(0.88)
    candidates["eta"] = candidates.apply(
        lambda row: estimate_order_eta(user_row, merchant_row, row, period, current_time), axis=1
    )
    candidates["eta_score"] = 1.0 - (pd.to_numeric(candidates["eta"], errors="coerce") / 80.0).clip(upper=1.0)
    candidates["load_score"] = 1.0 / (1.0 + candidates["load"].astype(float))
    candidates["score"] = candidates.apply(lambda row: rider_score(float(row["eta"]), row), axis=1)
    candidates["pickup_distance_km"] = candidates.apply(
        lambda row: haversine_km(
            float(row["lng"]),
            float(row["lat"]),
            float(merchant_row.get("lng", 116.40)),
            float(merchant_row.get("lat", 39.92)),
        ),
        axis=1,
    )
    candidates = candidates.sort_values(["score", "eta", "pickup_distance_km"], ascending=[False, True, True]).head(top_n)
    candidates = candidates.reset_index(drop=True)
    candidates["rank"] = np.arange(1, len(candidates) + 1)
    candidates["reason"] = candidates.apply(rider_candidate_reason, axis=1)
    return candidates[columns]


def build_peak_trace(
    data: PreparedData,
    policies: dict[str, tuple[object, str]],
    seed: int = 42,
    steps: int = 6,
    requests_per_step: int = 8,
    top_k: int = 10,
) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    truth = data.truth_by_user()
    known_users = set(data.users["user_id"].astype(str))
    eval_users = [u for u in truth if u in known_users]
    if not eval_users:
        return pd.DataFrame()

    users_df = data.users.set_index("user_id", drop=False)
    merchants_df = data.merchants.set_index("wm_poi_id", drop=False)
    rows = []

    for policy_name, (model, rider_policy) in policies.items():
        riders = generate_riders(data.merchants, n_riders=160, seed=seed + len(policy_name))
        etas: list[float] = []
        completed = 0
        timeout = 0
        current_time = 0

        for step in range(steps):
            current_time = step * 12
            sample_size = min(requests_per_step, len(eval_users))
            request_users = random.choice(eval_users, size=sample_size, replace=False).tolist()
            periods = {user_id: "lunch" for user_id in request_users}
            rec_result = model.recommend(request_users, top_k, periods)
            step_etas: list[float] = []
            step_completed = 0
            step_timeout = 0

            for user_id in request_users:
                recs = rec_result.recommendations.get(user_id, [])
                true_items = truth.get(user_id, set())
                hits = [merchant_id for merchant_id in recs if merchant_id in true_items]
                chosen = hits[0] if hits else (recs[0] if recs else None)
                if chosen is None or chosen not in merchants_df.index or user_id not in users_df.index:
                    continue

                available = riders[riders["available_at"] <= current_time].copy()
                if available.empty:
                    available = riders.copy()
                rider_id, eta = assign_order(
                    users_df.loc[user_id],
                    merchants_df.loc[chosen],
                    available,
                    rider_policy,
                    "lunch",
                    current_time,
                )
                if rider_id is None:
                    continue
                update_rider_after_assignment(riders, rider_id, eta, current_time)
                completed += 1
                step_completed += 1
                etas.append(float(eta))
                step_etas.append(float(eta))
                is_timeout = int(float(eta) > 45.0)
                timeout += is_timeout
                step_timeout += is_timeout

            assigned = riders[pd.to_numeric(riders["assigned"], errors="coerce").fillna(0) > 0]
            rows.append(
                {
                    "policy": policy_name,
                    "step": step + 1,
                    "minute": current_time,
                    "step_completed_orders": step_completed,
                    "step_avg_eta": float(np.mean(step_etas)) if step_etas else 0.0,
                    "step_timeout_rate": float(step_timeout / step_completed) if step_completed else 0.0,
                    "completed_orders": completed,
                    "avg_eta": float(np.mean(etas)) if etas else 0.0,
                    "timeout_rate": float(timeout / completed) if completed else 0.0,
                    "active_riders": int(len(assigned)),
                    "rider_load_std": float(assigned["assigned"].std(ddof=0)) if len(assigned) else 0.0,
                }
            )

    return pd.DataFrame(rows)
