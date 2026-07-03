from __future__ import annotations

import inspect
import json
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

from .data import PreparedData
from .kg import kg_path_summary
from .rerank import estimate_user_merchant_eta, fairness_scores, haversine_km, supply_score_for_merchant
from .rider_sim import (
    acceptance_probability,
    assign_order,
    assign_orders_batch,
    estimate_order_eta,
    generate_riders,
    rider_score,
    update_rider_after_delivery,
)


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


VERIFIED_DEMO_USERS = [
    ("老店命中多", "169480"),
    ("命中新店", "167641"),
    ("只点一类", "174231"),
    ("客单价高", "72409"),
    ("常点多类", "147726"),
]


def demo_user_cases(data: PreparedData) -> dict[str, str]:
    """Return distinctive, data-verified users for the interactive demo.

    The preferred TRD users were checked against the demo's default
    Seq-xQuAD-Tripartite Top-20 results. Fallbacks keep mock/custom datasets
    useful and derive their labels from the data instead of assuming TRD IDs.
    """
    users = data.users
    cases: dict[str, str] = {}

    def add(case_type: str, user_id: object) -> None:
        user_id_str = str(user_id)
        if user_id_str and user_id_str.lower() != "nan" and user_id_str not in cases.values():
            cases[f"{case_type} · 用户 {user_id_str}"] = user_id_str

    user_ids = set(users["user_id"].astype(str))
    for case_type, user_id in VERIFIED_DEMO_USERS:
        if user_id in user_ids:
            add(case_type, user_id)

    if len(cases) < 5 and not data.orders_train.empty:
        train = data.orders_train[["user_id", "wm_poi_id"]].copy()
        train["user_id"] = train["user_id"].astype(str)
        train["wm_poi_id"] = train["wm_poi_id"].astype(str)
        stats = train.groupby("user_id").agg(
            history_orders=("wm_poi_id", "size"),
            unique_merchants=("wm_poi_id", "nunique"),
        )
        truth = data.truth_by_user()
        history = data.history_by_user()
        stats["repeat_truth"] = [
            len(truth.get(user_id, set()).intersection(history.get(user_id, []))) for user_id in stats.index
        ]
        repeat_candidates = stats[stats["repeat_truth"] > 0]
        if not repeat_candidates.empty:
            add("常点老店", repeat_candidates["history_orders"].idxmax())

        merchant_categories = data.merchants[["wm_poi_id", "primary_first_tag_id"]]
        category_history = train.merge(merchant_categories, on="wm_poi_id", how="left")
        category_stats = category_history.groupby("user_id").agg(
            history_orders=("wm_poi_id", "size"),
            unique_categories=("primary_first_tag_id", "nunique"),
        )
        active = category_stats[category_stats["history_orders"] >= 3]
        if not active.empty:
            add("常点多类", active.sort_values(["unique_categories", "history_orders"]).index[-1])

    price_col = "avg_order_price" if "avg_order_price" in users.columns else "avg_pay_amt"
    if len(cases) < 5 and price_col in users.columns and len(users):
        valid = users[pd.to_numeric(users[price_col], errors="coerce").notna()].copy()
        if not valid.empty:
            prices = pd.to_numeric(valid[price_col], errors="coerce")
            add("客单价高", valid.loc[prices.idxmax(), "user_id"])

    for _, row in users.head(20).iterrows():
        if len(cases) >= 5:
            break
        add("普通用户", row["user_id"])

    return cases


def recommendation_card_batches(frame: pd.DataFrame, columns: int = 3) -> list[pd.DataFrame]:
    """Split every recommendation row into display batches without truncation."""
    if columns <= 0:
        raise ValueError("columns must be positive")
    return [frame.iloc[start : start + columns] for start in range(0, len(frame), columns)]


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


def _xquad_rank_scores(model, recs: list[str], components_by_item: dict[str, dict[str, float]]) -> dict[str, float]:
    if (
        not recs
        or not components_by_item
        or not hasattr(model, "diversity_weight")
        or not hasattr(model, "tail_weight")
        or not hasattr(model, "merchants")
    ):
        return {}

    scores = {
        str(merchant_id): float(values.get("final_score", 0.0))
        for merchant_id, values in components_by_item.items()
    }
    candidates = [
        merchant_id
        for merchant_id in sorted(scores, key=scores.get, reverse=True)[:80]
        if merchant_id in model.merchants.index
    ]
    if not candidates:
        return {}

    min_score = min(scores[merchant_id] for merchant_id in candidates)
    max_score = max(scores[merchant_id] for merchant_id in candidates)
    scale = max(max_score - min_score, 1e-9)
    relevance_weight = max(
        1.0 - float(getattr(model, "diversity_weight", 0.0)) - float(getattr(model, "tail_weight", 0.0)),
        0.0,
    )
    covered_categories: set[str] = set()
    rank_scores: dict[str, float] = {}
    for merchant_id in [str(item) for item in recs]:
        if merchant_id not in scores or merchant_id not in model.merchants.index:
            continue
        merchant = model.merchants.loc[merchant_id]
        category = str(merchant.get("primary_first_tag_id", "unknown"))
        category_gain = 0.0 if category in covered_categories else 1.0
        order_count = float(merchant.get("order_count", 0) or 0)
        tail_gain = 1.0 / (1.0 + np.log1p(order_count))
        relevance = (scores[merchant_id] - min_score) / scale
        rank_scores[merchant_id] = float(
            relevance_weight * relevance
            + float(getattr(model, "diversity_weight", 0.0)) * category_gain
            + float(getattr(model, "tail_weight", 0.0)) * tail_gain
        )
        covered_categories.add(category)
    return rank_scores


def build_recommendation_frame(
    data: PreparedData,
    model,
    user_id: str,
    recs: list[str],
    period: str,
    ranking_scores: dict[str, float] | None = None,
) -> pd.DataFrame:
    merchants = data.merchants.set_index("wm_poi_id", drop=False)
    users = data.users.set_index("user_id", drop=False)
    user_row = users.loc[user_id]
    model_name = str(getattr(model, "name", type(model).__name__))
    fairness_lookup = getattr(model, "fair", fairness_scores(data.merchants))
    history = data.orders_train[data.orders_train["user_id"].astype(str) == str(user_id)]
    repeat_counts = Counter(history["wm_poi_id"].astype(str))
    truth_merchants = data.truth_by_user().get(str(user_id), set())
    category_profile = user_category_profile(data, user_id, top_n=20)
    category_counts = dict(zip(category_profile["category"].astype(str), category_profile["orders"].astype(int)))
    ranking_scores = ranking_scores or {}
    normalized_components = {}
    if hasattr(model, "_component_scores_for_candidates"):
        candidate_pool = list(recs)
        if hasattr(model, "_sequential_candidates"):
            try:
                candidate_pool = list(getattr(model, "_sequential_candidates")(user_id))
            except Exception:
                candidate_pool = list(recs)
        candidate_pool = list(dict.fromkeys([str(item) for item in candidate_pool] + [str(item) for item in recs]))
        normalized_components = model._component_scores_for_candidates(user_id, candidate_pool, period)
    xquad_rank_scores = _xquad_rank_scores(model, recs, normalized_components)
    uses_tripartite = bool(
        hasattr(model, "component_scores")
        and (
            float(getattr(model, "fairness_weight", 0.0))
            + float(getattr(model, "eta_weight", 0.0))
            + float(getattr(model, "supply_weight", 0.0))
            > 0.0
        )
    )
    uses_session_spu = bool(hasattr(model, "session_weight") or hasattr(model, "spu_weight"))
    uses_xquad = bool(xquad_rank_scores)

    rows = []
    for rank, merchant_id in enumerate(recs, start=1):
        merchant_id = str(merchant_id)
        merchant = merchants.loc[merchant_id]
        if hasattr(model, "component_scores"):
            components = normalized_components.get(merchant_id, model.component_scores(user_id, merchant_id, period))
            user_weight = float(getattr(model, "user_weight", 1.0))
            fairness_weight = float(getattr(model, "fairness_weight", 0.0))
            eta_weight = float(getattr(model, "eta_weight", 0.0))
            supply_weight = float(getattr(model, "supply_weight", 0.0))
            session_weight = float(getattr(model, "session_weight", 0.0))
            spu_weight = float(getattr(model, "spu_weight", 0.0))
        else:
            user_score = float(model.user_score(user_id, merchant_id, period))
            eta = estimate_user_merchant_eta(user_row, merchant, period)
            final_score = float(ranking_scores.get(str(merchant_id), user_score))
            components = {
                "user_score": user_score,
                "merchant_fairness": float(fairness_lookup.get(str(merchant_id), 0.0)),
                "eta_minutes": float(eta),
                "eta_score": float(1.0 - min(eta / 70.0, 1.0)),
                "supply_score": float(supply_score_for_merchant(merchant)),
                "final_score": final_score,
            }
            user_weight = 1.0
            fairness_weight = 0.0
            eta_weight = 0.0
            supply_weight = 0.0
            session_weight = 0.0
            spu_weight = 0.0
        category = merchant_category(merchant)
        repeat = int(repeat_counts.get(str(merchant_id), 0))
        is_truth = merchant_id in truth_merchants
        truth_label = "推荐命中" if is_truth else ""
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
        if components.get("session_score", 0.0) > 0:
            reasons.append("训练期会话点击")
        if components.get("spu_score", 0.0) > 0:
            reasons.append("SPU菜品类目匹配")
        kg_summary = kg_path_summary(data, user_id, merchant_id)
        if kg_summary.repeat_orders:
            reasons.append("KG复购路径")
        elif kg_summary.category_orders:
            reasons.append("KG品类路径")
        if kg_summary.area_orders:
            reasons.append("KG区域路径")
        if not reasons:
            reasons.append("综合得分靠前")

        rows.append(
            {
                "rank": rank,
                "merchant_id": str(merchant_id),
                "merchant_name": merchant_display_name(merchant),
                "category": category,
                "repeat_orders": repeat,
                "is_truth": is_truth,
                "truth_label": truth_label,
                "distance_km": float(distance_km),
                "avg_price": merchant_price,
                "poi_score": float(merchant.get("poi_score", 0) or 0),
                "order_count": int(float(merchant.get("order_count", 0) or 0)),
                "final_score": float(components["final_score"]),
                "rank_score": float(xquad_rank_scores.get(merchant_id, components["final_score"])),
                "user_score": float(components["user_score"]),
                "fairness": float(components["merchant_fairness"]),
                "eta_minutes": float(components["eta_minutes"]),
                "eta_score": float(components["eta_score"]),
                "supply": float(components["supply_score"]),
                "session_score": float(components.get("session_score", 0.0)),
                "spu_score": float(components.get("spu_score", 0.0)),
                "user_contrib": float(user_weight * components.get("user_score_norm", components["user_score"])),
                "fairness_contrib": float(
                    fairness_weight * components.get("merchant_fairness_norm", components["merchant_fairness"])
                ),
                "eta_contrib": float(eta_weight * components.get("eta_score_norm", components["eta_score"])),
                "supply_contrib": float(supply_weight * components.get("supply_score_norm", components["supply_score"])),
                "session_contrib": float(session_weight * components.get("session_score", 0.0)),
                "spu_contrib": float(spu_weight * components.get("spu_score", 0.0)),
                "uses_tripartite": uses_tripartite,
                "uses_session_spu": uses_session_spu,
                "uses_xquad": uses_xquad,
                "model_name": model_name,
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
    candidates["acceptance_prob"] = candidates.apply(
        lambda row: acceptance_probability(user_row, merchant_row, row, float(row["eta"])), axis=1
    )
    candidates["score"] = candidates.apply(
        lambda row: rider_score(float(row["eta"]), row, float(row["acceptance_prob"])), axis=1
    )
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


def choose_peak_replay_merchant(
    recs: list[str],
    scores: dict[str, float],
    rng: np.random.Generator,
) -> str | None:
    if not recs:
        return None
    candidates = [str(item) for item in recs]
    raw_scores = np.asarray([float(scores.get(item, 0.0)) for item in candidates], dtype=float)
    if np.all(np.isfinite(raw_scores)) and float(raw_scores.max() - raw_scores.min()) > 1e-9:
        normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        normalized_scores = np.linspace(1.0, 0.0, len(candidates))
    rank_prior = np.asarray([1.0 / np.log2(rank + 1) for rank in range(1, len(candidates) + 1)], dtype=float)
    utilities = 1.15 * normalized_scores + 0.30 * rank_prior
    shifted = utilities - float(utilities.max())
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    return candidates[int(rng.choice(np.arange(len(candidates)), p=probs))]


def peak_assignment_score(
    user_row: pd.Series,
    merchant_row: pd.Series,
    rider_row: pd.Series,
    rider_policy: str,
    eta: float,
) -> float:
    if rider_policy == "load_aware":
        acceptance = acceptance_probability(user_row, merchant_row, rider_row, eta)
        return rider_score(eta, rider_row, acceptance)
    if rider_policy == "min_eta":
        return -float(eta)
    if rider_policy == "nearest":
        return -haversine_km(
            float(rider_row.get("lng", 116.40)),
            float(rider_row.get("lat", 39.92)),
            float(merchant_row.get("lng", 116.40)),
            float(merchant_row.get("lat", 39.92)),
        )
    return 0.0


def peak_match_record(
    order_index: int,
    order_id: str,
    user_id: str,
    merchant_id: str,
    user_row: pd.Series,
    merchant_row: pd.Series,
    rider_id: str,
    rider_row: pd.Series,
    eta: float,
    score: float,
    slot_number: int = 0,
) -> dict[str, object]:
    return {
        "order_index": int(order_index),
        "order_id": str(order_id),
        "user_id": str(user_id),
        "merchant_id": str(merchant_id),
        "merchant_name": merchant_display_name(merchant_row),
        "rider_id": str(rider_id),
        "slot_number": int(slot_number),
        "eta": float(eta),
        "score": float(score),
        "rider_load": int(float(rider_row.get("load", 0) or 0)),
        "user_lng": float(user_row.get("lng", 116.40)),
        "user_lat": float(user_row.get("lat", 39.92)),
        "merchant_lng": float(merchant_row.get("lng", 116.40)),
        "merchant_lat": float(merchant_row.get("lat", 39.92)),
        "rider_lng": float(rider_row.get("lng", 116.40)),
        "rider_lat": float(rider_row.get("lat", 39.92)),
    }


def build_peak_trace(
    data: PreparedData,
    policies: dict[str, tuple[object, str] | tuple[object, str, str]],
    seed: int = 42,
    steps: int = 6,
    requests_per_step: int = 8,
    top_k: int = 10,
) -> pd.DataFrame:
    truth = data.truth_by_user()
    known_users = set(data.users["user_id"].astype(str))
    history_users = set(data.orders_train["user_id"].astype(str))
    eval_users = [u for u in truth if u in known_users and u in history_users]
    if not eval_users:
        eval_users = [u for u in truth if u in known_users]
    if not eval_users:
        return pd.DataFrame()

    users_df = data.users.set_index("user_id", drop=False)
    merchants_df = data.merchants.set_index("wm_poi_id", drop=False)
    request_rng = np.random.default_rng(seed)
    request_batches = []
    for _ in range(steps):
        sample_size = min(requests_per_step, len(eval_users))
        request_batches.append(request_rng.choice(eval_users, size=sample_size, replace=False).astype(str).tolist())
    base_riders = generate_riders(data.merchants, n_riders=160, seed=seed)
    rows = []

    for policy_name, policy_config in policies.items():
        model = policy_config[0]
        rider_policy = str(policy_config[1])
        assignment_mode = str(policy_config[2]) if len(policy_config) >= 3 else "greedy"
        riders = base_riders.copy(deep=True)
        choice_rng = np.random.default_rng(seed + 17)
        etas: list[float] = []
        completed = 0
        timeout = 0
        current_time = 0

        for step in range(steps):
            current_time = step * 12
            request_users = request_batches[step]
            periods = {user_id: "lunch" for user_id in request_users}
            rec_result = model.recommend(request_users, top_k, periods)
            step_etas: list[float] = []
            step_completed = 0
            step_timeout = 0
            step_example: dict[str, object] = {}
            pending_orders: list[dict[str, object]] = []
            step_matches: list[dict[str, object]] = []

            for request_index, user_id in enumerate(request_users):
                recs = rec_result.recommendations.get(user_id, [])
                chosen = choose_peak_replay_merchant(recs, rec_result.scores.get(user_id, {}), choice_rng)
                if chosen is None or chosen not in merchants_df.index or user_id not in users_df.index:
                    continue

                user_row = users_df.loc[user_id]
                merchant_row = merchants_df.loc[chosen]
                if assignment_mode == "batch":
                    pending_orders.append(
                        {
                            "order_id": f"{policy_name}-{step}-{len(pending_orders)}-{user_id}",
                            "user_id": str(user_id),
                            "merchant_id": str(chosen),
                            "user_row": user_row,
                            "merchant_row": merchant_row,
                        }
                    )
                    continue

                available = riders[riders["available_at"] <= current_time].copy()
                if available.empty:
                    available = riders.copy()
                rider_id, eta = assign_order(
                    user_row,
                    merchant_row,
                    available,
                    rider_policy,
                    "lunch",
                    current_time,
                )
                if rider_id is None:
                    continue
                rider_match = available[available["rider_id"].astype(str) == str(rider_id)]
                rider_row = rider_match.iloc[0] if not rider_match.empty else pd.Series(dtype=object)
                order_id = f"{policy_name}-{step}-{request_index}-{user_id}"
                step_matches.append(
                    peak_match_record(
                        request_index,
                        order_id,
                        str(user_id),
                        str(chosen),
                        user_row,
                        merchant_row,
                        rider_id,
                        rider_row,
                        float(eta),
                        peak_assignment_score(user_row, merchant_row, rider_row, rider_policy, float(eta)),
                    )
                )
                if not step_example and not rider_match.empty:
                    step_example = {
                        "sample_user_id": str(user_id),
                        "sample_merchant_id": str(chosen),
                        "sample_merchant_name": merchant_display_name(merchant_row),
                        "sample_rider_id": str(rider_id),
                        "sample_eta": float(eta),
                        "sample_user_lng": float(user_row.get("lng", 116.40)),
                        "sample_user_lat": float(user_row.get("lat", 39.92)),
                        "sample_merchant_lng": float(merchant_row.get("lng", 116.40)),
                        "sample_merchant_lat": float(merchant_row.get("lat", 39.92)),
                        "sample_rider_lng": float(rider_row.get("lng", 116.40)),
                        "sample_rider_lat": float(rider_row.get("lat", 39.92)),
                        "sample_rider_load": int(float(rider_row.get("load", 0) or 0)),
                    }
                update_rider_after_delivery(riders, rider_id, user_row, eta, current_time)
                completed += 1
                step_completed += 1
                etas.append(float(eta))
                step_etas.append(float(eta))
                is_timeout = int(float(eta) > 45.0)
                timeout += is_timeout
                step_timeout += is_timeout

            if assignment_mode == "batch" and pending_orders:
                available = riders[riders["available_at"] <= current_time].copy()
                if available.empty:
                    available = riders.copy()
                assignments = assign_orders_batch(pending_orders, available, rider_policy, "lunch", current_time)
                orders_by_id = {str(order["order_id"]): order for order in pending_orders}
                batch_matches: list[dict[str, object]] = []
                for _, assignment in assignments.iterrows():
                    order = orders_by_id.get(str(assignment["order_id"]))
                    if order is None:
                        continue
                    rider_id = str(assignment["rider_id"])
                    eta = float(assignment["eta"])
                    rider_match = available[available["rider_id"].astype(str) == rider_id]
                    rider_row = rider_match.iloc[0] if not rider_match.empty else pd.Series(dtype=object)
                    user_row = order["user_row"]
                    merchant_row = order["merchant_row"]
                    batch_matches.append(
                        peak_match_record(
                            int(assignment.get("order_index", len(batch_matches))),
                            str(assignment["order_id"]),
                            str(order["user_id"]),
                            str(order["merchant_id"]),
                            user_row,
                            merchant_row,
                            rider_id,
                            rider_row,
                            eta,
                            float(assignment.get("score", 0.0)),
                            int(assignment.get("slot_number", 0)),
                        )
                    )
                    if not step_example and not rider_match.empty:
                        step_example = {
                            "sample_user_id": str(order["user_id"]),
                            "sample_merchant_id": str(order["merchant_id"]),
                            "sample_merchant_name": merchant_display_name(merchant_row),
                            "sample_rider_id": rider_id,
                            "sample_eta": eta,
                            "sample_user_lng": float(user_row.get("lng", 116.40)),
                            "sample_user_lat": float(user_row.get("lat", 39.92)),
                            "sample_merchant_lng": float(merchant_row.get("lng", 116.40)),
                            "sample_merchant_lat": float(merchant_row.get("lat", 39.92)),
                            "sample_rider_lng": float(rider_row.get("lng", 116.40)),
                            "sample_rider_lat": float(rider_row.get("lat", 39.92)),
                            "sample_rider_load": int(float(rider_row.get("load", 0) or 0)),
                        }
                    update_rider_after_delivery(riders, rider_id, order["user_row"], eta, current_time)
                    completed += 1
                    step_completed += 1
                    etas.append(eta)
                    step_etas.append(eta)
                    is_timeout = int(eta > 45.0)
                    timeout += is_timeout
                    step_timeout += is_timeout
                if batch_matches:
                    step_matches = batch_matches
                    step_example["batch_order_count"] = len(pending_orders)
                    step_example["batch_matched_count"] = len(batch_matches)
                    step_example["batch_matches_json"] = json.dumps(batch_matches, ensure_ascii=False)
            if step_matches:
                step_example["step_order_count"] = len(pending_orders) if assignment_mode == "batch" else len(request_users)
                step_example["step_matched_count"] = len(step_matches)
                step_example["step_matches_json"] = json.dumps(step_matches, ensure_ascii=False)

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
                    "rider_policy": rider_policy,
                    "assignment_mode": assignment_mode,
                    **step_example,
                }
            )

    return pd.DataFrame(rows)


def top_dishes_for_user(data: PreparedData, user_id: str, merchant_id: str, k: int = 6) -> pd.DataFrame:
    """商家菜品(SPU)级推荐：按菜品在该商家的历史销量排序，并标注与用户口味类目的契合。

    TRD 菜品名称与类目均为匿名化编号，展示时保留编号、价格与销量证据。
    """
    columns = ["菜品", "类目", "价格", "历史销量", "契合用户口味"]
    if data.order_spus_train.empty or data.spus.empty:
        return pd.DataFrame(columns=columns)
    order_spus = data.order_spus_train
    merchant_spus = order_spus[order_spus["wm_poi_id"].astype(str) == str(merchant_id)]
    if merchant_spus.empty:
        return pd.DataFrame(columns=columns)
    counts = merchant_spus.groupby("wm_food_spu_id").size().rename("sales")
    spus = data.spus.copy()
    spus["wm_food_spu_id"] = spus["wm_food_spu_id"].astype(str)
    counts.index = counts.index.astype(str)
    joined = spus.set_index("wm_food_spu_id").join(counts, how="inner")
    if joined.empty:
        return pd.DataFrame(columns=columns)

    user_spus = order_spus[order_spus["user_id"].astype(str) == str(user_id)]
    user_categories: set[str] = set()
    if not user_spus.empty and "category" in spus.columns:
        user_joined = user_spus.merge(
            data.spus[["wm_food_spu_id", "category"]], on="wm_food_spu_id", how="left"
        ).dropna(subset=["category"])
        user_categories = set(user_joined["category"].astype(str).value_counts().head(3).index)

    joined = joined.sort_values("sales", ascending=False).head(max(k * 3, k))
    joined["match"] = joined.get("category", pd.Series(index=joined.index, dtype=str)).astype(str).isin(user_categories)
    joined = joined.sort_values(["match", "sales"], ascending=[False, False]).head(k)
    return pd.DataFrame(
        {
            "菜品": ["菜品 " + str(idx) for idx in joined.index],
            "类目": joined.get("category", "").astype(str),
            "价格": pd.to_numeric(joined.get("price", np.nan), errors="coerce").round(1),
            "历史销量": joined["sales"].astype(int),
            "契合用户口味": joined["match"].map({True: "是", False: ""}),
        }
    ).reset_index(drop=True)


def merchant_supply_pressure(data: PreparedData, merchant_row: pd.Series, period: str = "lunch") -> dict[str, object]:
    """商家供给压力评估：需求分位 × 高峰倍率 vs 供给能力，输出爆单风险与品类配额建议。"""
    order_counts = pd.to_numeric(data.merchants["order_count"], errors="coerce").fillna(0)
    merchant_orders = float(merchant_row.get("order_count", 0) or 0)
    demand_percentile = float((order_counts <= merchant_orders).mean())
    peak_multiplier = 1.6 if period in {"lunch", "dinner"} else 1.0
    capacity = supply_score_for_merchant(merchant_row)
    pressure = demand_percentile * peak_multiplier * (1.0 - 0.5 * capacity)
    if pressure >= 1.15:
        risk_level, risk_advice = "高", "高峰时段接近产能上限：建议限流爆款、预告出餐延迟或临时增加出餐位。"
    elif pressure >= 0.75:
        risk_level, risk_advice = "中", "高峰需求明显：建议提前备货热销品类，压缩低销菜品的现做占比。"
    else:
        risk_level, risk_advice = "低", "当前供给余量充足，可承接推荐系统带来的额外曝光。"

    quota = pd.DataFrame()
    if not data.order_spus_train.empty and "category" in data.spus.columns:
        merchant_spus = data.order_spus_train[
            data.order_spus_train["wm_poi_id"].astype(str) == str(merchant_row.get("wm_poi_id"))
        ]
        if not merchant_spus.empty:
            categories = merchant_spus.merge(
                data.spus[["wm_food_spu_id", "category"]], on="wm_food_spu_id", how="left"
            ).dropna(subset=["category"])
            if not categories.empty:
                share = categories["category"].astype(str).value_counts(normalize=True).head(6)
                quota = pd.DataFrame(
                    {
                        "品类": share.index,
                        "受欢迎度": (share * 100).round(1).astype(str) + "%",
                        "建议供给配额": (share * peak_multiplier * 100).clip(upper=100).round(0).astype(int).astype(str) + "%",
                    }
                )
    return {
        "demand_percentile": demand_percentile,
        "peak_multiplier": peak_multiplier,
        "capacity_score": capacity,
        "pressure": float(pressure),
        "risk_level": risk_level,
        "risk_advice": risk_advice,
        "category_quota": quota,
        "expected_peak_orders": float(merchant_orders / max(len(data.orders_train), 1) * 1000 * peak_multiplier),
    }


def enroute_opportunities(
    data: PreparedData,
    rider_row: pd.Series,
    merchant_row: pd.Series,
    user_row: pd.Series,
    period: str = "lunch",
    k: int = 5,
    candidate_pool: int = 60,
) -> pd.DataFrame:
    """骑手顺路单推荐：给定当前配送路径（骑手→商家→用户），评估附近商家新单的边际绕行成本。"""
    from .rider_sim import route_insertion_cost

    columns = ["商家", "距路径中点", "边际绕行", "顺路评级"]
    rider = rider_row.copy()
    rider["route"] = [
        {"kind": "pickup", "lng": float(merchant_row.get("lng", 0.0)), "lat": float(merchant_row.get("lat", 0.0)), "order_id": "current"},
        {"kind": "dropoff", "lng": float(user_row.get("lng", 0.0)), "lat": float(user_row.get("lat", 0.0)), "order_id": "current"},
    ]
    mid_lng = (float(merchant_row.get("lng", 0.0)) + float(user_row.get("lng", 0.0))) / 2
    mid_lat = (float(merchant_row.get("lat", 0.0)) + float(user_row.get("lat", 0.0))) / 2
    merchants = data.merchants.copy()
    merchants["lng"] = pd.to_numeric(merchants["lng"], errors="coerce")
    merchants["lat"] = pd.to_numeric(merchants["lat"], errors="coerce")
    merchants = merchants.dropna(subset=["lng", "lat"])
    merchants["mid_distance"] = merchants.apply(
        lambda row: haversine_km(float(row["lng"]), float(row["lat"]), mid_lng, mid_lat), axis=1
    )
    nearby = merchants[merchants["wm_poi_id"].astype(str) != str(merchant_row.get("wm_poi_id"))]
    nearby = nearby.nsmallest(candidate_pool, "mid_distance")

    rows = []
    for _, candidate in nearby.iterrows():
        # 假设新单的送达点在候选商家常见配送半径内（用用户当前位置方向近似）。
        pseudo_user = pd.Series({"lng": float(user_row.get("lng", 0.0)), "lat": float(user_row.get("lat", 0.0))})
        cost = route_insertion_cost(rider, candidate, pseudo_user, period)
        if cost is None:
            continue
        rows.append(
            {
                "商家": merchant_display_name(candidate),
                "距路径中点": f"{candidate['mid_distance']:.2f} km",
                "边际绕行": f"+{cost['detour']:.1f} min",
                "_detour": cost["detour"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows).sort_values("_detour").head(k)
    frame["顺路评级"] = pd.cut(
        frame["_detour"], bins=[-1, 8, 15, float("inf")], labels=["顺路", "小幅绕行", "明显绕行"]
    ).astype(str)
    return frame.drop(columns=["_detour"]).reset_index(drop=True)
