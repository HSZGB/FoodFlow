from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .data import PreparedData
from .metrics import gini
from .recommenders import (
    PopularRecommender,
    SessionSpuTripartiteRecommender,
    SeqTunedRecommender,
    SeqXQuadTripartiteRecommender,
    UserOnlyRecommender,
    build_learned_ltr_recommender,
    learned_ltr_model_name,
)
from .rider_data import RiderCalibration
from .rider_sim import assign_order, assign_orders_batch, generate_riders, update_rider_after_delivery


@dataclass(frozen=True)
class SimulationPolicy:
    name: str
    recommender: str
    rider_policy: str
    assignment_mode: str = "greedy"
    fairness: bool = False


def learned_ltr_policy_name() -> str:
    return f"{learned_ltr_model_name()} + MinETA"


DEFAULT_POLICIES = [
    SimulationPolicy("Popular + Nearest", "popular", "nearest"),
    SimulationPolicy("UserOnly + MinETA", "useronly", "min_eta"),
    SimulationPolicy("Seq-Tuned + MinETA", "seq_tuned", "min_eta"),
    SimulationPolicy(learned_ltr_policy_name(), "learned_ltr", "min_eta"),
    SimulationPolicy("Seq-xQuAD-Tripartite + Greedy", "seq_xquad_tripartite", "load_aware", fairness=True),
    SimulationPolicy(
        "Seq-xQuAD-Tripartite + Batch",
        "seq_xquad_tripartite",
        "load_aware",
        assignment_mode="batch",
        fairness=True,
    ),
    SimulationPolicy(
        "Session-SPU-Tripartite + Batch",
        "session_spu_tripartite",
        "load_aware",
        assignment_mode="batch",
        fairness=True,
    ),
    SimulationPolicy("Session-SPU-Tripartite + Greedy", "session_spu_tripartite", "load_aware", fairness=True),
]


def _select_recommender(data: PreparedData, name: str, seed: int):
    if name == "popular":
        return PopularRecommender().fit(data)
    if name == "useronly":
        return UserOnlyRecommender().fit(data)
    if name == "seq_tuned":
        return SeqTunedRecommender().fit(data)
    if name == "learned_ltr":
        return build_learned_ltr_recommender(seed=seed).fit(data)
    if name == "lightgbm_ltr":
        from .recommenders import LightGBMRankerRecommender

        return LightGBMRankerRecommender(seed=seed).fit(data)
    if name == "logistic_ltr":
        from .recommenders import LogisticLTRRecommender

        return LogisticLTRRecommender(seed=seed).fit(data)
    if name == "seq_xquad_tripartite":
        return SeqXQuadTripartiteRecommender().fit(data)
    if name == "session_spu_tripartite":
        return SessionSpuTripartiteRecommender().fit(data)
    raise ValueError(name)


def _stable_policy_seed(seed: int, policy_name: str) -> int:
    digest = hashlib.md5(f"{seed}:{policy_name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _choice_model(
    recs: list[str],
    scores: dict[str, float],
    truth: set[str],
    rng: np.random.Generator,
    no_order_utility: float = 2.0,
    score_weight: float = 1.2,
    truth_bonus: float = 1.75,
    rank_weight: float = 0.25,
) -> str | None:
    if not recs:
        return None
    candidates = [str(item) for item in recs[:10]]
    raw_scores = np.asarray([float(scores.get(item, 0.0)) for item in candidates], dtype=float)
    if np.all(np.isfinite(raw_scores)) and float(raw_scores.max() - raw_scores.min()) > 1e-9:
        normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        normalized_scores = np.linspace(1.0, 0.0, len(candidates))
    rank_prior = np.asarray([1.0 / np.log2(rank + 1) for rank in range(1, len(candidates) + 1)], dtype=float)
    hit_bonus = np.asarray([truth_bonus if item in truth else 0.0 for item in candidates], dtype=float)
    utilities = -0.45 + score_weight * normalized_scores + rank_weight * rank_prior + hit_bonus
    all_utilities = np.concatenate([[no_order_utility], utilities])
    shifted = all_utilities - float(all_utilities.max())
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    selected = int(rng.choice(np.arange(len(probs)), p=probs))
    if selected == 0:
        return None
    return candidates[selected - 1]


def platform_utility(metrics: dict[str, float]) -> float:
    timeout = metrics.get("timeout_rate", 0.0)
    return float(
        0.25 * metrics.get("user_satisfaction", 0.0)
        + 0.25 * metrics.get("on_time_rate", 0.0)
        + 0.20 * metrics.get("completion_rate", 0.0)
        + 0.15 * (1.0 - metrics.get("merchant_exposure_gini", 0.0))
        + 0.10 * (1.0 - metrics.get("rider_load_cv", 0.0))
        - 0.05 * timeout
    )


def run_simulation(
    data: PreparedData,
    policies: list[SimulationPolicy] | None = None,
    seed: int = 42,
    requests_per_step: int = 16,
    steps: int = 8,
    top_k: int = 10,
    verbose: bool = False,
    rider_calibration: RiderCalibration | None = None,
) -> pd.DataFrame:
    policies = policies or DEFAULT_POLICIES
    truth = data.truth_by_user()
    eval_users = list(truth.keys())
    if not eval_users:
        raise ValueError("No test users available for simulation.")
    users_df = data.users.set_index("user_id", drop=False)
    merchants_df = data.merchants.set_index("wm_poi_id", drop=False)
    results = []
    request_seed = _stable_policy_seed(seed, "request-users")
    rider_seed = _stable_policy_seed(seed, "riders")

    for policy_index, policy in enumerate(policies, start=1):
        recommender_seed = _stable_policy_seed(seed, policy.recommender)
        request_rng = np.random.default_rng(request_seed)
        choice_rng = np.random.default_rng(_stable_policy_seed(seed, f"choice:{policy.recommender}"))
        if verbose:
            print(
                f"[simulate] {policy_index}/{len(policies)} {policy.name}: fitting recommender...",
                flush=True,
            )
        fit_started = perf_counter()
        recommender = _select_recommender(data, policy.recommender, recommender_seed)
        fit_done = perf_counter()
        if verbose:
            print(
                f"[simulate] {policy_index}/{len(policies)} {policy.name}: "
                f"running {steps} steps x {requests_per_step} requests...",
                flush=True,
            )
        riders = generate_riders(data.merchants, n_riders=120, seed=rider_seed, calibration=rider_calibration)
        exposure = Counter()
        completed = 0
        total_orders = 0
        timeouts = 0
        etas: list[float] = []
        satisfaction: list[float] = []
        assignment_mode = policy.assignment_mode

        for step in range(steps):
            current_time = step * 5
            ready = riders["available_at"] <= current_time
            riders.loc[ready, "load"] = (
                pd.to_numeric(riders.loc[ready, "load"], errors="coerce").fillna(0).astype(int) - 1
            ).clip(lower=0)
            request_users = request_rng.choice(eval_users, size=min(requests_per_step, len(eval_users)), replace=False)
            periods = {u: "lunch" for u in request_users}
            rec_result = recommender.recommend(list(request_users), top_k, periods)
            pending_orders: list[dict[str, object]] = []
            for user_id in request_users:
                recs = rec_result.recommendations[user_id]
                for rank, merchant_id in enumerate(recs, start=1):
                    exposure[merchant_id] += 1.0 / np.log2(rank + 1)
                chosen = _choice_model(
                    recs,
                    rec_result.scores.get(user_id, {}),
                    truth.get(user_id, set()),
                    choice_rng,
                )
                if chosen is None or chosen not in merchants_df.index or user_id not in users_df.index:
                    continue
                total_orders += 1
                user_row = users_df.loc[user_id]
                merchant_row = merchants_df.loc[chosen]
                order = {
                    "order_id": f"{policy_index}-{step}-{len(pending_orders)}-{user_id}",
                    "user_id": str(user_id),
                    "merchant_id": str(chosen),
                    "user_row": user_row,
                    "merchant_row": merchant_row,
                    "satisfaction": 1.0 if chosen in truth.get(user_id, set()) else 0.45,
                }
                if assignment_mode == "batch":
                    pending_orders.append(order)
                    continue
                available = riders[riders["load"] <= 2].copy()
                rider_id, eta = assign_order(user_row, merchant_row, available, policy.rider_policy, "lunch", current_time)
                if rider_id is None:
                    continue
                update_rider_after_delivery(riders, rider_id, user_row, eta, current_time)
                completed += 1
                etas.append(eta)
                timeouts += int(eta > 45.0)
                satisfaction.append(float(order["satisfaction"]))

            if assignment_mode == "batch" and pending_orders:
                available = riders[riders["load"] <= 2].copy()
                assignments = assign_orders_batch(
                    pending_orders,
                    available,
                    policy.rider_policy,
                    "lunch",
                    current_time,
                )
                orders_by_id = {str(order["order_id"]): order for order in pending_orders}
                for _, assignment in assignments.iterrows():
                    order = orders_by_id.get(str(assignment["order_id"]))
                    if order is None:
                        continue
                    eta = float(assignment["eta"])
                    update_rider_after_delivery(
                        riders,
                        str(assignment["rider_id"]),
                        order["user_row"],
                        eta,
                        current_time,
                    )
                    completed += 1
                    etas.append(eta)
                    timeouts += int(eta > 45.0)
                    satisfaction.append(float(order["satisfaction"]))

        exposure_values = [exposure[m] for m in data.merchant_ids]
        load_values = pd.to_numeric(riders["assigned"], errors="coerce").fillna(0).to_numpy(dtype=float)
        current_load_values = pd.to_numeric(riders["load"], errors="coerce").fillna(0).to_numpy(dtype=float)
        income_values = pd.to_numeric(riders["income"], errors="coerce").fillna(0).to_numpy(dtype=float)
        active_riders = float((load_values > 0).sum())
        load_cv = float(load_values.std() / load_values.mean()) if load_values.mean() > 0 else 0.0
        row = {
            "policy": policy.name,
            "recommender": policy.recommender,
            "rider_policy": policy.rider_policy,
            "assignment_mode": assignment_mode,
            "choice_model": "mnl_softmax",
            "completed_orders": completed,
            "total_orders": total_orders,
            "unassigned_orders": max(total_orders - completed, 0),
            "completion_rate": completed / max(total_orders, 1),
            "avg_eta": float(np.mean(etas)) if etas else 0.0,
            "p95_eta": float(np.percentile(etas, 95)) if etas else 0.0,
            "timeout_rate": timeouts / max(completed, 1),
            "on_time_rate": 1.0 - timeouts / max(completed, 1),
            "avg_rider_load": float(current_load_values.mean()) if len(current_load_values) else 0.0,
            "rider_load_std": float(load_values.std()),
            "rider_load_cv": min(load_cv, 1.0),
            "active_rider_rate": active_riders / max(float(len(load_values)), 1.0),
            "rider_income_gini": gini(income_values),
            "merchant_exposure_gini": gini(exposure_values),
            "user_satisfaction": float(np.mean(satisfaction)) if satisfaction else 0.0,
            "rider_speed_kmph": float(pd.to_numeric(riders.get("speed_kmph", 0), errors="coerce").mean()),
            "rider_service_minutes": float(pd.to_numeric(riders.get("service_minutes", 0), errors="coerce").mean()),
        }
        row["platform_utility"] = platform_utility(row)
        results.append(row)
        if verbose:
            policy_done = perf_counter()
            print(
                f"[simulate] {policy_index}/{len(policies)} {policy.name}: "
                f"fit={fit_done - fit_started:.2f}s "
                f"simulate={policy_done - fit_done:.2f}s "
                f"orders={completed} avg_eta={row['avg_eta']:.2f} "
                f"timeout={row['timeout_rate']:.4f} utility={row['platform_utility']:.4f}",
                flush=True,
            )

    return pd.DataFrame(results)
