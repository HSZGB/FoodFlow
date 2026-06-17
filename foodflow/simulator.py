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
    LightGBMRankerRecommender,
    PopularRecommender,
    SeqTunedRecommender,
    SeqXQuadTripartiteRecommender,
    UserOnlyRecommender,
)
from .rider_sim import assign_order, assign_orders_batch, generate_riders, update_rider_after_delivery


@dataclass(frozen=True)
class SimulationPolicy:
    name: str
    recommender: str
    rider_policy: str
    fairness: bool = False


DEFAULT_POLICIES = [
    SimulationPolicy("Popular + Nearest", "popular", "nearest"),
    SimulationPolicy("UserOnly + MinETA", "useronly", "min_eta"),
    SimulationPolicy("Seq-Tuned + MinETA", "seq_tuned", "min_eta"),
    SimulationPolicy("LightGBM-LTR + MinETA", "lightgbm_ltr", "min_eta"),
    SimulationPolicy("Seq-xQuAD-Tripartite", "seq_xquad_tripartite", "load_aware", fairness=True),
    SimulationPolicy("Seq-xQuAD-Tripartite-Batch", "seq_xquad_tripartite", "batch_max_weight", fairness=True),
]


def _select_recommender(data: PreparedData, name: str, seed: int):
    if name == "popular":
        return PopularRecommender().fit(data)
    if name == "useronly":
        return UserOnlyRecommender().fit(data)
    if name == "seq_tuned":
        return SeqTunedRecommender().fit(data)
    if name == "lightgbm_ltr":
        return LightGBMRankerRecommender(seed=seed).fit(data)
    if name == "seq_xquad_tripartite":
        return SeqXQuadTripartiteRecommender().fit(data)
    raise ValueError(name)


def _stable_policy_seed(seed: int, policy_name: str) -> int:
    digest = hashlib.md5(f"{seed}:{policy_name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _choice_model(recs: list[str], truth: set[str], rng: np.random.Generator) -> str | None:
    for item in recs[:10]:
        if item in truth:
            return item
    if not recs:
        return None
    if rng.random() < 0.45:
        weights = np.linspace(1.0, 0.2, len(recs[:10]))
        weights = weights / weights.sum()
        return str(rng.choice(recs[:10], p=weights))
    return None


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
        riders = generate_riders(data.merchants, n_riders=120, seed=rider_seed)
        exposure = Counter()
        completed = 0
        total_orders = 0
        timeouts = 0
        etas: list[float] = []
        satisfaction: list[float] = []

        for step in range(steps):
            current_time = step * 5
            ready = riders["available_at"] <= current_time
            riders.loc[ready, "load"] = (
                pd.to_numeric(riders.loc[ready, "load"], errors="coerce").fillna(0).astype(int) - 1
            ).clip(lower=0)
            request_users = request_rng.choice(eval_users, size=min(requests_per_step, len(eval_users)), replace=False)
            periods = {u: "lunch" for u in request_users}
            rec_result = recommender.recommend(list(request_users), top_k, periods)
            pending_orders: list[dict] = []
            for user_id in request_users:
                recs = rec_result.recommendations[user_id]
                for rank, merchant_id in enumerate(recs, start=1):
                    exposure[merchant_id] += 1.0 / np.log2(rank + 1)
                chosen = _choice_model(recs, truth.get(user_id, set()), choice_rng)
                if chosen is None or chosen not in merchants_df.index or user_id not in users_df.index:
                    continue
                total_orders += 1
                user_row = users_df.loc[user_id]
                merchant_row = merchants_df.loc[chosen]
                matched_truth = chosen in truth.get(user_id, set())
                if policy.rider_policy == "batch_max_weight":
                    pending_orders.append(
                        {
                            "order_id": f"{policy_index}-{step}-{user_id}",
                            "user_id": user_id,
                            "merchant_id": chosen,
                            "user_row": user_row,
                            "merchant_row": merchant_row,
                            "matched_truth": matched_truth,
                        }
                    )
                    continue
                available = riders[riders["load"] <= 2].copy()
                rider_id, eta = assign_order(user_row, merchant_row, available, policy.rider_policy, "lunch", current_time)
                if rider_id is None:
                    continue
                update_rider_after_delivery(riders, rider_id, user_row, eta, current_time)
                completed += 1
                etas.append(eta)
                timeouts += int(eta > 45.0)
                satisfaction.append(1.0 if matched_truth else 0.45)
            if policy.rider_policy == "batch_max_weight" and pending_orders:
                available = riders[riders["load"] <= 2].copy()
                assignments = assign_orders_batch(pending_orders, available, "lunch", current_time)
                for assignment in assignments:
                    order = pending_orders[int(assignment["order_index"])]
                    eta = float(assignment["eta"])
                    update_rider_after_delivery(riders, str(assignment["rider_id"]), order["user_row"], eta, current_time)
                    completed += 1
                    etas.append(eta)
                    timeouts += int(eta > 45.0)
                    satisfaction.append(1.0 if order["matched_truth"] else 0.45)

        exposure_values = [exposure[m] for m in data.merchant_ids]
        load_values = pd.to_numeric(riders["assigned"], errors="coerce").fillna(0).to_numpy(dtype=float)
        current_load_values = pd.to_numeric(riders["load"], errors="coerce").fillna(0).to_numpy(dtype=float)
        income_values = pd.to_numeric(riders["income"], errors="coerce").fillna(0).to_numpy(dtype=float)
        active_riders = float((load_values > 0).sum())
        load_cv = float(load_values.std() / load_values.mean()) if load_values.mean() > 0 else 0.0
        row = {
            "policy": policy.name,
            "completed_orders": completed,
            "total_orders": total_orders,
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
