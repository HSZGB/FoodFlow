from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PreparedData
from .metrics import gini
from .recommenders import (
    OursBalancedRecommender,
    OursFullRecommender,
    SeqTripartiteRecommender,
    PopularRecommender,
    SequentialHybridRecommender,
    UserOnlyRecommender,
)
from .rider_sim import assign_order, generate_riders, update_rider_after_assignment


@dataclass(frozen=True)
class SimulationPolicy:
    name: str
    recommender: str
    rider_policy: str
    fairness: bool = False


DEFAULT_POLICIES = [
    SimulationPolicy("Popular + Nearest", "popular", "nearest"),
    SimulationPolicy("UserOnly + Nearest", "useronly", "nearest"),
    SimulationPolicy("UserOnly + MinETA", "useronly", "min_eta"),
    SimulationPolicy("Seq-Hybrid + MinETA", "seq_hybrid", "min_eta"),
    SimulationPolicy("Seq-Hybrid + LoadAware", "seq_hybrid", "load_aware"),
    SimulationPolicy("Seq-Tripartite", "seq_tripartite", "load_aware", fairness=True),
    SimulationPolicy("Ours-Balanced", "ours_balanced", "load_aware", fairness=True),
    SimulationPolicy("Ours w/o Fairness", "useronly", "load_aware"),
    SimulationPolicy("Ours-Full", "ours", "load_aware", fairness=True),
]


def _select_recommender(data: PreparedData, name: str, seed: int):
    if name == "popular":
        return PopularRecommender().fit(data)
    if name == "useronly":
        return UserOnlyRecommender().fit(data)
    if name == "seq_hybrid":
        return SequentialHybridRecommender().fit(data)
    if name == "seq_tripartite":
        return SeqTripartiteRecommender().fit(data)
    if name == "ours":
        return OursFullRecommender().fit(data)
    if name == "ours_balanced":
        return OursBalancedRecommender().fit(data)
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
) -> pd.DataFrame:
    policies = policies or DEFAULT_POLICIES
    truth = data.truth_by_user()
    eval_users = list(truth.keys())
    if not eval_users:
        raise ValueError("No test users available for simulation.")
    users_df = data.users.set_index("user_id", drop=False)
    merchants_df = data.merchants.set_index("wm_poi_id", drop=False)
    results = []

    for policy in policies:
        policy_seed = _stable_policy_seed(seed, policy.name)
        rng = np.random.default_rng(policy_seed)
        recommender = _select_recommender(data, policy.recommender, seed)
        riders = generate_riders(data.merchants, n_riders=120, seed=policy_seed)
        exposure = Counter()
        completed = 0
        total_orders = 0
        timeouts = 0
        etas: list[float] = []
        satisfaction: list[float] = []

        for step in range(steps):
            current_time = step * 5
            riders.loc[riders["available_at"] <= current_time, "load"] = riders.loc[
                riders["available_at"] <= current_time, "load"
            ].clip(upper=1)
            request_users = rng.choice(eval_users, size=min(requests_per_step, len(eval_users)), replace=False)
            periods = {u: "lunch" for u in request_users}
            rec_result = recommender.recommend(list(request_users), top_k, periods)
            for user_id in request_users:
                recs = rec_result.recommendations[user_id]
                for rank, merchant_id in enumerate(recs, start=1):
                    exposure[merchant_id] += 1.0 / np.log2(rank + 1)
                chosen = _choice_model(recs, truth.get(user_id, set()), rng)
                if chosen is None or chosen not in merchants_df.index or user_id not in users_df.index:
                    continue
                total_orders += 1
                user_row = users_df.loc[user_id]
                merchant_row = merchants_df.loc[chosen]
                available = riders[riders["load"] <= 2].copy()
                rider_id, eta = assign_order(user_row, merchant_row, available, policy.rider_policy, "lunch", current_time)
                if rider_id is None:
                    continue
                update_rider_after_assignment(riders, rider_id, eta, current_time)
                completed += 1
                etas.append(eta)
                timeouts += int(eta > 45.0)
                satisfaction.append(1.0 if chosen in truth.get(user_id, set()) else 0.45)

        exposure_values = [exposure[m] for m in data.merchant_ids]
        load_values = pd.to_numeric(riders["assigned"], errors="coerce").fillna(0).to_numpy(dtype=float)
        load_cv = float(load_values.std() / load_values.mean()) if load_values.mean() > 0 else 0.0
        row = {
            "policy": policy.name,
            "completed_orders": completed,
            "total_orders": total_orders,
            "completion_rate": completed / max(total_orders, 1),
            "avg_eta": float(np.mean(etas)) if etas else 0.0,
            "timeout_rate": timeouts / max(completed, 1),
            "on_time_rate": 1.0 - timeouts / max(completed, 1),
            "rider_load_std": float(load_values.std()),
            "rider_load_cv": min(load_cv, 1.0),
            "merchant_exposure_gini": gini(exposure_values),
            "user_satisfaction": float(np.mean(satisfaction)) if satisfaction else 0.0,
        }
        row["platform_utility"] = platform_utility(row)
        results.append(row)

    return pd.DataFrame(results)
