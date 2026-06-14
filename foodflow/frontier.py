from __future__ import annotations

import pandas as pd


POLICY_MODEL_MAP = {
    "Popular + Nearest": "Popular",
    "UserOnly + MinETA": "UserOnly",
    "Seq-Tuned + MinETA": "Seq-Tuned",
    "Seq-xQuAD-Tripartite": "Seq-xQuAD-Tripartite",
}


def is_pareto_frontier(
    df: pd.DataFrame,
    maximize: list[str],
    minimize: list[str],
) -> pd.Series:
    """Return a boolean mask where True means the row is not dominated by another row."""
    if df.empty:
        return pd.Series(dtype=bool)
    metrics = maximize + minimize
    values = df[metrics].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mask = []
    for idx, row in values.iterrows():
        dominated = False
        for other_idx, other in values.iterrows():
            if idx == other_idx:
                continue
            no_worse = all(other[col] >= row[col] for col in maximize) and all(
                other[col] <= row[col] for col in minimize
            )
            strictly_better = any(other[col] > row[col] for col in maximize) or any(
                other[col] < row[col] for col in minimize
            )
            if no_worse and strictly_better:
                dominated = True
                break
        mask.append(not dominated)
    return pd.Series(mask, index=df.index)


def build_tripartite_frontier(
    offline: pd.DataFrame,
    simulation: pd.DataFrame,
    policy_model_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    policy_model_map = policy_model_map or POLICY_MODEL_MAP
    offline_index = offline.set_index("model", drop=False)
    rows = []
    for _, sim_row in simulation.iterrows():
        policy = str(sim_row["policy"])
        model_name = policy_model_map.get(policy)
        if model_name not in offline_index.index:
            continue
        off_row = offline_index.loc[model_name]
        rows.append(
            {
                "policy": policy,
                "model": model_name,
                "Recall@20": float(off_row["Recall@20"]),
                "NDCG@20": float(off_row["NDCG@20"]),
                "ExposureGini": float(off_row["ExposureGini"]),
                "Coverage@20": float(off_row["Coverage@20"]),
                "avg_eta": float(sim_row["avg_eta"]),
                "timeout_rate": float(sim_row["timeout_rate"]),
                "on_time_rate": float(sim_row["on_time_rate"]),
                "user_satisfaction": float(sim_row["user_satisfaction"]),
                "platform_utility": float(sim_row["platform_utility"]),
            }
        )
    frontier = pd.DataFrame(rows)
    if frontier.empty:
        return frontier
    frontier["is_frontier"] = is_pareto_frontier(
        frontier,
        maximize=["Recall@20", "platform_utility"],
        minimize=["ExposureGini", "avg_eta", "timeout_rate"],
    )
    return frontier.sort_values(["is_frontier", "platform_utility", "Recall@20"], ascending=[False, False, False])
