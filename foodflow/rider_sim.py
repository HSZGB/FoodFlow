from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rerank import haversine_km


@dataclass
class RiderState:
    rider_id: str
    lng: float
    lat: float
    load: int
    available_at: int
    reliability: float
    income: float = 0.0
    assigned: int = 0


ASSIGNMENT_COLUMNS = [
    "order_id",
    "user_id",
    "merchant_id",
    "rider_id",
    "eta",
    "score",
]


def generate_riders(merchants: pd.DataFrame, n_riders: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    anchors = merchants.sample(n=n_riders, replace=True, random_state=seed).reset_index(drop=True)
    riders = pd.DataFrame(
        {
            "rider_id": [f"r{i:04d}" for i in range(n_riders)],
            "lng": pd.to_numeric(anchors["lng"], errors="coerce").fillna(116.40).to_numpy() + rng.normal(0, 0.012, n_riders),
            "lat": pd.to_numeric(anchors["lat"], errors="coerce").fillna(39.92).to_numpy() + rng.normal(0, 0.010, n_riders),
            "load": rng.integers(0, 2, n_riders),
            "available_at": rng.integers(0, 10, n_riders),
            "reliability": rng.normal(0.9, 0.06, n_riders).clip(0.72, 0.99),
            "income": np.zeros(n_riders),
            "assigned": np.zeros(n_riders, dtype=int),
        }
    )
    return riders


def estimate_order_eta(
    user_row: pd.Series,
    merchant_row: pd.Series,
    rider_row: pd.Series,
    period: str,
    current_time: int = 0,
) -> float:
    rider_to_store = haversine_km(
        float(rider_row.get("lng", 116.40)),
        float(rider_row.get("lat", 39.92)),
        float(merchant_row.get("lng", 116.40)),
        float(merchant_row.get("lat", 39.92)),
    )
    store_to_user = haversine_km(
        float(merchant_row.get("lng", 116.40)),
        float(merchant_row.get("lat", 39.92)),
        float(user_row.get("lng", 116.40)),
        float(user_row.get("lat", 39.92)),
    )
    prep = 9.0 + (1.0 - float(merchant_row.get("food_comment_avg_score", 4.2)) / 5.0) * 9.0
    peak = 6.0 if period in {"lunch", "dinner"} else 2.0
    load_penalty = float(rider_row.get("load", 0)) * 5.0
    wait = max(float(rider_row.get("available_at", 0)) - current_time, 0.0)
    return float(wait + prep + peak + rider_to_store / 20.0 * 60.0 + store_to_user / 22.0 * 60.0 + load_penalty)


def rider_score(eta: float, rider_row: pd.Series) -> float:
    eta_score = 1.0 - min(eta / 80.0, 1.0)
    reliability = float(rider_row.get("reliability", 0.88))
    load_score = 1.0 / (1.0 + float(rider_row.get("load", 0)))
    return float(0.58 * eta_score + 0.27 * reliability + 0.15 * load_score)


def assign_order(
    user_row: pd.Series,
    merchant_row: pd.Series,
    riders: pd.DataFrame,
    policy: str,
    period: str = "lunch",
    current_time: int = 0,
) -> tuple[str | None, float]:
    if riders.empty:
        return None, float("inf")
    candidates = riders.copy()
    candidates["eta"] = candidates.apply(
        lambda row: estimate_order_eta(user_row, merchant_row, row, period, current_time), axis=1
    )
    if policy == "nearest":
        candidates["pickup_distance"] = candidates.apply(
            lambda row: haversine_km(
                float(row["lng"]),
                float(row["lat"]),
                float(merchant_row.get("lng", 116.40)),
                float(merchant_row.get("lat", 39.92)),
            ),
            axis=1,
        )
        chosen = candidates.sort_values(["pickup_distance", "eta"]).iloc[0]
    elif policy == "min_eta":
        chosen = candidates.sort_values(["eta", "load"]).iloc[0]
    elif policy == "load_aware":
        candidates["score"] = candidates.apply(lambda row: rider_score(float(row["eta"]), row), axis=1)
        chosen = candidates.sort_values("score", ascending=False).iloc[0]
    else:
        raise ValueError(f"Unknown rider policy: {policy}")
    return str(chosen["rider_id"]), float(chosen["eta"])


def assign_orders_batch(
    orders: list[dict[str, object]],
    riders: pd.DataFrame,
    policy: str,
    period: str = "lunch",
    current_time: int = 0,
) -> pd.DataFrame:
    """Assign a batch of orders with one maximum-weight bipartite matching."""
    if not orders or riders.empty:
        return pd.DataFrame(columns=ASSIGNMENT_COLUMNS)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        rows = []
        remaining = riders.copy()
        for order in orders:
            rider_id, eta = assign_order(
                order["user_row"],
                order["merchant_row"],
                remaining,
                policy,
                period,
                current_time,
            )
            if rider_id is None:
                continue
            rider_row = remaining[remaining["rider_id"].astype(str) == str(rider_id)]
            score = rider_score(float(eta), rider_row.iloc[0]) if not rider_row.empty else 0.0
            rows.append(
                {
                    "order_id": str(order["order_id"]),
                    "user_id": str(order["user_id"]),
                    "merchant_id": str(order["merchant_id"]),
                    "rider_id": str(rider_id),
                    "eta": float(eta),
                    "score": float(score),
                }
            )
            remaining = remaining[remaining["rider_id"].astype(str) != str(rider_id)]
            if remaining.empty:
                break
        return pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)

    candidates = riders.copy().reset_index(drop=True)
    costs = np.zeros((len(orders), len(candidates)), dtype=float)
    etas = np.zeros_like(costs)
    scores = np.zeros_like(costs)

    for order_index, order in enumerate(orders):
        user_row = order["user_row"]
        merchant_row = order["merchant_row"]
        for rider_index, (_, rider_row) in enumerate(candidates.iterrows()):
            eta = estimate_order_eta(user_row, merchant_row, rider_row, period, current_time)
            score = rider_score(float(eta), rider_row)
            etas[order_index, rider_index] = eta
            scores[order_index, rider_index] = score
            if policy == "nearest":
                pickup_distance = haversine_km(
                    float(rider_row["lng"]),
                    float(rider_row["lat"]),
                    float(merchant_row.get("lng", 116.40)),
                    float(merchant_row.get("lat", 39.92)),
                )
                costs[order_index, rider_index] = pickup_distance + eta * 1e-4
            elif policy == "min_eta":
                costs[order_index, rider_index] = eta
            elif policy == "load_aware":
                costs[order_index, rider_index] = -score + eta * 1e-4
            else:
                raise ValueError(f"Unknown rider policy: {policy}")

    order_indices, rider_indices = linear_sum_assignment(costs)
    rows = []
    for order_index, rider_index in zip(order_indices, rider_indices):
        order = orders[int(order_index)]
        rider_row = candidates.iloc[int(rider_index)]
        rows.append(
            {
                "order_id": str(order["order_id"]),
                "user_id": str(order["user_id"]),
                "merchant_id": str(order["merchant_id"]),
                "rider_id": str(rider_row["rider_id"]),
                "eta": float(etas[order_index, rider_index]),
                "score": float(scores[order_index, rider_index]),
            }
        )
    return pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)


def update_rider_after_assignment(riders: pd.DataFrame, rider_id: str, eta: float, current_time: int) -> None:
    idx = riders.index[riders["rider_id"].astype(str) == str(rider_id)]
    if len(idx) == 0:
        return
    i = idx[0]
    riders.loc[i, "load"] = int(riders.loc[i, "load"]) + 1
    riders.loc[i, "available_at"] = int(current_time + eta)
    riders.loc[i, "income"] = float(riders.loc[i, "income"]) + 5.0 + eta * 0.08
    riders.loc[i, "assigned"] = int(riders.loc[i, "assigned"]) + 1
