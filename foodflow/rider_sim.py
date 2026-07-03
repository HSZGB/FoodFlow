from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .rider_data import RiderCalibration
from .rerank import haversine_km


@dataclass
class RiderState:
    rider_id: str
    lng: float
    lat: float
    load: int
    available_at: int
    reliability: float
    speed_kmph: float = 20.0
    service_radius_km: float = 6.0
    acceptance_rate: float = 0.9
    service_minutes: float = 10.0
    income: float = 0.0
    assigned: int = 0


ASSIGNMENT_COLUMNS = [
    "order_index",
    "order_id",
    "user_id",
    "merchant_id",
    "rider_id",
    "eta",
    "score",
    "slot_number",
]


def generate_riders(
    merchants: pd.DataFrame,
    n_riders: int = 80,
    seed: int = 42,
    calibration: RiderCalibration | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    calibration = calibration or RiderCalibration()
    anchors = merchants.sample(n=n_riders, replace=True, random_state=seed).reset_index(drop=True)
    initial_loads = rng.poisson(calibration.initial_load_lambda, n_riders).clip(0, 3)
    speed_values = rng.normal(calibration.speed_kmph, max(calibration.speed_kmph * 0.12, 1.0), n_riders).clip(8.0, 45.0)
    riders = pd.DataFrame(
        {
            "rider_id": [f"r{i:04d}" for i in range(n_riders)],
            "lng": pd.to_numeric(anchors["lng"], errors="coerce").fillna(116.40).to_numpy() + rng.normal(0, 0.012, n_riders),
            "lat": pd.to_numeric(anchors["lat"], errors="coerce").fillna(39.92).to_numpy() + rng.normal(0, 0.010, n_riders),
            "load": initial_loads.astype(int),
            "available_at": rng.integers(0, 10, n_riders),
            "reliability": rng.normal(calibration.reliability_mean, calibration.reliability_std, n_riders).clip(0.72, 0.99),
            "speed_kmph": speed_values,
            "service_radius_km": rng.normal(6.0, 1.0, n_riders).clip(3.5, 9.0),
            "acceptance_rate": rng.normal(0.88, 0.08, n_riders).clip(0.55, 0.99),
            "service_minutes": rng.normal(calibration.service_minutes, 2.0, n_riders).clip(5.0, 30.0),
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
    service_minutes = float(rider_row.get("service_minutes", 9.0))
    prep = service_minutes + (1.0 - float(merchant_row.get("food_comment_avg_score", 4.2)) / 5.0) * 5.0
    peak = 6.0 if period in {"lunch", "dinner"} else 2.0
    load_penalty = float(rider_row.get("load", 0)) * 5.0
    wait = max(float(rider_row.get("available_at", 0)) - current_time, 0.0)
    pickup_speed = max(float(rider_row.get("speed_kmph", 20.0) or 20.0), 1.0)
    delivery_speed = max(pickup_speed * 1.08, 1.0)
    return float(wait + prep + peak + rider_to_store / pickup_speed * 60.0 + store_to_user / delivery_speed * 60.0 + load_penalty)


def acceptance_probability(
    user_row: pd.Series,
    merchant_row: pd.Series,
    rider_row: pd.Series,
    eta: float,
) -> float:
    """
    基于距离、ETA、骑手属性等因素，估计骑手接受订单的概率。
    """
    pickup_distance = haversine_km(
        float(rider_row.get("lng", 116.40)),
        float(rider_row.get("lat", 39.92)),
        float(merchant_row.get("lng", 116.40)),
        float(merchant_row.get("lat", 39.92)),
    )
    delivery_distance = haversine_km(
        float(merchant_row.get("lng", 116.40)),
        float(merchant_row.get("lat", 39.92)),
        float(user_row.get("lng", 116.40)),
        float(user_row.get("lat", 39.92)),
    )
    service_radius = max(float(rider_row.get("service_radius_km", 6.0) or 6.0), 0.1)
    base = float(rider_row.get("acceptance_rate", 0.88) or 0.88) * float(rider_row.get("reliability", 0.88) or 0.88)
    distance_factor = np.exp(-max(pickup_distance + delivery_distance - service_radius, 0.0) / 3.0)
    load_factor = 1.0 / (1.0 + 0.35 * float(rider_row.get("load", 0) or 0))
    eta_factor = 1.0 - min(max(eta - 35.0, 0.0) / 80.0, 0.75)
    return float(np.clip(base * distance_factor * load_factor * eta_factor, 0.02, 0.99))


def rider_score(eta: float, rider_row: pd.Series, acceptance_prob: float | None = None) -> float:
    eta_score = 1.0 - min(eta / 80.0, 1.0)
    reliability = float(rider_row.get("reliability", 0.88))
    load_score = 1.0 / (1.0 + float(rider_row.get("load", 0)))
    accept_score = float(acceptance_prob if acceptance_prob is not None else rider_row.get("acceptance_rate", 0.88))
    return float(0.50 * eta_score + 0.20 * reliability + 0.15 * load_score + 0.15 * accept_score)


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
        candidates["acceptance_prob"] = candidates.apply(
            lambda row: acceptance_probability(user_row, merchant_row, row, float(row["eta"])), axis=1
        )
        candidates["score"] = candidates.apply(lambda row: rider_score(float(row["eta"]), row, float(row["acceptance_prob"])), axis=1)
        chosen = candidates.sort_values("score", ascending=False).iloc[0]
    else:
        raise ValueError(f"Unknown rider policy: {policy}")
    return str(chosen["rider_id"]), float(chosen["eta"])


def assign_orders_batch(
    orders: list[dict[str, object]],
    riders: pd.DataFrame,
    policy: str = "load_aware",
    period: str = "lunch",
    current_time: int = 0,
    max_load: int = 3,
) -> pd.DataFrame:
    """Assign a batch against rider capacity slots with one optimal matching."""
    if not orders or riders.empty:
        return pd.DataFrame(columns=ASSIGNMENT_COLUMNS)
    candidates = riders.reset_index(drop=True).copy()
    slots: list[dict[str, object]] = []
    for rider_idx, rider in candidates.iterrows():
        current_load = int(float(rider.get("load", 0) or 0))
        capacity = max(max_load - current_load, 0)
        for slot_number in range(capacity):
            slot_rider = rider.copy()
            slot_rider["load"] = current_load + slot_number
            slots.append({"rider_idx": int(rider_idx), "slot_number": slot_number, "rider": slot_rider})
    if not slots:
        return pd.DataFrame(columns=ASSIGNMENT_COLUMNS)

    score_matrix = np.full((len(orders), len(slots)), -1e6, dtype=float)
    eta_matrix = np.full((len(orders), len(slots)), np.inf, dtype=float)

    for order_idx, order in enumerate(orders):
        user_row = order["user_row"]
        merchant_row = order["merchant_row"]
        for slot_idx, slot in enumerate(slots):
            rider = slot["rider"]
            eta = estimate_order_eta(user_row, merchant_row, rider, period, current_time)
            acceptance = acceptance_probability(user_row, merchant_row, rider, eta)
            timeout_risk = min(max((eta - 45.0) / 60.0, 0.0), 1.0)
            if policy == "nearest":
                pickup_distance = haversine_km(
                    float(rider.get("lng", 116.40)),
                    float(rider.get("lat", 39.92)),
                    float(merchant_row.get("lng", 116.40)),
                    float(merchant_row.get("lat", 39.92)),
                )
                score = -pickup_distance - eta * 1e-4
            elif policy == "min_eta":
                score = -eta
            elif policy == "load_aware":
                score = rider_score(eta, rider, acceptance) - 0.20 * timeout_risk
            else:
                raise ValueError(f"Unknown rider policy: {policy}")
            score_matrix[order_idx, slot_idx] = score
            eta_matrix[order_idx, slot_idx] = eta

    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    assignments: list[dict[str, object]] = []
    for order_idx, slot_idx in zip(row_ind, col_ind):
        score = float(score_matrix[order_idx, slot_idx])
        if score <= -1e5:
            continue
        slot = slots[int(slot_idx)]
        rider = candidates.iloc[int(slot["rider_idx"])]
        order = orders[order_idx]
        assignments.append(
            {
                "order_index": int(order_idx),
                "order_id": order.get("order_id"),
                "user_id": order.get("user_id"),
                "merchant_id": order.get("merchant_id"),
                "rider_id": str(rider["rider_id"]),
                "eta": float(eta_matrix[order_idx, slot_idx]),
                "score": score,
                "slot_number": int(slot["slot_number"]),
            }
        )
    assignments.sort(key=lambda item: (str(item["rider_id"]), float(item["eta"])))
    return pd.DataFrame(assignments, columns=ASSIGNMENT_COLUMNS)


def update_rider_after_assignment(riders: pd.DataFrame, rider_id: str, eta: float, current_time: int) -> None:
    """
    订单派出去后，更新骑手负载、收入、忙碌时间。
    """
    idx = riders.index[riders["rider_id"].astype(str) == str(rider_id)]
    if len(idx) == 0:
        return
    i = idx[0]
    riders.loc[i, "load"] = int(riders.loc[i, "load"]) + 1
    riders.loc[i, "available_at"] = max(int(riders.loc[i, "available_at"]), int(current_time + eta))
    riders.loc[i, "income"] = float(riders.loc[i, "income"]) + 5.0 + eta * 0.08
    riders.loc[i, "assigned"] = int(riders.loc[i, "assigned"]) + 1


def update_rider_after_delivery(
    riders: pd.DataFrame,
    rider_id: str,
    user_row: pd.Series,
    eta: float,
    current_time: int,
) -> None:
    """
    骑手位置移动到用户位置，模拟配送完成。
    """
    update_rider_after_assignment(riders, rider_id, eta, current_time)
    idx = riders.index[riders["rider_id"].astype(str) == str(rider_id)]
    if len(idx) == 0:
        return
    i = idx[0]
    riders.loc[i, "lng"] = float(user_row.get("lng", riders.loc[i, "lng"]))
    riders.loc[i, "lat"] = float(user_row.get("lat", riders.loc[i, "lat"]))
