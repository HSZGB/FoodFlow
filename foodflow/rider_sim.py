from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .rider_data import RiderCalibration
from .rerank import haversine_km, road_km


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
    to_store_minutes = road_km(rider_to_store) / pickup_speed * 60.0
    # 备餐与骑手赶往商家并行发生，取二者较大值而非相加。
    return float(
        wait
        + peak
        + max(prep, to_store_minutes)
        + road_km(store_to_user) / delivery_speed * 60.0
        + load_penalty
    )


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


# ---------------------------------------------------------------------------
# 路径感知（顺路）派单：骑手维护取/送航点序列，新单按最小边际绕行成本插入。
# 相比只看直线距离的贪心，这允许"配送途中顺路接单"：如果新单的取送点落在
# 当前路径附近，边际成本接近零。插入枚举为 O(路径长²)，路径长 ≤ 2×最大负载，
# 计算量可忽略（cheapest-insertion 启发式，餐饮配送调度文献的标准基线）。
# ---------------------------------------------------------------------------

PICKUP_STOP_MINUTES = 3.0
DROPOFF_STOP_MINUTES = 2.0
# 派单目标 = 边际绕行成本 + ROUTE_ETA_WEIGHT × 该单 ETA：
# 纯 detour 最小化会为省车队里程牺牲送达时效，等权组合在 mock 与真实地理
# 上都取得更好的超时率/效用（见 docs/ALGORITHM_RIDER_IMPROVEMENTS.md）。
ROUTE_ETA_WEIGHT = 1.0


def ensure_route_column(riders: pd.DataFrame) -> None:
    if "route" not in riders.columns:
        riders["route"] = [[] for _ in range(len(riders))]


def _rider_route(rider_row: pd.Series) -> list[dict[str, object]]:
    route = rider_row.get("route")
    return list(route) if isinstance(route, list) else []


def _route_leg_minutes(lng1: float, lat1: float, lng2: float, lat2: float, speed_kmph: float) -> float:
    return road_km(haversine_km(lng1, lat1, lng2, lat2)) / max(speed_kmph, 1.0) * 60.0


def _route_total_minutes(start_lng: float, start_lat: float, route: list[dict[str, object]], speed_kmph: float) -> float:
    total = 0.0
    lng, lat = start_lng, start_lat
    for waypoint in route:
        total += _route_leg_minutes(lng, lat, float(waypoint["lng"]), float(waypoint["lat"]), speed_kmph)
        total += PICKUP_STOP_MINUTES if waypoint["kind"] == "pickup" else DROPOFF_STOP_MINUTES
        lng, lat = float(waypoint["lng"]), float(waypoint["lat"])
    return total


def _minutes_to_waypoint(start_lng: float, start_lat: float, route: list[dict[str, object]], target_index: int, speed_kmph: float) -> float:
    total = 0.0
    lng, lat = start_lng, start_lat
    for index, waypoint in enumerate(route):
        total += _route_leg_minutes(lng, lat, float(waypoint["lng"]), float(waypoint["lat"]), speed_kmph)
        total += PICKUP_STOP_MINUTES if waypoint["kind"] == "pickup" else DROPOFF_STOP_MINUTES
        lng, lat = float(waypoint["lng"]), float(waypoint["lat"])
        if index == target_index:
            return total
    return total


def route_pending_orders(rider_row: pd.Series) -> int:
    return sum(1 for waypoint in _rider_route(rider_row) if waypoint["kind"] == "dropoff")


def route_insertion_cost(
    rider_row: pd.Series,
    merchant_row: pd.Series,
    user_row: pd.Series,
    period: str = "lunch",
    current_time: int = 0,
) -> dict[str, float] | None:
    """Best (pickup, dropoff) insertion into the rider's current route.

    返回边际绕行分钟数（detour）、该单送达 ETA 与插入位置；容量不足返回 None。
    """
    route = _rider_route(rider_row)
    speed = max(float(rider_row.get("speed_kmph", 20.0) or 20.0), 1.0)
    lng, lat = float(rider_row.get("lng", 116.40)), float(rider_row.get("lat", 39.92))
    base_minutes = _route_total_minutes(lng, lat, route, speed)

    pickup = {"kind": "pickup", "lng": float(merchant_row.get("lng", lng)), "lat": float(merchant_row.get("lat", lat))}
    dropoff = {"kind": "dropoff", "lng": float(user_row.get("lng", lng)), "lat": float(user_row.get("lat", lat))}

    prep = float(rider_row.get("service_minutes", 10.0)) + (
        1.0 - float(merchant_row.get("food_comment_avg_score", 4.2)) / 5.0
    ) * 5.0
    peak = 6.0 if period in {"lunch", "dinner"} else 2.0

    best: dict[str, float] | None = None
    for i in range(len(route) + 1):
        for j in range(i, len(route) + 1):
            candidate = route[:i] + [pickup] + route[i:j] + [dropoff] + route[j:]
            total = _route_total_minutes(lng, lat, candidate, speed)
            detour = total - base_minutes
            pickup_index = i
            dropoff_index = j + 1
            # 队列等待已由"沿路径行进"刻画，不叠加 available_at；
            # 备餐与骑手赶到取餐点并行，取较大值。
            t_pickup = _minutes_to_waypoint(lng, lat, candidate, pickup_index, speed)
            t_dropoff = _minutes_to_waypoint(lng, lat, candidate, dropoff_index, speed)
            eta = peak + max(prep, t_pickup) + (t_dropoff - t_pickup)
            if best is None or detour < best["detour"] - 1e-9 or (
                abs(detour - best["detour"]) <= 1e-9 and eta < best["eta"]
            ):
                best = {"detour": float(detour), "eta": float(eta), "insert_pickup": i, "insert_dropoff": j}
    return best


def _route_objective(cost: dict[str, float], objective: str) -> float:
    if objective == "min_eta":
        return cost["eta"]
    return cost["detour"] + ROUTE_ETA_WEIGHT * cost["eta"]


def assign_order_route(
    user_row: pd.Series,
    merchant_row: pd.Series,
    riders: pd.DataFrame,
    period: str = "lunch",
    current_time: int = 0,
    max_load: int = 3,
    objective: str = "detour_eta",
) -> dict[str, object] | None:
    """Greedy route-aware dispatch: pick the rider with the cheapest insertion."""
    if riders.empty:
        return None
    ensure_route_column(riders)
    best: dict[str, object] | None = None
    for _, rider in riders.iterrows():
        if route_pending_orders(rider) >= max_load:
            continue
        cost = route_insertion_cost(rider, merchant_row, user_row, period, current_time)
        if cost is None:
            continue
        value = _route_objective(cost, objective)
        if best is None or value < float(best["value"]):
            best = {
                "rider_id": str(rider["rider_id"]),
                "value": float(value),
                "enroute": route_pending_orders(rider) > 0,
                **cost,
            }
    return best


def assign_orders_route_batch(
    orders: list[dict[str, object]],
    riders: pd.DataFrame,
    period: str = "lunch",
    current_time: int = 0,
    max_load: int = 3,
    objective: str = "detour_eta",
) -> list[dict[str, object]]:
    """Batch route-aware dispatch via Hungarian rounds.

    每轮为每个骑手至多分配一单（成本 = 边际绕行 + 超时惩罚），应用插入后
    重新计算下一轮成本，直到订单派完或无可行骑手。相比容量槽位模型，
    这里的成本反映骑手实时路径，天然产生"顺路合单"。
    """
    if not orders or riders.empty:
        return []
    ensure_route_column(riders)
    remaining = list(range(len(orders)))
    assignments: list[dict[str, object]] = []
    for _ in range(len(orders)):
        if not remaining:
            break
        rider_indices = [
            idx for idx, (_, rider) in enumerate(riders.iterrows()) if route_pending_orders(rider) < max_load
        ]
        if not rider_indices:
            break
        cost_matrix = np.full((len(remaining), len(rider_indices)), 1e6, dtype=float)
        details: dict[tuple[int, int], dict[str, object]] = {}
        for row_pos, order_idx in enumerate(remaining):
            order = orders[order_idx]
            for col_pos, rider_pos in enumerate(rider_indices):
                rider = riders.iloc[rider_pos]
                cost = route_insertion_cost(rider, order["merchant_row"], order["user_row"], period, current_time)
                if cost is None:
                    continue
                cost_matrix[row_pos, col_pos] = _route_objective(cost, objective)
                details[(row_pos, col_pos)] = {
                    "rider_id": str(rider["rider_id"]),
                    "enroute": route_pending_orders(rider) > 0,
                    **cost,
                }
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_this_round: list[int] = []
        for row_pos, col_pos in zip(row_ind, col_ind):
            if cost_matrix[row_pos, col_pos] >= 1e5:
                continue
            order_idx = remaining[row_pos]
            detail = details[(row_pos, col_pos)]
            order = orders[order_idx]
            apply_route_assignment(
                riders,
                str(detail["rider_id"]),
                order["merchant_row"],
                order["user_row"],
                str(order.get("order_id")),
                int(detail["insert_pickup"]),
                int(detail["insert_dropoff"]),
                float(detail["eta"]),
                current_time,
            )
            assignments.append(
                {
                    "order_index": int(order_idx),
                    "order_id": order.get("order_id"),
                    "user_id": order.get("user_id"),
                    "merchant_id": order.get("merchant_id"),
                    "rider_id": str(detail["rider_id"]),
                    "eta": float(detail["eta"]),
                    "detour": float(detail["detour"]),
                    "enroute": bool(detail["enroute"]),
                }
            )
            assigned_this_round.append(order_idx)
        if not assigned_this_round:
            break
        remaining = [idx for idx in remaining if idx not in assigned_this_round]
    return assignments


def apply_route_assignment(
    riders: pd.DataFrame,
    rider_id: str,
    merchant_row: pd.Series,
    user_row: pd.Series,
    order_id: str,
    insert_pickup: int,
    insert_dropoff: int,
    eta: float,
    current_time: int,
) -> None:
    idx = riders.index[riders["rider_id"].astype(str) == str(rider_id)]
    if len(idx) == 0:
        return
    i = idx[0]
    route = list(riders.at[i, "route"]) if isinstance(riders.at[i, "route"], list) else []
    pickup = {
        "kind": "pickup",
        "lng": float(merchant_row.get("lng", riders.at[i, "lng"])),
        "lat": float(merchant_row.get("lat", riders.at[i, "lat"])),
        "order_id": str(order_id),
    }
    dropoff = {
        "kind": "dropoff",
        "lng": float(user_row.get("lng", riders.at[i, "lng"])),
        "lat": float(user_row.get("lat", riders.at[i, "lat"])),
        "order_id": str(order_id),
    }
    route = route[:insert_pickup] + [pickup] + route[insert_pickup:insert_dropoff] + [dropoff] + route[insert_dropoff:]
    riders.at[i, "route"] = route
    riders.loc[i, "load"] = route_pending_orders(riders.iloc[i])
    riders.loc[i, "available_at"] = max(int(riders.loc[i, "available_at"]), int(current_time + eta))
    riders.loc[i, "income"] = float(riders.loc[i, "income"]) + 5.0 + eta * 0.08
    riders.loc[i, "assigned"] = int(riders.loc[i, "assigned"]) + 1


def advance_riders_along_routes(riders: pd.DataFrame, elapsed_minutes: float) -> None:
    """Move riders along their routes for `elapsed_minutes` of simulated time.

    走完的航点被弹出：经过 dropoff 即完成一单（load 由剩余 dropoff 数决定），
    骑手位置更新到最后经过的航点。替代旧实现"派单即瞬移到用户"的处理。
    """
    ensure_route_column(riders)
    for i in riders.index:
        route = riders.at[i, "route"]
        if not isinstance(route, list) or not route:
            continue
        budget = float(elapsed_minutes)
        lng, lat = float(riders.loc[i, "lng"]), float(riders.loc[i, "lat"])
        speed = max(float(riders.loc[i, "speed_kmph"] or 20.0), 1.0)
        remaining = list(route)
        while remaining:
            waypoint = remaining[0]
            leg = _route_leg_minutes(lng, lat, float(waypoint["lng"]), float(waypoint["lat"]), speed)
            stop = PICKUP_STOP_MINUTES if waypoint["kind"] == "pickup" else DROPOFF_STOP_MINUTES
            if leg + stop <= budget:
                budget -= leg + stop
                lng, lat = float(waypoint["lng"]), float(waypoint["lat"])
                remaining.pop(0)
                continue
            # 预算不足以完成整段：沿当前路段按比例部分推进（时间预算不跨调用
            # 累积，若不做部分推进，超过单步时长的路段将永远走不完）。
            if leg > 0 and budget > 0:
                fraction = min(budget / leg, 1.0)
                lng += (float(waypoint["lng"]) - lng) * fraction
                lat += (float(waypoint["lat"]) - lat) * fraction
            break
        riders.at[i, "route"] = remaining
        riders.loc[i, "lng"] = lng
        riders.loc[i, "lat"] = lat
        riders.loc[i, "load"] = sum(1 for waypoint in remaining if waypoint["kind"] == "dropoff")
