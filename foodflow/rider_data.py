from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rerank import haversine_km


@dataclass(frozen=True)
class RiderCalibration:
    speed_kmph: float = 20.0
    service_minutes: float = 10.0
    initial_load_lambda: float = 0.6
    reliability_mean: float = 0.9
    reliability_std: float = 0.05


CANONICAL_TASK_COLUMNS = {"accept_time", "finish_time", "pickup_lng", "pickup_lat", "delivery_lng", "delivery_lat"}
LADE_COLUMN_ALIASES = {
    "finish_time": ("delivery_time", "delivery_gps_time"),
    "pickup_lng": ("accept_gps_lng",),
    "pickup_lat": ("accept_gps_lat",),
    "delivery_lng": ("delivery_gps_lng", "lng"),
    "delivery_lat": ("delivery_gps_lat", "lat"),
}


def _first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def normalize_delivery_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    """Return delivery tasks in FoodFlow's canonical rider calibration schema.

    LaDe delivery CSVs expose task-accept/task-finish events as
    accept_gps_* and delivery_gps_* columns. Keeping the mapping here lets the
    CLI consume LaDe directly without mutating the source dataset.
    """
    if CANONICAL_TASK_COLUMNS.issubset(tasks.columns):
        return tasks.copy()

    out = pd.DataFrame(index=tasks.index)
    if "courier_id" in tasks.columns:
        out["courier_id"] = tasks["courier_id"]
    elif "postman_id" in tasks.columns:
        out["courier_id"] = tasks["postman_id"]

    for target in CANONICAL_TASK_COLUMNS:
        source = target if target in tasks.columns else _first_existing_column(tasks, LADE_COLUMN_ALIASES.get(target, ()))
        if source is not None:
            out[target] = tasks[source]
    return out


def _minutes_between(start: pd.Series, finish: pd.Series) -> pd.Series:
    start_numeric = pd.to_numeric(start, errors="coerce")
    finish_numeric = pd.to_numeric(finish, errors="coerce")
    numeric_delta = finish_numeric - start_numeric
    if numeric_delta.notna().any():
        return numeric_delta.astype(float)

    start_dt = pd.to_datetime(start, errors="coerce")
    finish_dt = pd.to_datetime(finish, errors="coerce")
    return (finish_dt - start_dt).dt.total_seconds() / 60.0


def _relative_minutes(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.astype(float)
    dt = pd.to_datetime(values, errors="coerce")
    if not dt.notna().any():
        return numeric.astype(float)
    return (dt - dt.min()).dt.total_seconds() / 60.0


def _active_loads_at_accept(frame: pd.DataFrame) -> list[float]:
    if "courier_id" not in frame.columns:
        return []
    starts = _relative_minutes(frame["accept_time"])
    finishes = _relative_minutes(frame["finish_time"])
    intervals = pd.DataFrame({"courier_id": frame["courier_id"], "start": starts, "finish": finishes})
    intervals = intervals.replace([np.inf, -np.inf], np.nan).dropna()
    intervals = intervals[intervals["finish"] > intervals["start"]]
    loads: list[float] = []
    for _, group in intervals.groupby("courier_id", sort=False):
        active_finishes: list[float] = []
        for row in group.sort_values("start").itertuples(index=False):
            active_finishes = [finish for finish in active_finishes if finish > float(row.start)]
            loads.append(float(len(active_finishes)))
            active_finishes.append(float(row.finish))
    return loads


def estimate_rider_calibration(tasks: pd.DataFrame) -> RiderCalibration:
    if tasks.empty:
        return RiderCalibration()
    frame = normalize_delivery_tasks(tasks)
    missing = CANONICAL_TASK_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Delivery task data missing required columns: {sorted(missing)}")

    durations = _minutes_between(frame["accept_time"], frame["finish_time"])
    distances = frame.apply(
        lambda row: haversine_km(
            float(row["pickup_lng"]),
            float(row["pickup_lat"]),
            float(row["delivery_lng"]),
            float(row["delivery_lat"]),
        ),
        axis=1,
    )
    valid = pd.DataFrame({"duration": durations, "distance": distances}).replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid["duration"] > 0) & (valid["distance"] > 0)]
    if valid.empty:
        return RiderCalibration()

    speed = (valid["distance"] / valid["duration"] * 60.0).clip(lower=8.0, upper=45.0)
    service_minutes = valid["duration"].clip(lower=5.0, upper=90.0)
    load_samples = _active_loads_at_accept(frame)
    if "courier_id" in frame.columns:
        if load_samples:
            initial_load_lambda = float(np.clip(np.mean(load_samples), 0.2, 2.5))
            reliability_std = float(np.clip(np.std(load_samples) * 0.04 + 0.02, 0.02, 0.10))
        else:
            task_counts = frame.groupby("courier_id").size()
            initial_load_lambda = float(np.clip(task_counts.mean() / 3.0, 0.2, 2.5))
            reliability_std = float(np.clip(task_counts.std(ddof=0) / max(task_counts.mean(), 1.0) * 0.05, 0.02, 0.10))
    else:
        initial_load_lambda = 0.6
        reliability_std = 0.05
    on_time = float((valid["duration"] <= valid["duration"].quantile(0.75)).mean())
    reliability_mean = float(np.clip(0.72 + 0.24 * on_time, 0.72, 0.99))

    return RiderCalibration(
        speed_kmph=float(speed.median()),
        service_minutes=float(service_minutes.median()),
        initial_load_lambda=initial_load_lambda,
        reliability_mean=reliability_mean,
        reliability_std=reliability_std,
    )
