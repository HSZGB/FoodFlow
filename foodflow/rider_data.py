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


def _minutes_between(start: pd.Series, finish: pd.Series) -> pd.Series:
    start_numeric = pd.to_numeric(start, errors="coerce")
    finish_numeric = pd.to_numeric(finish, errors="coerce")
    numeric_delta = finish_numeric - start_numeric
    if numeric_delta.notna().any():
        return numeric_delta.astype(float)

    start_dt = pd.to_datetime(start, errors="coerce")
    finish_dt = pd.to_datetime(finish, errors="coerce")
    return (finish_dt - start_dt).dt.total_seconds() / 60.0


def estimate_rider_calibration(tasks: pd.DataFrame) -> RiderCalibration:
    if tasks.empty:
        return RiderCalibration()
    required = {"accept_time", "finish_time", "pickup_lng", "pickup_lat", "delivery_lng", "delivery_lat"}
    missing = required - set(tasks.columns)
    if missing:
        raise ValueError(f"Delivery task data missing required columns: {sorted(missing)}")

    frame = tasks.copy()
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
    if "courier_id" in frame.columns:
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
