from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .rerank import haversine_km

# 与 simulator.py 的超时判定保持同一口径（eta > 45 分钟计为超时）。
FOOD_DELIVERY_SLA_MINUTES = 45.0


@dataclass(frozen=True)
class RiderCalibration:
    speed_kmph: float = 20.0
    service_minutes: float = 10.0
    initial_load_lambda: float = 0.6
    reliability_mean: float = 0.9
    reliability_std: float = 0.05


CANONICAL_TASK_COLUMNS = {"accept_time", "finish_time", "pickup_lng", "pickup_lat", "delivery_lng", "delivery_lat"}
LADE_COLUMN_ALIASES = {
    "accept_time": ("receipt_time",),
    "finish_time": ("delivery_time", "delivery_gps_time", "sign_time"),
    "pickup_lng": ("accept_gps_lng", "receipt_lng"),
    "pickup_lat": ("accept_gps_lat", "receipt_lat"),
    "delivery_lng": ("delivery_gps_lng", "sign_lng", "poi_lng", "lng"),
    "delivery_lat": ("delivery_gps_lat", "sign_lat", "poi_lat", "lat"),
}


def adapt_calibration_for_food_delivery(calibration: RiderCalibration) -> RiderCalibration:
    """Keep LaDe load/reliability signals while using food-delivery SLA scale.

    LaDe task duration includes parcel delivery batching and waiting behavior, so
    directly using its service-time median makes a 45-minute food-delivery SLA
    almost entirely timeout. This profile keeps the measured rider workload and
    reliability proxies, but anchors speed/service time to the default
    food-delivery simulation scale.
    """
    return RiderCalibration(
        speed_kmph=20.0,
        service_minutes=10.0,
        initial_load_lambda=calibration.initial_load_lambda,
        reliability_mean=calibration.reliability_mean,
        reliability_std=calibration.reliability_std,
    )


def _first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns and not frame[name].replace("", np.nan).isna().all():
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
    courier_source = _first_existing_column(tasks, ("courier_id", "postman_id", "delivery_user_id"))
    if courier_source is not None:
        out["courier_id"] = tasks[courier_source]

    for target in CANONICAL_TASK_COLUMNS:
        source = target if target in tasks.columns else _first_existing_column(tasks, LADE_COLUMN_ALIASES.get(target, ()))
        if source is not None:
            out[target] = tasks[source]
    return out


def _parse_datetimes(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%m-%d %H:%M:%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values[missing], errors="coerce")
    return parsed


def _minutes_between(start: pd.Series, finish: pd.Series) -> pd.Series:
    start_numeric = pd.to_numeric(start, errors="coerce")
    finish_numeric = pd.to_numeric(finish, errors="coerce")
    numeric_delta = finish_numeric - start_numeric
    if numeric_delta.notna().any():
        return numeric_delta.astype(float)

    start_dt = _parse_datetimes(start)
    finish_dt = _parse_datetimes(finish)
    return (finish_dt - start_dt).dt.total_seconds() / 60.0


def _relative_minutes(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.astype(float)
    dt = _parse_datetimes(values)
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


def _task_distances_km(frame: pd.DataFrame) -> pd.Series:
    pickup_lng = pd.to_numeric(frame["pickup_lng"], errors="coerce")
    pickup_lat = pd.to_numeric(frame["pickup_lat"], errors="coerce")
    delivery_lng = pd.to_numeric(frame["delivery_lng"], errors="coerce")
    delivery_lat = pd.to_numeric(frame["delivery_lat"], errors="coerce")
    coords = pd.concat([pickup_lng, pickup_lat, delivery_lng, delivery_lat], axis=1)
    if coords.abs().max().max() > 360:
        return ((delivery_lng - pickup_lng) ** 2 + (delivery_lat - pickup_lat) ** 2).pow(0.5) / 1000.0
    return frame.apply(
        lambda row: haversine_km(
            float(row["pickup_lng"]),
            float(row["pickup_lat"]),
            float(row["delivery_lng"]),
            float(row["delivery_lat"]),
        ),
        axis=1,
    )


def _task_measurements(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """Normalize tasks and derive per-task duration/distance/speed plus load samples."""
    frame = normalize_delivery_tasks(tasks)
    missing = CANONICAL_TASK_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Delivery task data missing required columns: {sorted(missing)}")

    durations = _minutes_between(frame["accept_time"], frame["finish_time"])
    distances = _task_distances_km(frame)
    valid = pd.DataFrame({"duration": durations, "distance": distances}).replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid["duration"] > 0) & (valid["distance"] > 0)]
    if not valid.empty:
        valid = valid.assign(speed_kmph=valid["distance"] / valid["duration"] * 60.0)
    return frame, valid, _active_loads_at_accept(frame)


def estimate_rider_calibration(tasks: pd.DataFrame, sla_minutes: float = FOOD_DELIVERY_SLA_MINUTES) -> RiderCalibration:
    if tasks.empty:
        return RiderCalibration()
    frame, valid, load_samples = _task_measurements(tasks)
    if valid.empty:
        return RiderCalibration()

    speed = valid["speed_kmph"].clip(lower=8.0, upper=45.0)
    service_minutes = valid["duration"].clip(lower=5.0, upper=90.0)
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
    # 准时率相对显式 SLA 阈值计算（与 simulator 的 45 分钟超时口径一致），
    # 而不是相对样本自身分位数——后者按定义恒等于分位点，与数据无关。
    on_time = float((valid["duration"] <= sla_minutes).mean())
    reliability_mean = float(np.clip(0.72 + 0.24 * on_time, 0.72, 0.99))

    return RiderCalibration(
        speed_kmph=float(speed.median()),
        service_minutes=float(service_minutes.median()),
        initial_load_lambda=initial_load_lambda,
        reliability_mean=reliability_mean,
        reliability_std=reliability_std,
    )


def _empirical_summary(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    quantiles = clean.quantile([0.25, 0.50, 0.75, 0.95]) if not clean.empty else pd.Series(dtype=float)
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()) if not clean.empty else float("nan"),
        "std": float(clean.std(ddof=0)) if not clean.empty else float("nan"),
        "p25": float(quantiles.get(0.25, float("nan"))),
        "p50": float(quantiles.get(0.50, float("nan"))),
        "p75": float(quantiles.get(0.75, float("nan"))),
        "p95": float(quantiles.get(0.95, float("nan"))),
    }


def _lognormal_fit(values: pd.Series, min_samples: int = 8) -> dict[str, float] | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean[clean > 0]
    if len(clean) < min_samples:
        return None
    shape, loc, scale = stats.lognorm.fit(clean, floc=0)
    ks = stats.kstest(clean, "lognorm", args=(shape, loc, scale))
    return {
        "sigma": float(shape),
        "median": float(scale),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def _simulated_input_ks(
    empirical: pd.Series,
    simulated: np.ndarray,
    min_samples: int = 8,
) -> dict[str, float] | None:
    clean = pd.to_numeric(empirical, errors="coerce").dropna()
    if len(clean) < min_samples:
        return None
    ks = stats.ks_2samp(clean.to_numpy(dtype=float), simulated)
    return {"ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue)}


def calibration_diagnostics(
    tasks: pd.DataFrame,
    applied: RiderCalibration,
    raw: RiderCalibration | None = None,
    profile: str = "raw",
    sla_minutes: float = FOOD_DELIVERY_SLA_MINUTES,
    seed: int = 42,
    n_samples: int = 20000,
) -> dict[str, object]:
    """Quantify how well the simulation's input distributions match the task data.

    仿真的速度/服务时长采样规则必须与 rider_sim.generate_riders 保持一致，
    这里对同一生成分布抽样并与外部任务数据做双样本 KS 检验，把
    "校准了什么、偏离多少"显式落进产物，而不是只报几个点估计。
    """
    raw = raw or applied
    frame, valid, load_samples = _task_measurements(tasks)
    rng = np.random.default_rng(seed)

    sim_speed = rng.normal(applied.speed_kmph, max(applied.speed_kmph * 0.12, 1.0), n_samples).clip(8.0, 45.0)
    sim_service = rng.normal(applied.service_minutes, 2.0, n_samples).clip(5.0, 30.0)
    sim_load = rng.poisson(applied.initial_load_lambda, n_samples).clip(0, 3)

    empirical_speed = valid["speed_kmph"] if not valid.empty else pd.Series(dtype=float)
    empirical_duration = valid["duration"] if not valid.empty else pd.Series(dtype=float)
    load_series = pd.Series(load_samples, dtype=float)

    lade_derived = ["initial_load_lambda", "reliability_mean", "reliability_std"]
    if profile == "raw":
        lade_derived = ["speed_kmph", "service_minutes", *lade_derived]
    provenance = {
        field: ("task-data-derived" if field in lade_derived else "food-delivery-default")
        for field in asdict(applied)
    }

    return {
        "profile": profile,
        "sla_minutes": float(sla_minutes),
        "on_time_rate_at_sla": float((empirical_duration <= sla_minutes).mean()) if len(empirical_duration) else float("nan"),
        "raw_calibration": asdict(raw),
        "calibration": asdict(applied),
        "parameter_provenance": provenance,
        "empirical": {
            "task_duration_minutes": _empirical_summary(empirical_duration),
            "task_speed_kmph": _empirical_summary(empirical_speed),
            "active_load_at_accept": _empirical_summary(load_series),
        },
        "lognormal_fit": {
            "task_duration_minutes": _lognormal_fit(empirical_duration),
            "task_speed_kmph": _lognormal_fit(empirical_speed),
        },
        "simulation_vs_data_ks": {
            "speed_kmph": _simulated_input_ks(empirical_speed, sim_speed),
            "service_minutes_vs_task_duration": _simulated_input_ks(empirical_duration, sim_service),
            "initial_load": _simulated_input_ks(load_series, sim_load.astype(float)),
        },
    }
