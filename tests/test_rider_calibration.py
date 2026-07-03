import pandas as pd

from foodflow.rider_data import (
    RiderCalibration,
    adapt_calibration_for_food_delivery,
    calibration_diagnostics,
    estimate_rider_calibration,
)
from foodflow.rider_sim import generate_riders
from foodflow.data import PreparedData
from foodflow.mock_data import make_mock_trd
from foodflow.preprocess import preprocess
from foodflow.simulator import run_simulation


def test_estimate_rider_calibration_from_delivery_tasks():
    tasks = pd.DataFrame(
        [
            {
                "courier_id": "r1",
                "accept_time": 0,
                "finish_time": 30,
                "pickup_lng": 116.400,
                "pickup_lat": 39.900,
                "delivery_lng": 116.430,
                "delivery_lat": 39.920,
            },
            {
                "courier_id": "r1",
                "accept_time": 10,
                "finish_time": 50,
                "pickup_lng": 116.410,
                "pickup_lat": 39.905,
                "delivery_lng": 116.440,
                "delivery_lat": 39.925,
            },
            {
                "courier_id": "r2",
                "accept_time": 5,
                "finish_time": 35,
                "pickup_lng": 116.390,
                "pickup_lat": 39.910,
                "delivery_lng": 116.420,
                "delivery_lat": 39.930,
            },
        ]
    )

    calibration = estimate_rider_calibration(tasks)

    assert calibration.speed_kmph > 0
    assert calibration.service_minutes > 0
    assert calibration.initial_load_lambda > 0
    assert 0.72 <= calibration.reliability_mean <= 0.99


def test_estimate_rider_calibration_accepts_lade_delivery_schema():
    tasks = pd.DataFrame(
        [
            {
                "order_id": 1,
                "city": "cq",
                "courier_id": 101,
                "accept_time": "2021-03-18 12:00:00",
                "accept_gps_lng": 116.400,
                "accept_gps_lat": 39.900,
                "delivery_time": "2021-03-18 12:24:00",
                "delivery_gps_lng": 116.430,
                "delivery_gps_lat": 39.920,
            },
            {
                "order_id": 2,
                "city": "cq",
                "courier_id": 101,
                "accept_time": "2021-03-18 12:10:00",
                "accept_gps_lng": 116.410,
                "accept_gps_lat": 39.905,
                "delivery_time": "2021-03-18 12:45:00",
                "delivery_gps_lng": 116.440,
                "delivery_gps_lat": 39.925,
            },
            {
                "order_id": 3,
                "city": "cq",
                "courier_id": 102,
                "accept_time": "2021-03-18 13:00:00",
                "accept_gps_lng": 116.390,
                "accept_gps_lat": 39.910,
                "delivery_time": "2021-03-18 13:20:00",
                "delivery_gps_lng": 116.420,
                "delivery_gps_lat": 39.930,
            },
        ]
    )

    calibration = estimate_rider_calibration(tasks)

    assert calibration.speed_kmph > 0
    assert calibration.service_minutes > 0
    assert calibration.initial_load_lambda > 0
    assert 0.72 <= calibration.reliability_mean <= 0.99


def test_estimate_rider_calibration_accepts_lade_five_cities_schema():
    tasks = pd.DataFrame(
        [
            {
                "delivery_user_id": "c1",
                "from_city_name": "上海市",
                "poi_lng": 10563512.0,
                "poi_lat": -7458320.0,
                "receipt_time": "03-18 13:35:00",
                "receipt_lng": 10561603.0,
                "receipt_lat": -7457997.0,
                "sign_time": "03-18 14:51:00",
                "sign_lng": None,
                "sign_lat": None,
                "ds": "0318",
            },
            {
                "delivery_user_id": "c1",
                "from_city_name": "上海市",
                "poi_lng": 10563600.0,
                "poi_lat": -7458400.0,
                "receipt_time": "03-18 14:00:00",
                "receipt_lng": 10562000.0,
                "receipt_lat": -7458100.0,
                "sign_time": "03-18 14:30:00",
                "sign_lng": None,
                "sign_lat": None,
                "ds": "0318",
            },
        ]
    )

    calibration = estimate_rider_calibration(tasks)

    assert calibration.speed_kmph > 0
    assert calibration.service_minutes > 0
    assert calibration.initial_load_lambda > 0


def test_initial_load_uses_task_overlap_not_total_history_volume():
    tasks = pd.DataFrame(
        [
            {
                "courier_id": "r1",
                "accept_time": i * 60,
                "finish_time": i * 60 + 20,
                "pickup_lng": 116.400,
                "pickup_lat": 39.900,
                "delivery_lng": 116.410,
                "delivery_lat": 39.910,
            }
            for i in range(30)
        ]
    )

    calibration = estimate_rider_calibration(tasks)

    assert calibration.initial_load_lambda < 1.0


def test_generate_riders_accepts_calibration_parameters():
    merchants = pd.DataFrame(
        [
            {"wm_poi_id": "m1", "lng": 116.40, "lat": 39.92},
            {"wm_poi_id": "m2", "lng": 116.41, "lat": 39.93},
        ]
    )
    calibration = estimate_rider_calibration(
        pd.DataFrame(
            [
                {
                    "courier_id": "r1",
                    "accept_time": 0,
                    "finish_time": 25,
                    "pickup_lng": 116.400,
                    "pickup_lat": 39.920,
                    "delivery_lng": 116.415,
                    "delivery_lat": 39.930,
                }
            ]
        )
    )

    riders = generate_riders(merchants, n_riders=8, seed=11, calibration=calibration)

    assert len(riders) == 8
    assert riders["speed_kmph"].mean() > 0
    assert riders["service_minutes"].mean() > 0


def test_run_simulation_accepts_rider_calibration(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    make_mock_trd(raw, seed=12, users=24, merchants=18, foods=42)
    preprocess(raw, processed, sample_orders=260, seed=12)
    data = PreparedData.load(processed)
    calibration = estimate_rider_calibration(
        pd.DataFrame(
            [
                {
                    "courier_id": "r1",
                    "accept_time": 0,
                    "finish_time": 22,
                    "pickup_lng": 116.400,
                    "pickup_lat": 39.920,
                    "delivery_lng": 116.410,
                    "delivery_lat": 39.928,
                }
            ]
        )
    )

    result = run_simulation(data, seed=12, requests_per_step=4, steps=2, top_k=5, rider_calibration=calibration)

    assert not result.empty
    assert "rider_speed_kmph" in result.columns
    assert result["rider_speed_kmph"].gt(0).all()


def _make_tasks(durations: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "courier_id": f"r{i % 3}",
                "accept_time": i * 10,
                "finish_time": i * 10 + duration,
                "pickup_lng": 116.400 + 0.001 * i,
                "pickup_lat": 39.900,
                "delivery_lng": 116.400 + 0.001 * i + 0.03,
                "delivery_lat": 39.920,
            }
            for i, duration in enumerate(durations)
        ]
    )


def test_reliability_uses_explicit_sla_threshold_not_sample_quantile():
    fast = estimate_rider_calibration(_make_tasks([20.0] * 12), sla_minutes=45.0)
    slow = estimate_rider_calibration(_make_tasks([80.0] * 12), sla_minutes=45.0)

    # 相对样本分位数定义的准时率恒等于分位点，快慢车队会得到相同 reliability；
    # 相对显式 SLA 定义时，全部超时的车队必须得到更低的 reliability。
    assert fast.reliability_mean > slow.reliability_mean
    assert slow.reliability_mean == 0.72


def test_calibration_diagnostics_reports_provenance_and_ks():
    tasks = _make_tasks([18.0, 22.0, 25.0, 30.0, 35.0, 40.0, 44.0, 50.0, 55.0, 60.0, 28.0, 33.0])
    raw = estimate_rider_calibration(tasks)
    applied = adapt_calibration_for_food_delivery(raw)

    diagnostics = calibration_diagnostics(tasks, applied, raw=raw, profile="food-scaled", seed=7)

    assert diagnostics["profile"] == "food-scaled"
    assert diagnostics["parameter_provenance"]["speed_kmph"] == "food-delivery-default"
    assert diagnostics["parameter_provenance"]["initial_load_lambda"] == "task-data-derived"
    assert diagnostics["empirical"]["task_duration_minutes"]["n"] == 12
    duration_fit = diagnostics["lognormal_fit"]["task_duration_minutes"]
    assert duration_fit is not None and 0.0 <= duration_fit["ks_statistic"] <= 1.0
    speed_ks = diagnostics["simulation_vs_data_ks"]["speed_kmph"]
    assert speed_ks is not None and 0.0 <= speed_ks["ks_statistic"] <= 1.0


def test_calibration_diagnostics_raw_profile_marks_speed_as_data_derived():
    tasks = _make_tasks([18.0, 22.0, 25.0, 30.0, 35.0, 40.0, 44.0, 50.0, 55.0, 60.0, 28.0, 33.0])
    raw = estimate_rider_calibration(tasks)

    diagnostics = calibration_diagnostics(tasks, raw, profile="raw", seed=7)

    assert diagnostics["parameter_provenance"]["speed_kmph"] == "task-data-derived"
    assert diagnostics["parameter_provenance"]["service_minutes"] == "task-data-derived"


def test_food_delivery_profile_rescales_lade_like_calibration():
    raw = RiderCalibration(
        speed_kmph=8.0,
        service_minutes=90.0,
        initial_load_lambda=2.5,
        reliability_mean=0.9,
        reliability_std=0.1,
    )

    adapted = adapt_calibration_for_food_delivery(raw)

    assert adapted.speed_kmph == 20.0
    assert adapted.service_minutes == 10.0
    assert adapted.initial_load_lambda == raw.initial_load_lambda
    assert adapted.reliability_mean == raw.reliability_mean
    assert adapted.reliability_std == raw.reliability_std
