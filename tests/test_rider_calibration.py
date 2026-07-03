import pandas as pd

from foodflow.rider_data import RiderCalibration, adapt_calibration_for_food_delivery, estimate_rider_calibration
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
