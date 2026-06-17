import pandas as pd

from foodflow.rider_sim import assign_order, assign_orders_batch, estimate_order_eta, generate_riders


def test_eta_positive_and_assignment():
    user = pd.Series({"lng": 116.40, "lat": 39.91})
    merchant = pd.Series({"lng": 116.41, "lat": 39.92, "food_comment_avg_score": 4.5})
    rider = pd.Series({"lng": 116.405, "lat": 39.915, "load": 0, "available_at": 0, "reliability": 0.95})
    eta = estimate_order_eta(user, merchant, rider, "lunch", 0)
    assert eta > 0
    riders = pd.DataFrame(
        [
            {"rider_id": "r1", "lng": 116.405, "lat": 39.915, "load": 0, "available_at": 0, "reliability": 0.95},
            {"rider_id": "r2", "lng": 116.50, "lat": 39.99, "load": 1, "available_at": 5, "reliability": 0.80},
        ]
    )
    rider_id, assigned_eta = assign_order(user, merchant, riders, "load_aware", "lunch", 0)
    assert rider_id in {"r1", "r2"}
    assert assigned_eta > 0


def test_batch_assignment_matches_multiple_orders():
    users = [
        pd.Series({"user_id": "u1", "lng": 116.40, "lat": 39.91}),
        pd.Series({"user_id": "u2", "lng": 116.45, "lat": 39.94}),
    ]
    merchants = [
        pd.Series({"wm_poi_id": "m1", "lng": 116.41, "lat": 39.92, "food_comment_avg_score": 4.5}),
        pd.Series({"wm_poi_id": "m2", "lng": 116.46, "lat": 39.95, "food_comment_avg_score": 4.4}),
    ]
    riders = pd.DataFrame(
        [
            {
                "rider_id": "r1",
                "lng": 116.405,
                "lat": 39.915,
                "load": 0,
                "available_at": 0,
                "reliability": 0.95,
                "speed_kmh": 22.0,
                "service_radius_km": 7.0,
                "acceptance_rate": 0.95,
            },
            {
                "rider_id": "r2",
                "lng": 116.455,
                "lat": 39.945,
                "load": 0,
                "available_at": 0,
                "reliability": 0.92,
                "speed_kmh": 21.0,
                "service_radius_km": 7.0,
                "acceptance_rate": 0.90,
            },
        ]
    )
    orders = [
        {"order_id": "o1", "user_row": users[0], "merchant_row": merchants[0]},
        {"order_id": "o2", "user_row": users[1], "merchant_row": merchants[1]},
    ]
    assignments = assign_orders_batch(orders, riders, "lunch", 0)
    assert len(assignments) == 2
    assert len({a["rider_id"] for a in assignments}) == 2
    assert all(float(a["eta"]) > 0 for a in assignments)


def test_batch_assignment_uses_rider_capacity_slots():
    users = [
        pd.Series({"user_id": "u1", "lng": 116.40, "lat": 39.91}),
        pd.Series({"user_id": "u2", "lng": 116.401, "lat": 39.911}),
    ]
    merchant = pd.Series({"wm_poi_id": "m1", "lng": 116.41, "lat": 39.92, "food_comment_avg_score": 4.5})
    riders = pd.DataFrame(
        [
            {
                "rider_id": "r1",
                "lng": 116.405,
                "lat": 39.915,
                "load": 1,
                "available_at": 0,
                "reliability": 0.95,
                "speed_kmh": 22.0,
                "service_radius_km": 7.0,
                "acceptance_rate": 0.95,
            }
        ]
    )
    orders = [
        {"order_id": "o1", "user_row": users[0], "merchant_row": merchant},
        {"order_id": "o2", "user_row": users[1], "merchant_row": merchant},
    ]
    assignments = assign_orders_batch(orders, riders, "lunch", 0)
    assert len(assignments) == 2
    assert {a["rider_id"] for a in assignments} == {"r1"}
    assert {a["slot_number"] for a in assignments} == {0, 1}


def test_generated_riders_include_operational_fields():
    merchants = pd.DataFrame(
        [
            {"wm_poi_id": "m1", "lng": 116.40, "lat": 39.91},
            {"wm_poi_id": "m2", "lng": 116.45, "lat": 39.94},
        ]
    )
    riders = generate_riders(merchants, n_riders=4, seed=7)
    assert {"speed_kmh", "service_radius_km", "acceptance_rate"}.issubset(riders.columns)
    assert riders["acceptance_rate"].between(0.0, 1.0).all()
