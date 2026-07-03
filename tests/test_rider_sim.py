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
                "speed_kmph": 22.0,
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
                "speed_kmph": 21.0,
                "service_radius_km": 7.0,
                "acceptance_rate": 0.90,
            },
        ]
    )
    orders = [
        {"order_id": "o1", "user_row": users[0], "merchant_row": merchants[0]},
        {"order_id": "o2", "user_row": users[1], "merchant_row": merchants[1]},
    ]
    assignments = assign_orders_batch(orders, riders, "load_aware", "lunch", 0)
    assert len(assignments) == 2
    assert assignments["rider_id"].nunique() == 2
    assert assignments["eta"].min() > 0


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
                "speed_kmph": 22.0,
                "service_radius_km": 7.0,
                "acceptance_rate": 0.95,
            }
        ]
    )
    orders = [
        {"order_id": "o1", "user_row": users[0], "merchant_row": merchant},
        {"order_id": "o2", "user_row": users[1], "merchant_row": merchant},
    ]
    assignments = assign_orders_batch(orders, riders, "load_aware", "lunch", 0)
    assert len(assignments) == 2
    assert set(assignments["rider_id"]) == {"r1"}
    assert set(assignments["slot_number"]) == {0, 1}


def test_generated_riders_include_operational_fields():
    merchants = pd.DataFrame(
        [
            {"wm_poi_id": "m1", "lng": 116.40, "lat": 39.91},
            {"wm_poi_id": "m2", "lng": 116.45, "lat": 39.94},
        ]
    )
    riders = generate_riders(merchants, n_riders=4, seed=7)
    assert {"speed_kmph", "service_radius_km", "acceptance_rate"}.issubset(riders.columns)
    assert riders["acceptance_rate"].between(0.0, 1.0).all()


def test_run_simulation_multi_seed_aggregates_mean_and_ci(tmp_path):
    from foodflow.data import PreparedData
    from foodflow.mock_data import make_mock_trd
    from foodflow.preprocess import preprocess
    from foodflow.simulator import DEFAULT_POLICIES, run_simulation_multi_seed

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    make_mock_trd(raw, seed=21, users=24, merchants=18, foods=42)
    preprocess(raw, processed, sample_orders=260, seed=21)
    data = PreparedData.load(processed)

    result = run_simulation_multi_seed(
        data,
        seeds=[1, 2, 3],
        policies=DEFAULT_POLICIES[:2],
        requests_per_step=4,
        steps=2,
        top_k=5,
    )

    assert len(result) == 2
    assert (result["n_seeds"] == 3).all()
    assert "avg_eta_std" in result.columns
    assert "platform_utility_ci95" in result.columns
    assert result["avg_eta_ci95"].ge(0).all()


def _route_test_riders():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "rider_id": "r_near",
                "lng": 121.30,
                "lat": 37.50,
                "load": 0,
                "available_at": 0,
                "reliability": 0.9,
                "speed_kmph": 20.0,
                "service_radius_km": 6.0,
                "acceptance_rate": 0.9,
                "service_minutes": 8.0,
                "income": 0.0,
                "assigned": 0,
            },
            {
                "rider_id": "r_far",
                "lng": 121.45,
                "lat": 37.62,
                "load": 0,
                "available_at": 0,
                "reliability": 0.9,
                "speed_kmph": 20.0,
                "service_radius_km": 6.0,
                "acceptance_rate": 0.9,
                "service_minutes": 8.0,
                "income": 0.0,
                "assigned": 0,
            },
        ]
    )


def test_route_insertion_prefers_enroute_rider():
    import pandas as pd

    from foodflow.rider_sim import (
        apply_route_assignment,
        assign_order_route,
        ensure_route_column,
        route_pending_orders,
    )

    riders = _route_test_riders()
    ensure_route_column(riders)
    merchant_a = pd.Series({"lng": 121.31, "lat": 37.505, "food_comment_avg_score": 4.5})
    user_a = pd.Series({"lng": 121.33, "lat": 37.52})
    first = assign_order_route(user_a, merchant_a, riders)
    assert first is not None and first["rider_id"] == "r_near"
    apply_route_assignment(
        riders, "r_near", merchant_a, user_a, "o1",
        int(first["insert_pickup"]), int(first["insert_dropoff"]), float(first["eta"]), 0,
    )
    assert route_pending_orders(riders.iloc[0]) == 1

    # 第二单的取送点都在 r_near 当前路径沿线：顺路插入的边际成本应远小于
    # 从零出发的空闲骑手绕行成本，即使 r_far 是"闲"的。
    merchant_b = pd.Series({"lng": 121.315, "lat": 37.508, "food_comment_avg_score": 4.5})
    user_b = pd.Series({"lng": 121.325, "lat": 37.515})
    second = assign_order_route(user_b, merchant_b, riders)
    assert second is not None
    assert second["rider_id"] == "r_near"
    assert bool(second["enroute"]) is True
    assert float(second["detour"]) < 20.0


def test_route_batch_assigns_all_orders_and_advances():
    import pandas as pd

    from foodflow.rider_sim import (
        advance_riders_along_routes,
        assign_orders_route_batch,
        ensure_route_column,
    )

    riders = _route_test_riders()
    ensure_route_column(riders)
    orders = [
        {
            "order_id": f"o{i}",
            "user_id": f"u{i}",
            "merchant_id": f"m{i}",
            "user_row": pd.Series({"lng": 121.31 + 0.01 * i, "lat": 37.51}),
            "merchant_row": pd.Series({"lng": 121.30 + 0.01 * i, "lat": 37.50, "food_comment_avg_score": 4.4}),
        }
        for i in range(3)
    ]
    assignments = assign_orders_route_batch(orders, riders)
    assert len(assignments) == 3
    assert all("detour" in a and a["eta"] > 0 for a in assignments)

    total_load_before = int(riders["load"].sum())
    assert total_load_before == 3
    advance_riders_along_routes(riders, elapsed_minutes=600.0)
    assert int(riders["load"].sum()) == 0
    assert all(len(route) == 0 for route in riders["route"])
