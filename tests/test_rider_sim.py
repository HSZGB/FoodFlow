import pandas as pd

from foodflow.rider_sim import assign_order, assign_orders_batch, estimate_order_eta


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


def test_batch_assignment_uses_distinct_riders():
    user_a = pd.Series({"lng": 116.40, "lat": 39.91})
    user_b = pd.Series({"lng": 116.48, "lat": 39.98})
    merchant_a = pd.Series({"lng": 116.41, "lat": 39.92, "food_comment_avg_score": 4.5})
    merchant_b = pd.Series({"lng": 116.49, "lat": 39.99, "food_comment_avg_score": 4.4})
    riders = pd.DataFrame(
        [
            {"rider_id": "r1", "lng": 116.405, "lat": 39.915, "load": 0, "available_at": 0, "reliability": 0.95},
            {"rider_id": "r2", "lng": 116.50, "lat": 39.99, "load": 0, "available_at": 0, "reliability": 0.92},
        ]
    )
    orders = [
        {
            "order_id": "o1",
            "user_id": "u1",
            "merchant_id": "m1",
            "user_row": user_a,
            "merchant_row": merchant_a,
        },
        {
            "order_id": "o2",
            "user_id": "u2",
            "merchant_id": "m2",
            "user_row": user_b,
            "merchant_row": merchant_b,
        },
    ]
    assignments = assign_orders_batch(orders, riders, "min_eta", "lunch", 0)
    assert len(assignments) == 2
    assert assignments["rider_id"].nunique() == 2
    assert assignments["eta"].min() > 0
