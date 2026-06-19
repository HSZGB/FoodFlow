import pandas as pd

from foodflow.frontier import build_tripartite_frontier, is_pareto_frontier
from foodflow.metrics import (
    category_jsd_at_k,
    evaluate_recommendations,
    gini,
    hitrate_at_k,
    jensen_shannon_divergence,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from foodflow.recommenders import _normalize_tripartite_components


def test_ranking_metrics():
    recs = ["a", "b", "c", "d"]
    truth = {"b", "d"}
    assert recall_at_k(recs, truth, 2) == 0.5
    assert hitrate_at_k(recs, truth, 2) == 1.0
    assert mrr_at_k(recs, truth, 3) == 0.5
    assert 0.0 < ndcg_at_k(recs, truth, 4) <= 1.0


def test_gini_edges():
    assert gini([0, 0, 0]) == 0.0
    assert gini([1, 1, 1]) == 0.0
    assert gini([0, 0, 10]) > 0.6


def test_category_calibration_metric():
    assert jensen_shannon_divergence({"a": 1.0}, {"a": 1.0}) == 0.0
    left_right = jensen_shannon_divergence({"a": 0.8, "b": 0.2}, {"a": 0.2, "b": 0.8})
    right_left = jensen_shannon_divergence({"a": 0.2, "b": 0.8}, {"a": 0.8, "b": 0.2})
    assert left_right == right_left
    category_by_merchant = {"m1": "food", "m2": "food", "m3": "drink", "m4": "drink"}
    assert category_jsd_at_k(["m1", "m2"], ["m1", "m2"], category_by_merchant, 2) == 0.0
    assert category_jsd_at_k(["m3", "m4"], ["m1", "m2"], category_by_merchant, 2) > 0.0

    merchants = pd.DataFrame(
        [
            {"wm_poi_id": "m1", "order_count": 10, "primary_first_tag_id": "food"},
            {"wm_poi_id": "m2", "order_count": 5, "primary_first_tag_id": "food"},
            {"wm_poi_id": "m3", "order_count": 2, "primary_first_tag_id": "drink"},
        ]
    )
    metrics = evaluate_recommendations(
        {"u1": ["m1", "m3"]},
        {"u1": {"m1"}},
        merchants,
        [2],
        {"u1": ["m1", "m2"]},
    )
    assert "CategoryJSD@2" in metrics
    assert metrics["CategoryJSD@2"] > 0.0


def test_repeat_and_explore_recall_segments():
    merchants = pd.DataFrame(
        [
            {"wm_poi_id": "m1", "order_count": 10},
            {"wm_poi_id": "m2", "order_count": 5},
            {"wm_poi_id": "m3", "order_count": 2},
        ]
    )
    metrics = evaluate_recommendations(
        {
            "u1": ["m1", "m3"],
            "u2": ["m3", "m1"],
        },
        {
            "u1": {"m1", "m2"},
            "u2": {"m3"},
        },
        merchants,
        [2],
        {
            "u1": ["m1", "m4"],
            "u2": ["m5"],
        },
    )
    assert metrics["RepeatRecall@2"] == 1.0
    assert metrics["ExploreRecall@2"] == 0.5


def test_tripartite_frontier_marks_dominated_rows():
    points = pd.DataFrame(
        [
            {"name": "strong", "recall": 0.5, "utility": 0.6, "gini": 0.2},
            {"name": "dominated", "recall": 0.4, "utility": 0.5, "gini": 0.3},
            {"name": "tradeoff", "recall": 0.6, "utility": 0.45, "gini": 0.25},
        ]
    )
    mask = is_pareto_frontier(points, maximize=["recall", "utility"], minimize=["gini"])
    assert mask.tolist() == [True, False, True]

    offline = pd.DataFrame(
        [
            {"model": "UserOnly", "Recall@20": 0.4, "NDCG@20": 0.3, "ExposureGini": 0.8, "Coverage@20": 0.2},
            {
                "model": "Seq-xQuAD-Tripartite",
                "Recall@20": 0.35,
                "NDCG@20": 0.28,
                "ExposureGini": 0.7,
                "Coverage@20": 0.25,
            },
            {
                "model": "LightGBM-LTR",
                "Recall@20": 0.42,
                "NDCG@20": 0.32,
                "ExposureGini": 0.75,
                "Coverage@20": 0.22,
            },
        ]
    )
    simulation = pd.DataFrame(
        [
            {
                "policy": "UserOnly + MinETA",
                "avg_eta": 55.0,
                "timeout_rate": 0.6,
                "on_time_rate": 0.4,
                "user_satisfaction": 0.8,
                "platform_utility": 0.45,
            },
            {
                "policy": "Seq-xQuAD-Tripartite + Greedy",
                "avg_eta": 50.0,
                "timeout_rate": 0.5,
                "on_time_rate": 0.5,
                "user_satisfaction": 0.75,
                "platform_utility": 0.5,
            },
            {
                "policy": "Seq-xQuAD-Tripartite + Batch",
                "avg_eta": 48.0,
                "timeout_rate": 0.45,
                "on_time_rate": 0.55,
                "user_satisfaction": 0.76,
                "platform_utility": 0.53,
            },
            {
                "policy": "LightGBM-LTR + MinETA",
                "avg_eta": 54.0,
                "timeout_rate": 0.55,
                "on_time_rate": 0.45,
                "user_satisfaction": 0.82,
                "platform_utility": 0.49,
            },
        ]
    )
    frontier = build_tripartite_frontier(offline, simulation)
    assert {"policy", "model", "is_frontier"}.issubset(frontier.columns)
    assert "LightGBM-LTR + MinETA" in set(frontier["policy"])
    assert "Seq-xQuAD-Tripartite + Batch" in set(frontier["policy"])
    assert frontier["is_frontier"].any()


def test_tripartite_component_normalization():
    rows = {
        "m1": {"user_score": 10.0, "merchant_fairness": 0.2, "eta_score": 0.5, "supply_score": 5.0, "final_score": 0.0},
        "m2": {"user_score": 20.0, "merchant_fairness": 0.8, "eta_score": 0.5, "supply_score": 9.0, "final_score": 0.0},
    }
    normalized = _normalize_tripartite_components(
        rows,
        {"user_score": 1.0, "merchant_fairness": 1.0, "eta_score": 1.0, "supply_score": 1.0},
    )
    assert normalized["m1"]["user_score_norm"] == 0.0
    assert normalized["m2"]["user_score_norm"] == 1.0
    assert normalized["m1"]["eta_score_norm"] == 0.0
    assert normalized["m2"]["final_score"] > normalized["m1"]["final_score"]
