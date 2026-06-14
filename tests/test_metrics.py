import pandas as pd

from foodflow.frontier import build_tripartite_frontier, is_pareto_frontier
from foodflow.metrics import gini, hitrate_at_k, mrr_at_k, ndcg_at_k, recall_at_k


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
            {"model": "Ours-Full", "Recall@20": 0.35, "NDCG@20": 0.28, "ExposureGini": 0.7, "Coverage@20": 0.25},
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
                "policy": "Ours-Full",
                "avg_eta": 50.0,
                "timeout_rate": 0.5,
                "on_time_rate": 0.5,
                "user_satisfaction": 0.75,
                "platform_utility": 0.5,
            },
        ]
    )
    frontier = build_tripartite_frontier(offline, simulation)
    assert {"policy", "model", "is_frontier"}.issubset(frontier.columns)
    assert frontier["is_frontier"].any()
