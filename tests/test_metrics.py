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
