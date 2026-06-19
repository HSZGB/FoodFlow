from pathlib import Path

from foodflow.data import PreparedData
from foodflow.explain import explain_recommendation
from foodflow.kg import build_lightweight_triples, kg_explanation_parts, kg_path_summary
from foodflow.mock_data import make_mock_trd
from foodflow.preprocess import preprocess


def test_lightweight_kg_paths_and_explanation(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    make_mock_trd(raw, seed=789, users=18, merchants=12, foods=30)
    preprocess(raw, processed, sample_orders=220, seed=789)
    data = PreparedData.load(processed)

    triples = build_lightweight_triples(data, user_limit=5, max_history_per_user=8)
    assert {"head", "relation", "tail", "evidence"}.issubset(triples.columns)
    assert {"has_category", "located_in_area", "has_price_range", "ordered_poi"}.issubset(set(triples["relation"]))

    order = data.orders_train.iloc[0]
    user_id = str(order["user_id"])
    merchant_id = str(order["wm_poi_id"])
    summary = kg_path_summary(data, user_id, merchant_id)
    assert summary.repeat_orders >= 1
    assert summary.paths
    assert summary.triples
    parts = kg_explanation_parts(summary)
    assert parts
    assert any("KG路径" in part or "证据路径" in part for part in parts)

    explanation = explain_recommendation(data, user_id, merchant_id)
    assert "证据路径" in explanation
    assert "预计履约时间" in explanation
