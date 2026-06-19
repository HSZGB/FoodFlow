from pathlib import Path

import pandas as pd

from foodflow.data import PreparedData
from foodflow.mock_data import make_mock_trd
from foodflow.preprocess import preprocess
from foodflow.recommenders import SessionSpuTripartiteRecommender
from foodflow.simulator import run_simulation


def test_preprocess_loads_optional_session_and_spu_signals(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    make_mock_trd(raw, seed=100, users=18, merchants=14, foods=32)

    preprocess(raw, processed, sample_orders=260, seed=100)
    data = PreparedData.load(processed)

    assert not data.session_interactions.empty
    assert {"wm_order_id", "user_id", "wm_poi_id", "rank"}.issubset(data.session_interactions.columns)
    assert not data.order_spus_train.empty
    assert {"wm_order_id", "user_id", "wm_poi_id", "wm_food_spu_id"}.issubset(data.order_spus_train.columns)


def test_session_spu_recommender_promotes_recent_clicked_merchants(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    make_mock_trd(raw, seed=101, users=20, merchants=16, foods=40)
    preprocess(raw, processed, sample_orders=280, seed=101)
    data = PreparedData.load(processed)

    session_row = data.session_interactions.iloc[0]
    user_id = str(session_row["user_id"])
    clicked = data.session_interactions[data.session_interactions["user_id"].astype(str) == user_id]
    clicked_ids = set(clicked["wm_poi_id"].astype(str))

    model = SessionSpuTripartiteRecommender().fit(data)
    recs = model.recommend([user_id], 8, {user_id: "lunch"}).recommendations[user_id]

    assert clicked_ids
    assert clicked_ids & set(recs)


def test_default_simulation_includes_session_spu_tripartite_policy(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    make_mock_trd(raw, seed=102, users=24, merchants=18, foods=42)
    preprocess(raw, processed, sample_orders=260, seed=102)
    data = PreparedData.load(processed)

    result = run_simulation(data, seed=102, requests_per_step=4, steps=2, top_k=5)

    assert "Session-SPU-Tripartite" in set(result["policy"])
