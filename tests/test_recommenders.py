from foodflow import recommenders


def test_build_recommenders_uses_unique_name_for_lightgbm_fallback(monkeypatch):
    monkeypatch.setattr(recommenders, "lightgbm_available", lambda: False)

    names = [model.name for model in recommenders.build_recommenders(seed=7)]

    assert len(names) == len(set(names))
    assert "Logistic-LTR" in names


def test_kg_tripartite_recommender_adds_kg_signal(tmp_path):
    from foodflow.data import PreparedData
    from foodflow.mock_data import make_mock_trd
    from foodflow.preprocess import preprocess
    from foodflow.recommenders import KGTripartiteRecommender

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    make_mock_trd(raw, seed=9, users=32, merchants=18, foods=45)
    preprocess(raw, processed, sample_orders=400, seed=9)
    data = PreparedData.load(processed)

    model = KGTripartiteRecommender().fit(data)
    user_id = str(data.orders_train["user_id"].astype(str).iloc[0])
    history_merchant = str(
        data.orders_train[data.orders_train["user_id"].astype(str) == user_id]["wm_poi_id"].astype(str).iloc[0]
    )

    components = model.component_scores(user_id, history_merchant)
    assert "kg_score" in components
    # 历史商家与用户的时间衰减 KG 兴趣必然共享品类/价位节点。
    assert components["kg_score"] > 0.0

    recs, scores = model.recommend_for_user(user_id, k=5)
    assert len(recs) == 5
    assert all(m in scores for m in recs)
