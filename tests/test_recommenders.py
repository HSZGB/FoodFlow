from foodflow import recommenders


def test_build_recommenders_uses_unique_name_for_lightgbm_fallback(monkeypatch):
    monkeypatch.setattr(recommenders, "lightgbm_available", lambda: False)

    names = [model.name for model in recommenders.build_recommenders(seed=7)]

    assert len(names) == len(set(names))
    assert "Seq-Tuned (LightGBM fallback)" in names
