from pathlib import Path

from foodflow.data import PreparedData
from foodflow.demo_support import (
    build_recommendation_frame,
    build_peak_trace,
    build_rider_policy_frame,
    demo_user_cases,
    streamlit_image_width_kwargs,
)
from foodflow.mock_data import make_mock_trd
from foodflow.preprocess import preprocess
from foodflow.recommenders import OursFullRecommender, UserOnlyRecommender
from foodflow.rider_sim import generate_riders


def test_streamlit_image_width_kwargs_compatibility():
    def old_image(image, use_column_width=None):
        return image, use_column_width

    def new_image(image, use_container_width=False):
        return image, use_container_width

    assert streamlit_image_width_kwargs(old_image) == {"use_column_width": True}
    assert streamlit_image_width_kwargs(new_image) == {"use_container_width": True}


def test_demo_recommendation_and_rider_frames(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    make_mock_trd(raw, seed=456, users=24, merchants=16, foods=40)
    preprocess(raw, processed, sample_orders=240, seed=456)

    data = PreparedData.load(processed)
    cases = demo_user_cases(data.users)
    assert cases
    assert set(cases.values()).issubset(set(data.user_ids))

    model = OursFullRecommender().fit(data)
    user_model = UserOnlyRecommender().fit(data)
    user_id = data.user_ids[0]
    recs = model.recommend([user_id], 5, {user_id: "lunch"}).recommendations[user_id]
    rec_frame = build_recommendation_frame(data, model, user_id, recs, "lunch")
    user_recs = user_model.recommend([user_id], 5, {user_id: "lunch"}).recommendations[user_id]
    user_frame = build_recommendation_frame(data, user_model, user_id, user_recs, "lunch")

    assert len(rec_frame) == 5
    assert {"merchant_name", "final_score", "reason", "eta_minutes"}.issubset(rec_frame.columns)
    assert len(user_frame) == 5
    assert user_frame["fairness_contrib"].eq(0).all()
    assert user_frame["eta_contrib"].eq(0).all()

    users = data.users.set_index("user_id", drop=False)
    merchants = data.merchants.set_index("wm_poi_id", drop=False)
    riders = generate_riders(data.merchants, n_riders=8, seed=456)
    rider_frame = build_rider_policy_frame(users.loc[user_id], merchants.loc[recs[0]], riders, "lunch")

    assert set(rider_frame["policy_key"]) == {"nearest", "min_eta", "load_aware"}
    assert rider_frame["eta"].min() > 0

    trace = build_peak_trace(
        data,
        {
            "UserOnly + MinETA": (user_model, "min_eta"),
            "Ours-Full": (model, "load_aware"),
        },
        seed=456,
        steps=3,
        requests_per_step=4,
        top_k=5,
    )
    assert set(trace["policy"]) == {"UserOnly + MinETA", "Ours-Full"}
    assert trace["step"].max() == 3
    assert trace["completed_orders"].max() > 0
