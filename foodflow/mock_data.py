from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import ensure_dir, write_csv


PERIODS = ["breakfast", "lunch", "dinner", "night"]
SCENES = ["search", "home_feed", "coupon", "history"]


def make_mock_trd(raw_dir: Path, seed: int = 42, users: int = 80, merchants: int = 45, foods: int = 120) -> None:
    rng = np.random.default_rng(seed)
    raw_dir = ensure_dir(raw_dir)

    user_ids = [f"u{i:04d}" for i in range(users)]
    merchant_ids = [f"m{i:04d}" for i in range(merchants)]
    food_ids = [f"f{i:04d}" for i in range(foods)]
    aor_ids = [f"aor{i:02d}" for i in range(12)]
    aoi_ids = [f"aoi{i:02d}" for i in range(18)]
    categories = [f"cat{i:02d}" for i in range(8)]

    user_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "avg_pay_amt": rng.normal(36, 10, users).clip(12, 90).round(2),
            "avg_pay_amt_weekdays": rng.normal(34, 9, users).clip(10, 85).round(2),
            "avg_pay_amt_weekends": rng.normal(39, 12, users).clip(12, 100).round(2),
        }
    )

    poi_category = rng.choice(categories, merchants)
    poi_df = pd.DataFrame(
        {
            "wm_poi_id": merchant_ids,
            "wm_poi_name": [f"FoodFlow Merchant {i}" for i in range(merchants)],
            "primary_first_tag_id": poi_category,
            "primary_second_tag_id": [f"{c}_sub{rng.integers(0, 3)}" for c in poi_category],
            "primary_third_tag_id": [f"{c}_leaf{rng.integers(0, 5)}" for c in poi_category],
            "poi_brand_id": [f"brand{rng.integers(0, 14):02d}" for _ in range(merchants)],
            "aor_id": rng.choice(aor_ids, merchants),
            "poi_score": rng.normal(4.35, 0.28, merchants).clip(3.3, 4.9).round(2),
            "delivery_comment_avg_score": rng.normal(4.28, 0.32, merchants).clip(3.1, 4.9).round(2),
            "food_comment_avg_score": rng.normal(4.32, 0.30, merchants).clip(3.0, 4.9).round(2),
        }
    )

    spu_df = pd.DataFrame(
        {
            "wm_food_spu_id": food_ids,
            "wm_food_spu_name": [f"Dish {i}" for i in range(foods)],
            "price": rng.normal(28, 12, foods).clip(6, 100).round(2),
            "category": rng.choice(categories, foods),
            "ingredients": rng.choice(["chicken", "beef", "rice", "noodle", "tea", "vegetable"], foods),
            "taste": rng.choice(["spicy", "sweet", "fresh", "salty", "light"], foods),
            "standfood_id": [f"sf{i:04d}" for i in range(foods)],
            "standfood_name": [f"Standard Dish {i}" for i in range(foods)],
        }
    )

    merchant_foods = {
        m: rng.choice(food_ids, size=rng.integers(5, 15), replace=False).tolist() for m in merchant_ids
    }
    merchant_bias = rng.power(2.2, merchants)
    merchant_bias = merchant_bias / merchant_bias.sum()
    user_home = {u: rng.choice(aoi_ids) for u in user_ids}
    user_cat_pref = {u: rng.choice(categories, size=3, replace=False).tolist() for u in user_ids}

    def sample_order(idx: int, day: int) -> dict:
        u = rng.choice(user_ids)
        preferred = [m for m, cat in zip(merchant_ids, poi_category) if cat in user_cat_pref[u]]
        if rng.random() < 0.72 and preferred:
            m = rng.choice(preferred)
        else:
            m = rng.choice(merchant_ids, p=merchant_bias)
        f = rng.choice(merchant_foods[m])
        period = rng.choice(PERIODS, p=[0.15, 0.42, 0.35, 0.08])
        timestamp = 1614556800 + day * 86400 + int(rng.integers(8 * 3600, 22 * 3600))
        return {
            "wm_order_id": f"o{idx:06d}",
            "wm_food_spu_id": f,
            "user_id": u,
            "wm_poi_id": m,
            "aor_id": poi_df.loc[poi_df["wm_poi_id"] == m, "aor_id"].iloc[0],
            "order_price": float(spu_df.loc[spu_df["wm_food_spu_id"] == f, "price"].iloc[0] + rng.normal(5, 3)),
            "order_timestamp": timestamp,
            "ord_period_name": period,
            "order_scene_name": rng.choice(SCENES),
            "aoi_id": user_home[u],
            "takedlvr_aoi_type_name": rng.choice(["home", "office", "school"], p=[0.5, 0.4, 0.1]),
            "dt": f"2021-03-{day + 1:02d}",
        }

    train = pd.DataFrame([sample_order(i, int(rng.integers(0, 21))) for i in range(1500)])
    test = pd.DataFrame([sample_order(2000 + i, int(rng.integers(21, 28))) for i in range(280)])
    test_label = test[["wm_order_id", "user_id", "wm_poi_id"]].copy()

    session = pd.concat([train, test], ignore_index=True)[["wm_order_id", "dt"]].copy()
    session["clicks"] = [
        "#".join(rng.choice(merchant_ids, size=5, replace=False).tolist()) for _ in range(len(session))
    ]

    spu_train = train[["wm_order_id", "wm_food_spu_id"]].copy()
    spu_test = test[["wm_order_id", "wm_food_spu_id"]].copy()

    write_csv(user_df, raw_dir / "users.txt")
    write_csv(poi_df, raw_dir / "pois.txt")
    write_csv(spu_df, raw_dir / "spus.txt")
    write_csv(train, raw_dir / "orders_train.txt")
    write_csv(test, raw_dir / "orders_test_poi.txt")
    write_csv(test_label, raw_dir / "orders_poi_test_label.txt")
    write_csv(spu_train, raw_dir / "orders_spu_train.txt")
    write_csv(spu_test, raw_dir / "orders_test_spu.txt")
    write_csv(spu_test.rename(columns={"wm_food_spu_id": "label_spu_id"}), raw_dir / "orders_spu_test_label.txt")
    write_csv(session, raw_dir / "orders_poi_session.txt")
    (raw_dir / "README.md").write_text(
        "Mock TRD-like dataset generated for FoodFlow smoke tests. Real source: Zenodo 10.5281/zenodo.8025855.\n",
        encoding="utf-8",
    )
    (raw_dir / "MOCK_DATASET").write_text("mock TRD-like smoke dataset\n", encoding="utf-8")
