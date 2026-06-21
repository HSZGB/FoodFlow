#!/usr/bin/env python3
"""Create a small TRD-compatible dataset for fast local verification."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_BUCKETS = ["<29", "[29,36)", "[36,49)", "[49,65)", ">=65"]
PERIODS = [0, 1, 2, 3, 4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_demo"))
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--pois", type=int, default=90)
    parser.add_argument("--spus", type=int, default=250)
    parser.add_argument("--train-orders", type=int, default=1800)
    parser.add_argument("--test-orders", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    users = []
    for user_id in range(args.users):
        base = rng.randrange(len(PRICE_BUCKETS))
        users.append(
            {
                "user_id": user_id,
                "avg_pay_amt": PRICE_BUCKETS[base],
                "avg_pay_amt_weekdays": PRICE_BUCKETS[max(0, base - rng.randrange(2))],
                "avg_pay_amt_weekends": PRICE_BUCKETS[min(len(PRICE_BUCKETS) - 1, base + rng.randrange(2))],
            }
        )
    pd.DataFrame(users).to_csv(args.raw_dir / "users.txt", sep="\t", index=False)

    pois = []
    poi_topics = {}
    for poi_id in range(args.pois):
        topic = poi_id % 9
        poi_topics[poi_id] = topic
        score = 4.35 + 0.6 * rng.random()
        pois.append(
            {
                "wm_poi_id": poi_id,
                "wm_poi_name": f"poi-{poi_id}",
                "primary_second_tag_name": topic,
                "primary_third_tag_name": topic * 10 + rng.randrange(4),
                "primary_first_tag_name": topic // 3,
                "poi_brand_id": poi_id % 20,
                "aor_id": poi_id % 6,
                "poi_score": round(score, 2),
                "delivery_comment_avg_score": round(min(5.0, score + 0.1 * rng.random()), 2),
                "food_comment_avg_score": round(min(5.0, score + 0.1 * rng.random()), 2),
            }
        )
    pd.DataFrame(pois).to_csv(args.raw_dir / "pois.txt", sep="\t", index=False)

    spus = []
    for spu_id in range(args.spus):
        topic = spu_id % 9
        ingredients = [topic * 5 + i for i in np_rng.choice(5, size=2, replace=False)]
        tastes = [topic % 5]
        spus.append(
            {
                "wm_food_spu_id": spu_id,
                "wm_food_spu_name": f"food-{spu_id}",
                "price": float(8 + (spu_id % 30)),
                "category": str([topic, topic + 20]),
                "ingredients": str(ingredients),
                "taste": str(tastes),
                "stand_food_id": spu_id % 120,
                "stand_food_name": f"standard-{spu_id % 120}",
            }
        )
    pd.DataFrame(spus).to_csv(args.raw_dir / "spus.txt", sep="\t", index=False)

    user_topics = {u: [u % 9, (u + 1) % 9] for u in range(args.users)}

    def sample_poi(user_id: int) -> int:
        if rng.random() < 0.78:
            topic = rng.choice(user_topics[user_id])
            candidates = [p for p, t in poi_topics.items() if t == topic]
            return rng.choice(candidates)
        return rng.randrange(args.pois)

    train_rows = []
    sessions = []
    spu_orders = []
    ts0 = 1614556800
    for order_id in range(args.train_orders):
        user_id = rng.randrange(args.users)
        poi_id = sample_poi(user_id)
        topic = poi_topics[poi_id]
        dt_day = 1 + order_id // max(1, args.train_orders // 21)
        timestamp = ts0 + dt_day * 86400 + rng.randrange(86400)
        train_rows.append(
            {
                "user_id": user_id,
                "wm_order_id": order_id,
                "wm_poi_id": poi_id,
                "aor_id": poi_id % 6,
                "order_price_interval": PRICE_BUCKETS[(topic + rng.randrange(2)) % len(PRICE_BUCKETS)],
                "order_timestamp": timestamp,
                "ord_period_name": rng.choice(PERIODS),
                "order_scene_name": rng.randrange(3),
                "aoi_id": rng.randrange(200),
                "takedlvr_aoi_type_name": rng.randrange(3),
                "dt": int(f"202103{min(21, dt_day):02d}"),
            }
        )
        click_len = rng.randrange(1, 8)
        click_pois = [sample_poi(user_id) for _ in range(click_len - 1)] + [poi_id]
        sessions.append({"wm_order_id": order_id, "clicks": "#".join(map(str, click_pois)), "dt": train_rows[-1]["dt"]})
        for _ in range(rng.randrange(1, 4)):
            spu = topic + 9 * rng.randrange(max(1, args.spus // 9))
            spu_orders.append({"wm_order_id": order_id, "wm_food_spu_id": spu % args.spus, "dt": train_rows[-1]["dt"]})

    test_rows = []
    test_labels = []
    offset = args.train_orders
    for j in range(args.test_orders):
        order_id = offset + j
        user_id = rng.randrange(args.users)
        poi_id = sample_poi(user_id)
        dt_day = 22 + j // max(1, args.test_orders // 7)
        timestamp = ts0 + dt_day * 86400 + rng.randrange(86400)
        test_rows.append(
            {
                "user_id": user_id,
                "wm_order_id": order_id,
                "aor_id": poi_id % 6,
                "order_timestamp": timestamp,
                "ord_period_name": rng.choice(PERIODS),
                "aoi_id": rng.randrange(200),
                "takedlvr_aoi_type_name": rng.randrange(3),
                "dt": int(f"202103{min(28, dt_day):02d}"),
            }
        )
        test_labels.append({"user_id": user_id, "wm_order_id": order_id, "wm_poi_id": poi_id, "dt": test_rows[-1]["dt"]})
        click_pois = [sample_poi(user_id) for _ in range(rng.randrange(1, 6))] + [poi_id]
        sessions.append({"wm_order_id": order_id, "clicks": "#".join(map(str, click_pois)), "dt": test_rows[-1]["dt"]})

    pd.DataFrame(train_rows).to_csv(args.raw_dir / "orders_train.txt", sep="\t", index=False)
    pd.DataFrame(test_rows).to_csv(args.raw_dir / "orders_test_poi.txt", sep="\t", index=False)
    pd.DataFrame(test_labels).to_csv(args.raw_dir / "orders_poi_test_label.txt", sep="\t", index=False)
    pd.DataFrame(sessions).to_csv(args.raw_dir / "orders_poi_session.txt", sep="\t", index=False)
    pd.DataFrame(spu_orders).to_csv(args.raw_dir / "orders_spu_train.txt", sep="\t", index=False)
    pd.DataFrame(test_rows).to_csv(args.raw_dir / "orders_test_spu.txt", sep="\t", index=False)
    pd.DataFrame(spu_orders[: max(1, len(spu_orders) // 5)]).to_csv(args.raw_dir / "orders_spu_test_label.txt", sep="\t", index=False)

    print(f"Wrote demo data to {args.raw_dir}")


if __name__ == "__main__":
    main()
