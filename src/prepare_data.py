from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NULL_VALUES = {"NULL", "null", "None", "nan", "NaN", "", "未知", None}
PRICE_TO_IDX = {"<29": 0, "[29,36)": 1, "[36,49)": 2, "[49,65)": 3, ">=65": 4}
SECONDS_PER_DAY = 86400.0


def clean_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in NULL_VALUES:
        return None
    return text


def parse_list_field(value: Any) -> list[str]:
    text = clean_value(value)
    if text is None:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed if clean_value(x) is not None]
    except (ValueError, SyntaxError):
        pass
    return [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]


def parse_clicks(value: Any) -> list[int]:
    text = clean_value(value)
    if text is None:
        return []
    out = []
    for part in text.split("#"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def price_to_float(value: Any) -> float:
    text = clean_value(value)
    if text is None or text not in PRICE_TO_IDX:
        return 0.5
    return PRICE_TO_IDX[text] / 4.0


def score_bin(value: Any) -> str | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x >= 4.85:
        return "excellent"
    if x >= 4.75:
        return "high"
    if x >= 4.60:
        return "medium"
    return "low"


def score_to_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value) / 5.0))
    except (TypeError, ValueError):
        return 0.0


class IdMap:
    def __init__(self, with_pad: bool = False) -> None:
        self.obj_to_idx: dict[str, int] = {}
        self.idx_to_obj: list[str] = []
        if with_pad:
            self.add("__PAD__")

    def add(self, obj: str) -> int:
        if obj not in self.obj_to_idx:
            self.obj_to_idx[obj] = len(self.idx_to_obj)
            self.idx_to_obj.append(obj)
        return self.obj_to_idx[obj]

    def get(self, obj: str) -> int | None:
        return self.obj_to_idx.get(obj)

    def __len__(self) -> int:
        return len(self.idx_to_obj)


def read_tsv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def maybe_sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def normalize_int_id(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=[column])
    out[column] = out[column].astype(int)
    return out


def most_common_price(values: pd.Series) -> str | None:
    counts = Counter(clean_value(x) for x in values)
    counts.pop(None, None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def build_processed(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    raw = Path(args.raw_dir)

    train_orders = read_tsv(raw / "orders_train.txt")
    test_context = read_tsv(raw / "orders_test_poi.txt")
    test_label = read_tsv(raw / "orders_poi_test_label.txt")
    users_df = read_tsv(raw / "users.txt")
    pois_df = read_tsv(raw / "pois.txt")
    sessions_df = read_tsv(raw / "orders_poi_session.txt")

    train_orders = maybe_sample(train_orders, args.max_train_orders, args.seed)
    train_orders = train_orders.sort_values("order_timestamp").reset_index(drop=True)
    test_merged = test_context.merge(
        test_label[["user_id", "wm_order_id", "wm_poi_id", "dt"]],
        on=["user_id", "wm_order_id", "dt"],
        how="inner",
    )
    test_merged = maybe_sample(test_merged, args.max_test_orders, args.seed + 1)
    test_merged = test_merged.sort_values("order_timestamp").reset_index(drop=True)
    pois_df = normalize_int_id(pois_df, "wm_poi_id")
    users_df = normalize_int_id(users_df, "user_id")

    sessions = dict(zip(sessions_df["wm_order_id"].astype(int), sessions_df["clicks"]))

    user_ids = sorted(set(train_orders["user_id"].astype(int)) | set(test_merged["user_id"].astype(int)))
    click_pois: set[int] = set()
    relevant_order_ids = set(train_orders["wm_order_id"].astype(int)) | set(test_merged["wm_order_id"].astype(int))
    for order_id in relevant_order_ids:
        for poi_id in parse_clicks(sessions.get(order_id)):
            click_pois.add(poi_id)

    poi_ids = (
        set(pois_df["wm_poi_id"])
        | set(train_orders["wm_poi_id"].astype(int))
        | set(test_merged["wm_poi_id"].astype(int))
        | click_pois
    )
    poi_ids = sorted(poi_ids)

    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    poi_map = {pid: idx for idx, pid in enumerate(poi_ids)}
    users = [str(uid) for uid in user_ids]
    pois = [str(pid) for pid in poi_ids]

    entity_map = IdMap(with_pad=True)
    relation_map = IdMap(with_pad=True)
    rel_names = [
        "has_cat1",
        "has_cat2",
        "has_cat3",
        "has_brand",
        "located_in_aor",
        "has_price",
        "has_poi_score",
        "has_delivery_score",
        "has_food_score",
        "has_food_category",
        "has_ingredient",
        "has_taste",
        "has_standard_food",
        "pref_hist",
        "pref_click",
    ]
    for name in rel_names:
        relation_map.add(name)

    poi_attrs: list[list[tuple[int, int]]] = [[] for _ in poi_ids]
    triples: list[tuple[str, str, str]] = []

    def add_attr(poi_id: int, rel: str, ent_type: str, value: Any) -> None:
        text = clean_value(value)
        if text is None or poi_id not in poi_map:
            return
        ent = f"{ent_type}:{text}"
        ent_idx = entity_map.add(ent)
        rel_idx = relation_map.add(rel)
        pidx = poi_map[poi_id]
        pair = (ent_idx, rel_idx)
        if pair not in poi_attrs[pidx]:
            poi_attrs[pidx].append(pair)
            triples.append((f"poi:{poi_id}", rel, ent))

    price_by_poi = (
        train_orders.groupby("wm_poi_id")["order_price_interval"].agg(most_common_price).to_dict()
        if "order_price_interval" in train_orders.columns
        else {}
    )

    pois_df = pois_df.copy()
    poi_meta_by_id: dict[int, dict[str, Any]] = {}
    for row in pois_df.to_dict("records"):
        poi_id = int(row["wm_poi_id"])
        poi_meta_by_id[poi_id] = row
        add_attr(poi_id, "has_cat1", "cat1", row.get("primary_first_tag_name"))
        add_attr(poi_id, "has_cat2", "cat2", row.get("primary_second_tag_name"))
        add_attr(poi_id, "has_cat3", "cat3", row.get("primary_third_tag_name"))
        add_attr(poi_id, "has_brand", "brand", row.get("poi_brand_id"))
        add_attr(poi_id, "located_in_aor", "aor", row.get("aor_id"))
        add_attr(poi_id, "has_price", "price", price_by_poi.get(poi_id))
        add_attr(poi_id, "has_poi_score", "poi_score", score_bin(row.get("poi_score")))
        add_attr(poi_id, "has_delivery_score", "delivery_score", score_bin(row.get("delivery_comment_avg_score")))
        add_attr(poi_id, "has_food_score", "food_score", score_bin(row.get("food_comment_avg_score")))

    spus_path = raw / "spus.txt"
    order_spu_path = raw / "orders_spu_train.txt"
    if spus_path.exists() and order_spu_path.exists():
        spus = read_tsv(spus_path)
        order_spu = read_tsv(order_spu_path, usecols=["wm_order_id", "wm_food_spu_id"])
        order_to_poi = train_orders[["wm_order_id", "wm_poi_id"]].drop_duplicates()
        order_spu = order_spu[order_spu["wm_order_id"].isin(set(order_to_poi["wm_order_id"]))]
        merged = order_spu.merge(order_to_poi, on="wm_order_id", how="inner").merge(spus, on="wm_food_spu_id", how="left")
        per_poi_entities: dict[int, Counter[tuple[str, str, str]]] = defaultdict(Counter)
        for row in merged.to_dict("records"):
            poi_id = int(row["wm_poi_id"])
            for value in parse_list_field(row.get("category")):
                per_poi_entities[poi_id][("has_food_category", "food_category", value)] += 1
            for value in parse_list_field(row.get("ingredients")):
                per_poi_entities[poi_id][("has_ingredient", "ingredient", value)] += 1
            for value in parse_list_field(row.get("taste")):
                per_poi_entities[poi_id][("has_taste", "taste", value)] += 1
            stand_food = clean_value(row.get("stand_food_id"))
            if stand_food is not None:
                per_poi_entities[poi_id][("has_standard_food", "standard_food", stand_food)] += 1
        for poi_id, counter in per_poi_entities.items():
            for (rel, ent_type, value), _ in counter.most_common(args.max_food_attrs_per_poi):
                add_attr(poi_id, rel, ent_type, value)

    user_feature = np.zeros((len(user_ids), 3), dtype=np.float32)
    users_df = users_df.copy()
    for row in users_df.to_dict("records"):
        uid = int(row["user_id"])
        if uid not in user_map:
            continue
        user_feature[user_map[uid]] = np.array(
            [
                price_to_float(row.get("avg_pay_amt")),
                price_to_float(row.get("avg_pay_amt_weekdays")),
                price_to_float(row.get("avg_pay_amt_weekends")),
            ],
            dtype=np.float32,
        )

    poi_feature = np.zeros((len(poi_ids), 6), dtype=np.float32)
    poi_aor: list[str | None] = [None for _ in poi_ids]
    poi_price_norm: list[float] = [0.5 for _ in poi_ids]
    max_aor = 1.0
    aor_values = []
    for meta in poi_meta_by_id.values():
        aor = clean_value(meta.get("aor_id"))
        if aor is not None:
            try:
                aor_values.append(float(aor))
            except ValueError:
                pass
    if aor_values:
        max_aor = max(1.0, max(aor_values))

    for pid, pidx in poi_map.items():
        meta = poi_meta_by_id.get(pid, {})
        price_norm = price_to_float(price_by_poi.get(pid))
        poi_price_norm[pidx] = price_norm
        aor = clean_value(meta.get("aor_id"))
        poi_aor[pidx] = aor
        try:
            aor_norm = float(aor) / max_aor if aor is not None else 0.0
        except ValueError:
            aor_norm = 0.0
        poi_feature[pidx] = np.array(
            [
                price_norm,
                score_to_float(meta.get("poi_score")),
                score_to_float(meta.get("delivery_comment_avg_score")),
                score_to_float(meta.get("food_comment_avg_score")),
                aor_norm,
                1.0 if pid in price_by_poi else 0.0,
            ],
            dtype=np.float32,
        )

    def make_basic(
        uidx: int,
        pidx: int,
        row: dict[str, Any],
        ent_ids: list[int] | None = None,
        rel_ids: list[int] | None = None,
        weights: list[float] | None = None,
    ) -> list[float]:
        user_price = float(user_feature[uidx, 0])
        cand_price = float(poi_price_norm[pidx])
        price_match = 1.0 - abs(user_price - cand_price)
        ctx_aor = clean_value(row.get("aor_id"))
        same_aor = 1.0 if ctx_aor is not None and ctx_aor == poi_aor[pidx] else 0.0
        period = int(row.get("ord_period_name", 0)) if str(row.get("ord_period_name", "0")).isdigit() else 0
        period_onehot = [1.0 if period == k else 0.0 for k in range(5)]
        dynamic_match = [0.0] * 6
        if ent_ids is not None and rel_ids is not None and weights is not None:
            attr_entities = {ent for ent, _rel in poi_attrs[pidx]}
            hist_total = click_total = 0.0
            hist_hit = click_hit = 0.0
            hist_count = click_count = 0
            for ent, rel, weight in zip(ent_ids, rel_ids, weights):
                if ent == 0 or weight <= 0:
                    continue
                if rel == hist_rel:
                    hist_total += weight
                    if ent in attr_entities:
                        hist_hit += weight
                        hist_count += 1
                elif rel == click_rel:
                    click_total += weight
                    if ent in attr_entities:
                        click_hit += weight
                        click_count += 1
            total = hist_total + click_total
            hit = hist_hit + click_hit
            dynamic_match = [
                math.log1p(hist_hit),
                math.log1p(click_hit),
                math.log1p(hit),
                hist_hit / hist_total if hist_total > 0 else 0.0,
                click_hit / click_total if click_total > 0 else 0.0,
                (hist_count + click_count) / max(1.0, float(len(attr_entities))),
            ]
        return (
            user_feature[uidx].tolist()
            + poi_feature[pidx].tolist()
            + [price_match, same_aor]
            + period_onehot
            + dynamic_match
        )

    hist_rel = relation_map.get("pref_hist")
    click_rel = relation_map.get("pref_click")
    assert hist_rel is not None and click_rel is not None

    def build_interest(
        history: list[tuple[int, int]],
        clicks: list[int],
        now_ts: int,
    ) -> tuple[list[int], list[int], list[float], list[int]]:
        weights: dict[tuple[int, int], float] = defaultdict(float)
        history_pois = []
        for poi_id, ts in history[-args.max_history :]:
            if poi_id not in poi_map:
                continue
            history_pois.append(poi_map[poi_id])
            delta_days = max(0.0, (now_ts - ts) / SECONDS_PER_DAY)
            decay = math.exp(-args.decay_lambda * delta_days)
            for ent_idx, _rel_idx in poi_attrs[poi_map[poi_id]]:
                weights[(ent_idx, hist_rel)] += decay
        recent_clicks = clicks[-args.max_clicks :]
        for pos, poi_id in enumerate(recent_clicks):
            if poi_id not in poi_map:
                continue
            recency_power = len(recent_clicks) - pos - 1
            weight = args.click_weight * (args.click_decay ** recency_power)
            for ent_idx, _rel_idx in poi_attrs[poi_map[poi_id]]:
                weights[(ent_idx, click_rel)] += weight
        top_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)[: args.max_interests]
        if not top_items:
            return [0], [0], [0.0], history_pois[-args.max_history :]
        ent_ids = [k[0] for k, _ in top_items]
        rel_ids = [k[1] for k, _ in top_items]
        vals = [float(v) for _, v in top_items]
        return ent_ids, rel_ids, vals, history_pois[-args.max_history :]

    all_poi_indices = list(range(len(poi_ids)))
    popularity = np.zeros(len(poi_ids), dtype=np.float32)
    train_user_pos: dict[int, set[int]] = defaultdict(set)
    for row in train_orders.to_dict("records"):
        uidx = user_map[int(row["user_id"])]
        pidx = poi_map[int(row["wm_poi_id"])]
        train_user_pos[uidx].add(pidx)
        popularity[pidx] += 1.0

    def sample_negative(uidx: int, rng: random.Random) -> int:
        positives = train_user_pos.get(uidx, set())
        for _ in range(100):
            cand = rng.choice(all_poi_indices)
            if cand not in positives:
                return cand
        return rng.choice(all_poi_indices)

    rng = random.Random(args.seed)
    user_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    train_instances: list[dict[str, Any]] = []
    for row in train_orders.to_dict("records"):
        uid = int(row["user_id"])
        poi_id = int(row["wm_poi_id"])
        uidx = user_map[uid]
        pidx = poi_map[poi_id]
        now_ts = int(row["order_timestamp"])
        clicks = parse_clicks(sessions.get(int(row["wm_order_id"])))
        ent_ids, rel_ids, weights, history_pois = build_interest(user_history[uidx], clicks, now_ts)
        base = {
            "user": uidx,
            "entity_ids": ent_ids,
            "relation_ids": rel_ids,
            "weights": weights,
            "history_pois": history_pois,
        }
        train_instances.append(
            {
                **base,
                "poi": pidx,
                "label": 1.0,
                "basic": make_basic(uidx, pidx, row, ent_ids, rel_ids, weights),
            }
        )
        for _ in range(args.negatives):
            neg = sample_negative(uidx, rng)
            train_instances.append(
                {
                        **base,
                        "poi": neg,
                        "label": 0.0,
                        "basic": make_basic(uidx, neg, row, ent_ids, rel_ids, weights),
                    }
                )
        user_history[uidx].append((poi_id, now_ts))

    # Use all train history before evaluating the last-week test labels.
    test_queries: list[dict[str, Any]] = []
    candidate_pool = np.arange(len(poi_ids))
    pop_prob = popularity + 1.0
    pop_prob = pop_prob / pop_prob.sum()
    for row in test_merged.to_dict("records"):
        uid = int(row["user_id"])
        poi_id = int(row["wm_poi_id"])
        if uid not in user_map or poi_id not in poi_map:
            continue
        uidx = user_map[uid]
        pidx = poi_map[poi_id]
        now_ts = int(row["order_timestamp"])
        clicks = parse_clicks(sessions.get(int(row["wm_order_id"])))
        ent_ids, rel_ids, weights, history_pois = build_interest(user_history[uidx], clicks, now_ts)
        candidates = [pidx]
        positives = train_user_pos.get(uidx, set()) | {pidx}
        attempts = 0
        while len(candidates) < args.eval_candidates and attempts < args.eval_candidates * 20:
            attempts += 1
            cand = int(np.random.choice(candidate_pool, p=pop_prob))
            if cand not in positives and cand not in candidates:
                candidates.append(cand)
        while len(candidates) < args.eval_candidates:
            cand = int(rng.choice(all_poi_indices))
            if cand not in positives and cand not in candidates:
                candidates.append(cand)
        labels = [1.0] + [0.0] * (len(candidates) - 1)
        basics = [make_basic(uidx, cand, row, ent_ids, rel_ids, weights) for cand in candidates]
        ranked_items = list(zip(candidates, labels, basics))
        rng.shuffle(ranked_items)
        candidates, labels, basics = map(list, zip(*ranked_items))
        test_queries.append(
            {
                "user": uidx,
                "positive": pidx,
                "candidates": candidates,
                "labels": labels,
                "entity_ids": ent_ids,
                "relation_ids": rel_ids,
                "weights": weights,
                "history_pois": history_pois,
                "basics": basics,
            }
        )

    relation_name_to_idx = {name: idx for idx, name in enumerate(relation_map.idx_to_obj)}
    data = {
        "users": users,
        "pois": pois,
        "entities": entity_map.idx_to_obj,
        "relations": relation_map.idx_to_obj,
        "relation_name_to_idx": relation_name_to_idx,
        "poi_attrs": poi_attrs,
        "user_feature": user_feature,
        "poi_feature": poi_feature,
        "basic_dim": len(train_instances[0]["basic"]) if train_instances else 16,
        "train_instances": train_instances,
        "test_queries": test_queries,
        "train_user_pos": {int(k): sorted(v) for k, v in train_user_pos.items()},
        "popularity": popularity,
        "kg_triples": triples,
        "config": vars(args),
    }
    return data


def save_outputs(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(data, f)
    summary = {
        "users": len(data["users"]),
        "pois": len(data["pois"]),
        "entities": len(data["entities"]),
        "relations": len(data["relations"]),
        "train_instances": len(data["train_instances"]),
        "test_queries": len(data["test_queries"]),
        "kg_triples": len(data["kg_triples"]),
        "basic_dim": data["basic_dim"],
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    triples_path = output.with_suffix(".kg_triples.tsv")
    with triples_path.open("w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\n")
        for h, r, t in data["kg_triples"]:
            f.write(f"{h}\t{r}\t{t}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/trd_small.pkl"))
    parser.add_argument("--max-train-orders", type=int, default=50000)
    parser.add_argument("--max-test-orders", type=int, default=5000)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--max-interests", type=int, default=64)
    parser.add_argument("--max-food-attrs-per-poi", type=int, default=30)
    parser.add_argument("--decay-lambda", type=float, default=0.12)
    parser.add_argument("--click-weight", type=float, default=1.6)
    parser.add_argument("--click-decay", type=float, default=0.88)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument("--eval-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = build_processed(args)
    save_outputs(data, args.output)


if __name__ == "__main__":
    main()
