from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foodflow.data import PreparedData
from foodflow.metrics import evaluate_recommendations
from foodflow.recommenders import SEQ_TUNED_WEIGHTS, SEQ_TUNED_XQUAD_WEIGHTS, SequentialHybridRecommender


FEATURE_NAMES = [
    "fast_recency",
    "slow_recency",
    "repeat",
    "transition",
    "category",
    "popularity",
    "quality",
]

BASE_WEIGHTS = {
    "fast_recency": 0.25,
    "slow_recency": 0.12,
    "repeat": 0.30,
    "transition": 0.23,
    "category": 0.05,
    "popularity": 0.03,
    "quality": 0.02,
}


def normalize_weights(weights: dict[str, float] | np.ndarray) -> np.ndarray:
    if isinstance(weights, dict):
        values = np.asarray([weights[name] for name in FEATURE_NAMES], dtype=float)
    else:
        values = np.asarray(weights, dtype=float)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Weight sum must be positive.")
    return values / total


def build_feature_cache(
    data: PreparedData,
    user_limit: int,
    candidate_limit: int,
) -> tuple[SequentialHybridRecommender, list[str], dict[str, tuple[list[str], np.ndarray]], dict[str, set[str]]]:
    truth_all = data.truth_by_user()
    users = list(truth_all.keys())[:user_limit]
    truth = {user_id: truth_all[user_id] for user_id in users}
    model = SequentialHybridRecommender().fit(data)
    cache: dict[str, tuple[list[str], np.ndarray]] = {}

    for user_id in users:
        seq = model.recent_by_user.get(user_id, [])
        counts = model.user_item_counts.get(user_id, Counter())
        merchant_ids = model._sequential_candidates(user_id)[:candidate_limit]
        features = []
        for merchant_id in merchant_ids:
            merchant = model.merchants.loc[merchant_id]
            category = merchant.get("primary_first_tag_id", "unknown")
            recency_fast, recency_slow = model._recency_scores(seq, merchant_id)
            features.append(
                [
                    recency_fast,
                    recency_slow,
                    np.log1p(counts.get(merchant_id, 0)) / np.log(5),
                    model._transition_score(seq, merchant_id),
                    float(model.user_cat_counts.get(user_id, {}).get(category, 0.0)),
                    float(model.pop_log.get(merchant_id, 0.0)),
                    float(model.quality.get(merchant_id, 0.5)),
                ]
            )
        cache[user_id] = (merchant_ids, np.asarray(features, dtype=float))
    return model, users, cache, truth


def recommendations_from_weights(
    model: SequentialHybridRecommender,
    users: list[str],
    cache: dict[str, tuple[list[str], np.ndarray]],
    weights: np.ndarray,
    k: int,
) -> dict[str, list[str]]:
    recommendations = {}
    for user_id in users:
        merchant_ids, features = cache[user_id]
        scores = features @ weights
        order = np.argsort(-scores)
        ranked = [merchant_ids[int(index)] for index in order]
        recommendations[user_id] = model._remove_seen_and_backfill(user_id, ranked, k)
    return recommendations


def candidate_weights(trials: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    base = normalize_weights(BASE_WEIGHTS)
    candidates = [
        ("Seq-Hybrid-base", base),
        ("Seq-Tuned-current", normalize_weights(SEQ_TUNED_WEIGHTS)),
        ("Seq-Tuned-xQuAD-current", normalize_weights(SEQ_TUNED_XQUAD_WEIGHTS)),
        ("repeat-heavy", normalize_weights([0.18, 0.08, 0.45, 0.18, 0.05, 0.04, 0.02])),
        ("transition-heavy", normalize_weights([0.18, 0.08, 0.28, 0.32, 0.05, 0.06, 0.03])),
    ]
    noise = np.asarray([0.06, 0.03, 0.08, 0.06, 0.03, 0.02, 0.01], dtype=float)
    for index in range(trials):
        weights = np.clip(base + rng.normal(0.0, noise), 0.005, None)
        candidates.append((f"local-search-{index:03d}", normalize_weights(weights)))
    return candidates


def run_search(
    processed_dir: Path,
    output: Path,
    user_limit: int,
    candidate_limit: int,
    trials: int,
    seed: int,
    top_k: list[int],
) -> pd.DataFrame:
    data = PreparedData.load(processed_dir)
    model, users, cache, truth = build_feature_cache(data, user_limit, candidate_limit)
    history = data.history_by_user()
    rows = []
    for label, weights in candidate_weights(trials, seed):
        recs = recommendations_from_weights(model, users, cache, weights, max(top_k))
        metrics = evaluate_recommendations(recs, truth, data.merchants, top_k, history)
        row = {"candidate": label}
        row.update({name: float(value) for name, value in zip(FEATURE_NAMES, weights)})
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["Recall@20", "NDCG@20"], ascending=[False, False])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search lightweight sequence feature weights for FoodFlow.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/seq_weight_search.csv"))
    parser.add_argument("--user-limit", type=int, default=80)
    parser.add_argument("--candidate-limit", type=int, default=140)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 20])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = run_search(
        processed_dir=args.processed_dir,
        output=args.output,
        user_limit=args.user_limit,
        candidate_limit=args.candidate_limit,
        trials=args.trials,
        seed=args.seed,
        top_k=args.top_k,
    )
    cols = ["candidate", *FEATURE_NAMES, "Recall@20", "NDCG@20", "HitRate@20", "CategoryJSD@20"]
    print(df[cols].head(10).to_string(index=False))
    print(f"\nWrote {len(df)} candidates to {args.output}")


if __name__ == "__main__":
    main()
