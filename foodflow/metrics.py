from __future__ import annotations

from collections import Counter
from math import log2
from typing import Iterable

import numpy as np
import pandas as pd


def recall_at_k(recs: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(recs[:k]) & truth) / min(len(truth), k)


def hitrate_at_k(recs: list[str], truth: set[str], k: int) -> float:
    return float(bool(set(recs[:k]) & truth))


def mrr_at_k(recs: list[str], truth: set[str], k: int) -> float:
    for idx, item in enumerate(recs[:k], start=1):
        if item in truth:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(recs: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    dcg = 0.0
    for idx, item in enumerate(recs[:k], start=1):
        if item in truth:
            dcg += 1.0 / log2(idx + 1)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def gini(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(np.clip(arr, 0, None))
    n = arr.size
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def jensen_shannon_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    p = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    q = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    midpoint = 0.5 * (p + q)

    def kl_divergence(values: np.ndarray, target: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / target[mask])))

    return float(0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint))


def _category_distribution(items: list[str], category_by_merchant: dict[str, str]) -> dict[str, float]:
    counts = Counter()
    for item in items:
        category = category_by_merchant.get(str(item))
        if category and category.lower() != "nan":
            counts[category] += 1
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {category: count / total for category, count in counts.items()}


def category_jsd_at_k(
    recs: list[str],
    history: list[str],
    category_by_merchant: dict[str, str],
    k: int,
) -> float:
    history_dist = _category_distribution(history, category_by_merchant)
    rec_dist = _category_distribution(recs[:k], category_by_merchant)
    return jensen_shannon_divergence(history_dist, rec_dist)


def _mean_segment_recall(
    recommendations: dict[str, list[str]],
    truth_by_user: dict[str, set[str]],
    history_by_user: dict[str, list[str]],
    users: list[str],
    k: int,
    repeat: bool,
) -> float:
    values: list[float] = []
    for user_id in users:
        history = set(history_by_user.get(user_id, []))
        truth = set(truth_by_user[user_id])
        segment_truth = truth & history if repeat else truth - history
        if not segment_truth:
            continue
        values.append(recall_at_k(recommendations[user_id], segment_truth, k))
    return float(np.mean(values)) if values else 0.0


def evaluate_recommendations(
    recommendations: dict[str, list[str]],
    truth_by_user: dict[str, set[str]],
    merchants: pd.DataFrame,
    ks: list[int],
    history_by_user: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    users = [u for u in truth_by_user if u in recommendations]
    if not users:
        raise ValueError("No overlapping users between recommendations and truth labels.")
    for k in ks:
        metrics[f"Recall@{k}"] = float(np.mean([recall_at_k(recommendations[u], truth_by_user[u], k) for u in users]))
        metrics[f"NDCG@{k}"] = float(np.mean([ndcg_at_k(recommendations[u], truth_by_user[u], k) for u in users]))
        metrics[f"MRR@{k}"] = float(np.mean([mrr_at_k(recommendations[u], truth_by_user[u], k) for u in users]))
        metrics[f"HitRate@{k}"] = float(np.mean([hitrate_at_k(recommendations[u], truth_by_user[u], k) for u in users]))
        if history_by_user is not None:
            metrics[f"RepeatRecall@{k}"] = _mean_segment_recall(
                recommendations,
                truth_by_user,
                history_by_user,
                users,
                k,
                repeat=True,
            )
            metrics[f"ExploreRecall@{k}"] = _mean_segment_recall(
                recommendations,
                truth_by_user,
                history_by_user,
                users,
                k,
                repeat=False,
            )
    top_k = max(ks)
    exposure = Counter()
    weighted = Counter()
    for recs in recommendations.values():
        for rank, merchant_id in enumerate(recs[:top_k], start=1):
            exposure[merchant_id] += 1
            weighted[merchant_id] += 1.0 / np.log2(rank + 1)
    order_count = pd.to_numeric(merchants["order_count"], errors="coerce").fillna(0)
    catalog = merchants.loc[order_count > 0].copy()
    if catalog.empty:
        catalog = merchants.copy()
        order_count = pd.to_numeric(catalog["order_count"], errors="coerce").fillna(0)
    all_merchants = catalog["wm_poi_id"].astype(str).tolist()
    exposed = {m for m, count in exposure.items() if count > 0}
    metrics[f"Coverage@{top_k}"] = len(exposed) / max(len(all_merchants), 1)
    exposure_values = [weighted[m] for m in all_merchants]
    metrics["ExposureGini"] = gini(exposure_values)
    threshold = order_count.loc[catalog.index].quantile(0.8)
    long_tail = set(catalog.loc[order_count.loc[catalog.index] <= threshold, "wm_poi_id"].astype(str))
    metrics[f"LongTailExposure@{top_k}"] = len(exposed & long_tail) / max(len(exposed), 1)
    if history_by_user is not None:
        category_col = "primary_first_tag_id"
        if category_col not in merchants.columns:
            category_col = "primary_first_tag_name" if "primary_first_tag_name" in merchants.columns else ""
        if category_col:
            category_by_merchant = dict(zip(merchants["wm_poi_id"].astype(str), merchants[category_col].astype(str)))
            calibration_values = [
                category_jsd_at_k(
                    recommendations[u],
                    history_by_user.get(u, []),
                    category_by_merchant,
                    top_k,
                )
                for u in users
            ]
            metrics[f"CategoryJSD@{top_k}"] = float(np.mean(calibration_values))
    return metrics
