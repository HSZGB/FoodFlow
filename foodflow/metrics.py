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


def evaluate_recommendations(
    recommendations: dict[str, list[str]],
    truth_by_user: dict[str, set[str]],
    merchants: pd.DataFrame,
    ks: list[int],
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
    return metrics
