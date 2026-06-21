from __future__ import annotations

import numpy as np


def ranking_metrics(scores_list: list[np.ndarray], labels_list: list[np.ndarray], ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = 0.0
        metrics[f"Precision@{k}"] = 0.0
        metrics[f"HitRate@{k}"] = 0.0
        metrics[f"NDCG@{k}"] = 0.0
    mrr = 0.0
    auc = 0.0
    n_auc = 0

    for scores, labels in zip(scores_list, labels_list):
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        positives = int(labels.sum())
        positives = max(1, positives)
        pos_ranks = np.where(sorted_labels > 0)[0]
        first_rank = int(pos_ranks[0]) + 1 if len(pos_ranks) else len(labels) + 1
        mrr += 1.0 / first_rank
        for k in ks:
            top = sorted_labels[:k]
            hits = float(top.sum())
            metrics[f"Recall@{k}"] += hits / positives
            metrics[f"Precision@{k}"] += hits / k
            metrics[f"HitRate@{k}"] += 1.0 if hits > 0 else 0.0
            dcg = 0.0
            for i, rel in enumerate(top, start=1):
                if rel > 0:
                    dcg += 1.0 / np.log2(i + 1)
            ideal_hits = min(positives, k)
            idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
            metrics[f"NDCG@{k}"] += dcg / idcg if idcg > 0 else 0.0
        pos_scores = scores[labels > 0]
        neg_scores = scores[labels <= 0]
        if len(pos_scores) and len(neg_scores):
            auc += float((pos_scores[:, None] > neg_scores[None, :]).mean() + 0.5 * (pos_scores[:, None] == neg_scores[None, :]).mean())
            n_auc += 1

    n = max(1, len(scores_list))
    for key in list(metrics):
        metrics[key] /= n
    metrics["MRR"] = mrr / n
    metrics["AUC"] = auc / max(1, n_auc)
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    keys = ["Recall@5", "Recall@10", "NDCG@10", "MRR", "AUC"]
    return " | ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
