from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io_utils import load_pickle
from .metrics import ranking_metrics


def load_data(path: Path) -> dict:
    return load_pickle(path)


def popularity_scores(data: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pop = np.asarray(data["popularity"], dtype=np.float32)
    scores_list = []
    labels_list = []
    for q in data["test_queries"]:
        scores_list.append(pop[np.asarray(q["candidates"], dtype=np.int64)])
        labels_list.append(np.asarray(q["labels"], dtype=np.float32))
    return scores_list, labels_list


def build_sets(data: dict) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    user_items = {int(u): set(items) for u, items in data["train_user_pos"].items()}
    item_users: dict[int, set[int]] = defaultdict(set)
    for u, items in user_items.items():
        for item in items:
            item_users[item].add(u)
    return user_items, item_users


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def cosine_set(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / math.sqrt(len(a) * len(b))


def itemcf_scores(data: dict, max_history: int = 30) -> tuple[list[np.ndarray], list[np.ndarray]]:
    user_items, item_users = build_sets(data)
    sim_cache: dict[tuple[int, int], float] = {}

    def sim(i: int, j: int) -> float:
        if i == j:
            return 1.0
        key = (min(i, j), max(i, j))
        if key not in sim_cache:
            sim_cache[key] = jaccard(item_users.get(i, set()), item_users.get(j, set()))
        return sim_cache[key]

    scores_list = []
    labels_list = []
    for q in data["test_queries"]:
        hist = q.get("history_pois") or list(user_items.get(q["user"], set()))[-max_history:]
        scores = []
        for cand in q["candidates"]:
            scores.append(sum(sim(cand, h) for h in hist[-max_history:]))
        scores_list.append(np.asarray(scores, dtype=np.float32))
        labels_list.append(np.asarray(q["labels"], dtype=np.float32))
    return scores_list, labels_list


def usercf_scores(data: dict, max_neighbors: int = 80) -> tuple[list[np.ndarray], list[np.ndarray]]:
    user_items, item_users = build_sets(data)
    scores_list = []
    labels_list = []
    for q in data["test_queries"]:
        target_items = set(q.get("history_pois") or user_items.get(q["user"], set()))
        overlap_counts: dict[int, int] = defaultdict(int)
        for item in target_items:
            for other in item_users.get(item, set()):
                if other != q["user"]:
                    overlap_counts[other] += 1
        target_norm = math.sqrt(max(1, len(target_items)))
        sims = [
            (inter / (target_norm * math.sqrt(len(user_items.get(other, set())))), other)
            for other, inter in overlap_counts.items()
            if user_items.get(other)
        ]
        sims.sort(reverse=True)
        sims = sims[:max_neighbors]
        scores = np.zeros(len(q["candidates"]), dtype=np.float32)
        candidate_pos = {cand: pos for pos, cand in enumerate(q["candidates"])}
        for s, other in sims:
            for item in user_items.get(other, set()):
                pos = candidate_pos.get(item)
                if pos is not None:
                    scores[pos] += s
        scores_list.append(scores)
        labels_list.append(np.asarray(q["labels"], dtype=np.float32))
    return scores_list, labels_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data = load_data(args.data)

    results = {}
    for name, fn in [("Popularity", popularity_scores), ("ItemCF", itemcf_scores), ("UserCF", usercf_scores)]:
        scores, labels = fn(data)
        results[name] = ranking_metrics(scores, labels, ks=(5, 10, 20))
        print(name, json.dumps(results[name], ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
