from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .datasets import collate_instances
from .io_utils import load_pickle
from .metrics import ranking_metrics
from .run_baselines import itemcf_scores, popularity_scores, usercf_scores
from .run_experiment import MODEL_POLICIES, build_model, move_batch


def _filtered_interest(data: dict[str, Any], query: dict[str, Any], interest_policy: str) -> tuple[list[int], list[int], list[float]]:
    rel = data["relation_name_to_idx"]
    hist_rel = rel.get("pref_hist", -1)
    click_rel = rel.get("pref_click", -1)
    ents = query["entity_ids"]
    rels = query["relation_ids"]
    vals = query["weights"]
    if interest_policy == "none":
        keep: list[int] = []
    elif interest_policy == "hist":
        keep = [i for i, r in enumerate(rels) if r == hist_rel]
    elif interest_policy == "click":
        keep = [i for i, r in enumerate(rels) if r == click_rel]
    else:
        keep = list(range(len(ents)))
    if not keep:
        return [0], [0], [0.0]
    return [ents[i] for i in keep], [rels[i] for i in keep], [vals[i] for i in keep]


@torch.no_grad()
def neural_scores(
    data: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    batch_queries: int,
    max_queries: int | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))
    ckpt_args.setdefault("model", ckpt.get("model", "full"))
    ckpt_args.setdefault("embed_dim", 64)
    ckpt_args.setdefault("hidden_dim", 128)
    ckpt_args.setdefault("dropout", 0.0)
    model_name = ckpt_args["model"]
    interest_policy = MODEL_POLICIES[model_name]
    model = build_model(SimpleNamespace(**ckpt_args), data)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    queries = data["test_queries"][:max_queries] if max_queries else data["test_queries"]
    scores_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for start in range(0, len(queries), batch_queries):
        chunk = queries[start : start + batch_queries]
        instances = []
        lengths = []
        chunk_labels = []
        for query in chunk:
            ents, rels, vals = _filtered_interest(data, query, interest_policy)
            lengths.append(len(query["candidates"]))
            chunk_labels.append(np.asarray(query["labels"], dtype=np.float32))
            for poi, label, basic in zip(query["candidates"], query["labels"], query["basics"]):
                instances.append(
                    {
                        "user": query["user"],
                        "poi": poi,
                        "label": label,
                        "entity_ids": ents,
                        "relation_ids": rels,
                        "weights": vals,
                        "basic": basic,
                    }
                )
        batch = move_batch(collate_instances(instances, data), device)
        logits, _ = model(batch)
        flat_scores = logits.detach().cpu().numpy()
        offset = 0
        for length, labels in zip(lengths, chunk_labels):
            scores_list.append(flat_scores[offset : offset + length])
            labels_list.append(labels)
            offset += length
    return scores_list, labels_list


def normalize_per_query(scores: np.ndarray) -> np.ndarray:
    std = float(scores.std())
    if std < 1e-8:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - float(scores.mean())) / std).astype(np.float32)


def blend_scores(
    left: list[np.ndarray],
    right: list[np.ndarray],
    alpha: float,
) -> list[np.ndarray]:
    out = []
    for a, b in zip(left, right):
        out.append(alpha * normalize_per_query(a) + (1.0 - alpha) * normalize_per_query(b))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cf", choices=["itemcf", "usercf", "popularity"], default="itemcf")
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-queries", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = load_pickle(args.data)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    n_scores, labels = neural_scores(data, args.checkpoint, device, args.batch_queries, args.max_queries)
    if args.cf == "itemcf":
        cf_scores, cf_labels = itemcf_scores(data)
    elif args.cf == "usercf":
        cf_scores, cf_labels = usercf_scores(data)
    else:
        cf_scores, cf_labels = popularity_scores(data)
    if args.max_queries:
        cf_scores = cf_scores[: args.max_queries]
        cf_labels = cf_labels[: args.max_queries]
    if any(not np.array_equal(a, b) for a, b in zip(labels, cf_labels)):
        raise RuntimeError("Neural and CF labels are not aligned.")

    rows = []
    for alpha_text in args.alphas.split(","):
        alpha = float(alpha_text.strip())
        scores = blend_scores(n_scores, cf_scores, alpha)
        metrics = ranking_metrics(scores, labels, ks=(5, 10, 20))
        row = {"model": f"Hybrid neural:{alpha:.2f} {args.cf}:{1-alpha:.2f}", **metrics}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
