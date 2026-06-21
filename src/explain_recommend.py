from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .datasets import collate_instances
from .io_utils import load_pickle
from .model import DynamicKGAttentionRecommender
from .run_experiment import move_batch


def build_full_model(data: dict[str, Any], checkpoint: dict[str, Any] | None, embed_dim: int, hidden_dim: int) -> DynamicKGAttentionRecommender:
    args = checkpoint.get("args", {}) if checkpoint else {}
    model = DynamicKGAttentionRecommender(
        n_users=len(data["users"]),
        n_pois=len(data["pois"]),
        n_entities=len(data["entities"]),
        n_relations=len(data["relations"]),
        basic_dim=data["basic_dim"],
        embed_dim=int(args.get("embed_dim", embed_dim)),
        hidden_dim=int(args.get("hidden_dim", hidden_dim)),
        dropout=float(args.get("dropout", 0.0)),
        use_time_bias=True,
        use_relation_attention=True,
        use_basic_features=True,
    )
    if checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def instances_for_query(query: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "user": query["user"],
            "poi": poi,
            "label": label,
            "entity_ids": query["entity_ids"],
            "relation_ids": query["relation_ids"],
            "weights": query["weights"],
            "basic": basic,
        }
        for poi, label, basic in zip(query["candidates"], query["labels"], query["basics"])
    ]


def explain_paths(
    query: dict[str, Any],
    poi_idx: int,
    data: dict[str, Any],
    alpha: np.ndarray,
    max_paths: int,
) -> list[str]:
    poi_attr_entities = {ent for ent, _rel in data["poi_attrs"][poi_idx]}
    candidates = []
    for pos, ent in enumerate(query["entity_ids"]):
        if ent == 0 or ent not in poi_attr_entities:
            continue
        relation = data["relations"][query["relation_ids"][pos]]
        entity = data["entities"][ent]
        weight = query["weights"][pos]
        attn = float(alpha[pos]) if pos < len(alpha) else 0.0
        source = "近期点击临时兴趣" if relation == "pref_click" else "历史订单时间衰减兴趣"
        candidates.append((attn, weight, f"{source} -> {entity} -> 候选商家"))
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return [x[2] for x in candidates[:max_paths]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-paths", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data = load_pickle(args.data)
    checkpoint = None
    if args.checkpoint and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_full_model(data, checkpoint, args.embed_dim, args.hidden_dim).to(device)
    model.eval()

    query = data["test_queries"][args.query_index]
    batch = collate_instances(instances_for_query(query), data)
    with torch.no_grad():
        logits, extra = model(move_batch(batch, device))
        scores = torch.sigmoid(logits).cpu().numpy()
        alpha = extra["interest_alpha"][0].cpu().numpy()
    order = np.argsort(-scores)[: args.topk]
    print(f"user={data['users'][query['user']]} positive_poi={data['pois'][query['positive']]}")
    for rank, idx in enumerate(order, start=1):
        poi_idx = query["candidates"][idx]
        label = int(query["labels"][idx])
        print(f"\n#{rank} poi={data['pois'][poi_idx]} score={scores[idx]:.4f} label={label}")
        paths = explain_paths(query, poi_idx, data, alpha, args.max_paths)
        if paths:
            for path in paths:
                print(f"  - {path}")
        else:
            print("  - 与当前动态图谱兴趣无直接重合，主要由商家静态属性、用户向量和基础特征排序。")


if __name__ == "__main__":
    main()
