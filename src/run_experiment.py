from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .datasets import TakeoutTrainDataset, collate_instances
from .io_utils import json_ready, load_pickle
from .metrics import format_metrics, ranking_metrics
from .model import DynamicKGAttentionRecommender, MatrixFactorization


MODEL_POLICIES = {
    "mf": "none",
    "static_kg": "none",
    "kg_time": "hist",
    "kg_time_temp": "hist_click",
    "full": "hist_click",
}


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def build_model(args: argparse.Namespace, data: dict[str, Any]) -> nn.Module:
    if args.model == "mf":
        return MatrixFactorization(len(data["users"]), len(data["pois"]), embed_dim=args.embed_dim)
    return DynamicKGAttentionRecommender(
        n_users=len(data["users"]),
        n_pois=len(data["pois"]),
        n_entities=len(data["entities"]),
        n_relations=len(data["relations"]),
        basic_dim=data["basic_dim"],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_time_bias=args.model in {"kg_time", "kg_time_temp", "full"},
        use_relation_attention=args.model == "full",
        use_basic_features=args.model in {"kg_time_temp", "full", "static_kg"},
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: dict[str, Any],
    interest_policy: str,
    device: torch.device,
    max_queries: int | None = None,
    batch_queries: int = 64,
) -> dict[str, float]:
    model.eval()
    scores_list = []
    labels_list = []
    queries = data["test_queries"][:max_queries] if max_queries else data["test_queries"]
    rel = data["relation_name_to_idx"]
    hist_rel = rel.get("pref_hist", -1)
    click_rel = rel.get("pref_click", -1)

    def filtered_interest(query: dict[str, Any]) -> tuple[list[int], list[int], list[float]]:
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

    for start in range(0, len(queries), batch_queries):
        chunk = queries[start : start + batch_queries]
        instances = []
        lengths = []
        chunk_labels = []
        for query in chunk:
            ents, rels, vals = filtered_interest(query)
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
        batch = collate_instances(instances, data)
        batch = move_batch(batch, device)
        scores, _ = model(batch)
        arr = scores.detach().cpu().numpy()
        offset = 0
        for length, labels in zip(lengths, chunk_labels):
            scores_list.append(arr[offset : offset + length])
            labels_list.append(labels)
            offset += length
    return ranking_metrics(scores_list, labels_list, ks=(5, 10, 20))


def train(args: argparse.Namespace) -> dict[str, float]:
    data = load_pickle(Path(args.data))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    interest_policy = MODEL_POLICIES[args.model]
    dataset = TakeoutTrainDataset(data, interest_policy=interest_policy)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_instances, data=data),
    )

    model = build_model(args, data).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    metrics: dict[str, float] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in loader:
            batch = move_batch(batch, device)
            logits, _ = model(batch)
            loss = criterion(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * batch["label"].numel()
            seen += batch["label"].numel()
        loss_value = total_loss / max(1, seen)
        should_eval = args.eval_every > 0 and (epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs)
        if should_eval:
            metrics = evaluate(
                model,
                data,
                interest_policy,
                device,
                max_queries=args.eval_queries,
                batch_queries=args.eval_batch_queries,
            )
            print(f"epoch={epoch} loss={loss_value:.5f} {format_metrics(metrics)}")
        else:
            print(f"epoch={epoch} loss={loss_value:.5f}")

    if metrics is None:
        metrics = evaluate(
            model,
            data,
            interest_policy,
            device,
            max_queries=args.eval_queries,
            batch_queries=args.eval_batch_queries,
        )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": args.model, "metrics": metrics}, indent=2), encoding="utf-8")
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": args.model,
                "state_dict": model.state_dict(),
                "args": json_ready(vars(args)),
                "metrics": metrics,
            },
            ckpt,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_POLICIES), default="full")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-queries", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-batch-queries", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
