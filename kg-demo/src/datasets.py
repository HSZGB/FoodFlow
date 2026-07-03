from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TakeoutTrainDataset(Dataset):
    def __init__(self, data: dict[str, Any], interest_policy: str = "hist_click") -> None:
        self.data = data
        self.instances = data["train_instances"]
        self.interest_policy = interest_policy
        rel = data["relation_name_to_idx"]
        self.hist_rel = rel.get("pref_hist", -1)
        self.click_rel = rel.get("pref_click", -1)

    def __len__(self) -> int:
        return len(self.instances)

    def _filter_interest(self, item: dict[str, Any]) -> tuple[list[int], list[int], list[float]]:
        ents = item["entity_ids"]
        rels = item["relation_ids"]
        vals = item["weights"]
        if self.interest_policy == "none":
            return [0], [0], [0.0]
        if self.interest_policy == "hist":
            keep = [i for i, r in enumerate(rels) if r == self.hist_rel]
        elif self.interest_policy == "click":
            keep = [i for i, r in enumerate(rels) if r == self.click_rel]
        else:
            keep = list(range(len(ents)))
        if not keep:
            return [0], [0], [0.0]
        return [ents[i] for i in keep], [rels[i] for i in keep], [vals[i] for i in keep]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.instances[idx]
        ents, rels, vals = self._filter_interest(item)
        return {
            "user": item["user"],
            "poi": item["poi"],
            "label": item["label"],
            "entity_ids": ents,
            "relation_ids": rels,
            "weights": vals,
            "basic": item["basic"],
        }


def _pad_2d(rows: list[list[int] | list[float]], pad_value: int | float = 0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(1, max(len(x) for x in rows))
    is_float = any(isinstance(v, float) for row in rows for v in row)
    dtype = torch.float32 if is_float else torch.long
    out = torch.full((len(rows), max_len), pad_value, dtype=dtype)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for i, row in enumerate(rows):
        if not row:
            continue
        values = torch.tensor(row, dtype=dtype)
        out[i, : len(row)] = values
        mask[i, : len(row)] = values.ne(0) if dtype == torch.long else values.gt(0)
    return out, mask


def collate_instances(batch: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, torch.Tensor]:
    users = torch.tensor([x["user"] for x in batch], dtype=torch.long)
    pois = torch.tensor([x["poi"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    basic = torch.tensor(np.asarray([x["basic"] for x in batch], dtype=np.float32), dtype=torch.float32)

    interest_e, interest_mask = _pad_2d([x["entity_ids"] for x in batch], 0)
    interest_r, _ = _pad_2d([x["relation_ids"] for x in batch], 0)
    interest_w, weight_mask = _pad_2d([x["weights"] for x in batch], 0.0)
    interest_mask = interest_mask & weight_mask

    poi_attrs = data["poi_attrs"]
    attr_entities = []
    attr_relations = []
    for poi in pois.tolist():
        pairs = poi_attrs[poi]
        if pairs:
            attr_entities.append([p[0] for p in pairs])
            attr_relations.append([p[1] for p in pairs])
        else:
            attr_entities.append([0])
            attr_relations.append([0])
    attr_e, attr_mask = _pad_2d(attr_entities, 0)
    attr_r, _ = _pad_2d(attr_relations, 0)

    return {
        "user": users,
        "poi": pois,
        "label": labels,
        "interest_e": interest_e,
        "interest_r": interest_r,
        "interest_w": interest_w,
        "interest_mask": interest_mask,
        "attr_e": attr_e,
        "attr_r": attr_r,
        "attr_mask": attr_mask,
        "basic": basic,
    }


def make_query_batch(query: dict[str, Any], data: dict[str, Any], interest_policy: str) -> dict[str, torch.Tensor]:
    rel = data["relation_name_to_idx"]
    hist_rel = rel.get("pref_hist", -1)
    click_rel = rel.get("pref_click", -1)

    ents = query["entity_ids"]
    rels = query["relation_ids"]
    vals = query["weights"]
    if interest_policy == "none":
        keep = []
    elif interest_policy == "hist":
        keep = [i for i, r in enumerate(rels) if r == hist_rel]
    elif interest_policy == "click":
        keep = [i for i, r in enumerate(rels) if r == click_rel]
    else:
        keep = list(range(len(ents)))
    if keep:
        ents = [ents[i] for i in keep]
        rels = [rels[i] for i in keep]
        vals = [vals[i] for i in keep]
    else:
        ents, rels, vals = [0], [0], [0.0]

    instances = []
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
    return collate_instances(instances, data)
