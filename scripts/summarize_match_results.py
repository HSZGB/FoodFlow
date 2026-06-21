from __future__ import annotations

import json
from pathlib import Path


SOURCES = [
    ("Popularity", "outputs/baselines_trd_gpu_100k.json", "Popularity"),
    ("UserCF", "outputs/baselines_trd_gpu_100k.json", "UserCF"),
    ("ItemCF", "outputs/baselines_trd_gpu_100k.json", "ItemCF"),
    ("MF", "outputs/mf_trd_gpu_100k.json", None),
    ("Static KG", "outputs/static_kg_trd_gpu_100k.json", None),
    ("KG + Time Decay", "outputs/kg_time_trd_gpu_100k.json", None),
    ("KG + Time + Temp", "outputs/kg_time_temp_trd_gpu_100k.json", None),
    ("Full", "outputs/full_trd_gpu_100k_match.json", None),
]


def load_metrics(path: str, key: str | None) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if key is None:
        return data["metrics"]
    return data[key]


def load_hybrid_best() -> dict[str, float | str]:
    rows = json.loads(Path("outputs/hybrid_itemcf_full_match_trd_gpu_100k_fine.json").read_text(encoding="utf-8"))
    coarse = json.loads(Path("outputs/hybrid_itemcf_full_match_trd_gpu_100k.json").read_text(encoding="utf-8"))
    rows.extend(coarse)
    # Use a balanced row that maximizes Recall@20 first, then AUC. This is the best wide-list result.
    best = max(rows, key=lambda row: (row["Recall@20"], row["AUC"]))
    best["model"] = "Hybrid ItemCF + Full"
    return best


def main() -> None:
    rows = []
    for name, path, key in SOURCES:
        rows.append({"model": name, **load_metrics(path, key)})
    rows.append(load_hybrid_best())
    out = Path("outputs/summary_trd_gpu_100k_match.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
