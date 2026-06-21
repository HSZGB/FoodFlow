from __future__ import annotations

import json
from pathlib import Path


MODELS = [
    ("Popularity", "outputs/baselines_trd_gpu_100k.json", "Popularity"),
    ("UserCF", "outputs/baselines_trd_gpu_100k.json", "UserCF"),
    ("ItemCF", "outputs/baselines_trd_gpu_100k.json", "ItemCF"),
    ("MF", "outputs/mf_trd_gpu_100k.json", None),
    ("Static KG", "outputs/static_kg_trd_gpu_100k.json", None),
    ("KG + Time Decay", "outputs/kg_time_trd_gpu_100k.json", None),
    ("KG + Time + Temp", "outputs/kg_time_temp_trd_gpu_100k.json", None),
    ("Full (10 epoch)", "outputs/full_trd_gpu_100k.json", None),
    ("Full (early stop, 5 epoch)", "outputs/full_trd_gpu_100k_e5.json", None),
]


def load_metrics(path: str, key: str | None) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if key is not None:
        return data[key]
    return data["metrics"]


def main() -> None:
    rows = []
    for name, path, key in MODELS:
        metrics = load_metrics(path, key)
        rows.append({"model": name, **metrics})
    out = Path("outputs/summary_trd_gpu_100k.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
