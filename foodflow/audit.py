from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import REQUIRED_TRD_FILES
from .io import ensure_dir


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as fh:
        for _ in fh:
            count += 1
    return count


def _data_rows(path: Path) -> int:
    lines = _line_count(path)
    return max(lines - 1, 0)


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(sum(1 for _ in path.open("rb")) - 1)


def audit_data(
    raw_dir: Path,
    processed_dir: Path,
    output: Path,
    markdown: Path | None = None,
) -> dict[str, object]:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    output = Path(output)

    raw_files = {}
    for name in REQUIRED_TRD_FILES:
        path = raw_dir / name
        raw_files[name] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "rows": _data_rows(path),
        }

    processed_files = {}
    for name in [
        "users.csv",
        "merchants.csv",
        "spus.csv",
        "orders_train.csv",
        "orders_test.csv",
        "test_interactions.csv",
    ]:
        path = processed_dir / name
        processed_files[name] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "rows": _csv_rows(path),
        }

    note_path = processed_dir / "data_note.json"
    data_note = json.loads(note_path.read_text(encoding="utf-8")) if note_path.exists() else {}
    raw_train = int(raw_files.get("orders_train.txt", {}).get("rows", 0))
    processed_train = int(processed_files.get("orders_train.csv", {}).get("rows", 0))
    sample_fraction = processed_train / raw_train if raw_train else 0.0
    uses_full_train = raw_train > 0 and processed_train >= raw_train

    audit = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "required_raw_files_present": all(item["exists"] for item in raw_files.values()),
        "raw_files": raw_files,
        "processed_files": processed_files,
        "data_note": data_note,
        "raw_train_orders": raw_train,
        "processed_train_orders": processed_train,
        "train_sample_fraction": sample_fraction,
        "train_mode": "full" if uses_full_train else "sampled",
        "rider_data": "default synthetic proxy; LaDe delivery CSV can calibrate rider parameters, but TRD has no full rider state or dispatch records",
    }

    ensure_dir(output.parent)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown:
        _write_markdown_audit(audit, Path(markdown))
    return audit


def _write_markdown_audit(audit: dict[str, object], markdown: Path) -> None:
    ensure_dir(markdown.parent)
    raw_files = pd.DataFrame.from_dict(audit["raw_files"], orient="index").reset_index(names="file")
    processed_files = pd.DataFrame.from_dict(audit["processed_files"], orient="index").reset_index(names="file")
    train_mode = audit["train_mode"]
    sample_fraction = float(audit["train_sample_fraction"])
    text = f"""# FoodFlow 数据审计

## 结论

- 必需 TRD 原始文件是否齐全：`{audit["required_raw_files_present"]}`
- 当前训练集处理模式：`{train_mode}`
- 原始训练订单数：`{audit["raw_train_orders"]}`
- 处理后训练订单数：`{audit["processed_train_orders"]}`
- 训练订单使用比例：`{sample_fraction:.4f}`
- 骑手数据边界：TRD 不包含完整骑手状态与派单记录；默认使用固定 seed 合成 proxy，也可用 LaDe delivery CSV 校准骑手速度、服务时长和负载分布。

如果 `train_mode` 为 `sampled`，说明当前结果使用真实 TRD 数据的固定 seed 抽样版本；如果为 `full`，说明当前处理后的训练订单覆盖完整 `orders_train.txt`。

## 原始 TRD 文件

{raw_files.to_markdown(index=False)}

## 处理后文件

{processed_files.to_markdown(index=False)}
"""
    markdown.write_text(text, encoding="utf-8")
