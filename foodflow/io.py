from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm


def canonical_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: Path | str) -> pd.DataFrame:
    """Read TRD-style txt/csv files with a small delimiter sniff."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(4096)
    sep = "\t" if sample.count("\t") >= sample.count(",") else ","
    return pd.read_csv(path, sep=sep, engine="python")


def write_csv(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def md5sum(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path | str, expected_size: int | None = None) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    resume_from = out_path.stat().st_size if out_path.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else None
    if expected_size is not None and resume_from >= expected_size:
        return
    with requests.get(url, stream=True, timeout=(30, 300), headers=headers) as resp:
        resp.raise_for_status()
        if resume_from and resp.status_code != 206:
            resume_from = 0
            out_path.unlink(missing_ok=True)
        total = int(resp.headers.get("content-length", 0))
        if expected_size is not None and resume_from:
            total = expected_size
        with out_path.open("ab" if resume_from else "wb") as fh, tqdm(
            total=total,
            initial=resume_from if total else 0,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
            disable=total == 0,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    if expected_size is not None and out_path.stat().st_size < expected_size:
        raise RuntimeError(f"{out_path} is smaller than expected: {out_path.stat().st_size} < {expected_size}")


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col.endswith("_id") or col in {"user_id", "wm_poi_id", "wm_food_spu_id", "wm_order_id"}:
            df[col] = df[col].map(canonical_id)
    return df
