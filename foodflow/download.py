from __future__ import annotations

from pathlib import Path

import requests

from .config import OPTIONAL_TRD_FILES, REQUIRED_TRD_FILES, TRD_RECORD_API
from .io import download_file, ensure_dir


def fetch_trd_file_manifest() -> list[dict]:
    resp = requests.get(TRD_RECORD_API, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    return payload["files"]


def download_trd(raw_dir: Path, skip_graph: bool = True, required_only: bool = False) -> list[Path]:
    raw_dir = ensure_dir(raw_dir)
    marker = raw_dir / "MOCK_DATASET"
    if marker.exists():
        marker.unlink()
    wanted = set(REQUIRED_TRD_FILES)
    if not required_only:
        wanted.update(OPTIONAL_TRD_FILES)
    manifest = fetch_trd_file_manifest()
    downloaded: list[Path] = []
    for item in manifest:
        key = item["key"]
        if skip_graph and key == "graph.bin":
            continue
        if key not in wanted:
            continue
        out_path = raw_dir / key
        if out_path.exists() and out_path.stat().st_size == item["size"]:
            downloaded.append(out_path)
            continue
        download_file(item["links"]["self"], out_path, expected_size=item["size"])
        downloaded.append(out_path)
    missing = [name for name in REQUIRED_TRD_FILES if not (raw_dir / name).exists()]
    if missing:
        raise RuntimeError(f"TRD download incomplete, missing: {missing}")
    return downloaded
