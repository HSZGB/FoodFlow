#!/usr/bin/env python3
"""Download TRD text files from Zenodo.

The original record also contains graph.bin, which is large and not required by
this project because we rebuild the KG from text files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


RECORD_API = "https://zenodo.org/api/records/8025855"
SKIP_BY_DEFAULT = {"graph.bin"}


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urlopen(url) as response, tmp_path.open("wb") as f:
        total = response.headers.get("Content-Length")
        total_i = int(total) if total else None
        seen = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            seen += len(chunk)
            if total_i:
                pct = 100.0 * seen / total_i
                print(f"\r{out_path.name}: {seen / 1e6:.1f}MB/{total_i / 1e6:.1f}MB {pct:.1f}%", end="")
        print()
    tmp_path.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--include-graph", action="store_true")
    parser.add_argument("--files", nargs="*", default=None, help="Optional subset of file names.")
    args = parser.parse_args()

    with urlopen(RECORD_API) as response:
        record = json.load(response)

    wanted = set(args.files) if args.files else None
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    for item in record["files"]:
        name = item["key"]
        if wanted is not None and name not in wanted:
            continue
        if not args.include_graph and name in SKIP_BY_DEFAULT:
            continue
        out_path = args.raw_dir / name
        checksum = item.get("checksum", "")
        expected_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        if out_path.exists():
            if expected_md5 and md5sum(out_path) == expected_md5:
                print(f"{name}: exists and checksum OK")
                continue
            print(f"{name}: exists but checksum missing/mismatch, re-downloading")
        url = item["links"]["self"]
        print(f"Downloading {name} -> {out_path}")
        download_file(url, out_path)
        if expected_md5:
            got = md5sum(out_path)
            if got != expected_md5:
                raise RuntimeError(f"Checksum mismatch for {name}: {got} != {expected_md5}")
            print(f"{name}: checksum OK")


if __name__ == "__main__":
    main()
