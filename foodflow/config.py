from __future__ import annotations

from pathlib import Path

TRD_RECORD_API = "https://zenodo.org/api/records/8025855"

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")

REQUIRED_TRD_FILES = [
    "users.txt",
    "pois.txt",
    "spus.txt",
    "orders_train.txt",
    "orders_test_poi.txt",
    "orders_poi_test_label.txt",
]

OPTIONAL_TRD_FILES = [
    "orders_spu_train.txt",
    "orders_test_spu.txt",
    "orders_spu_test_label.txt",
    "orders_poi_session.txt",
    "README.md",
]
