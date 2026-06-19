from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import ensure_dir, normalize_id_columns, read_table, require_columns, write_csv


def _safe_read(raw_dir: Path, name: str) -> pd.DataFrame:
    return normalize_id_columns(read_table(raw_dir / name))


def _add_geo(df: pd.DataFrame, id_col: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = sorted(df[id_col].astype(str).unique())
    coords = {
        key: (float(rng.normal(116.40, 0.045)), float(rng.normal(39.92, 0.035)))
        for key in keys
    }
    df = df.copy()
    df["lng"] = df[id_col].astype(str).map(lambda x: coords[x][0])
    df["lat"] = df[id_col].astype(str).map(lambda x: coords[x][1])
    return df


def _price_interval_to_midpoint(value: object) -> float:
    text = str(value)
    if text == "nan" or text == "NULL":
        return np.nan
    if text.startswith("<"):
        return float(text[1:]) * 0.85
    if text.startswith(">="):
        return float(text[2:]) * 1.15
    if text.startswith("[") and "," in text:
        lo, hi = text.strip("[]()").split(",", 1)
        return (float(lo) + float(hi)) / 2
    return pd.to_numeric(text, errors="coerce")


def _build_session_interactions(raw_dir: Path, test: pd.DataFrame) -> pd.DataFrame:
    path = raw_dir / "orders_poi_session.txt"
    columns = ["wm_order_id", "user_id", "wm_poi_id", "rank"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    sessions = _safe_read(raw_dir, "orders_poi_session.txt")
    if "user_id" not in sessions.columns and "wm_order_id" in sessions.columns and "user_id" in test.columns:
        sessions = sessions.merge(test[["wm_order_id", "user_id"]].drop_duplicates("wm_order_id"), on="wm_order_id", how="left")

    rows: list[dict[str, object]] = []
    if "clicks" in sessions.columns:
        for _, row in sessions.iterrows():
            clicked = [item.strip() for item in str(row.get("clicks", "")).split(",") if item.strip()]
            for rank, merchant_id in enumerate(clicked, start=1):
                rows.append(
                    {
                        "wm_order_id": row.get("wm_order_id", ""),
                        "user_id": row.get("user_id", ""),
                        "wm_poi_id": merchant_id,
                        "rank": rank,
                    }
                )
    elif "wm_poi_id" in sessions.columns:
        keep = [col for col in ["wm_order_id", "user_id", "wm_poi_id"] if col in sessions.columns]
        long_sessions = sessions[keep].dropna(subset=["user_id", "wm_poi_id"]).copy()
        long_sessions["rank"] = long_sessions.groupby(["wm_order_id", "user_id"]).cumcount() + 1
        rows = long_sessions[columns].to_dict("records")

    out = pd.DataFrame(rows, columns=columns)
    return out.dropna(subset=["user_id", "wm_poi_id"]).copy()


def _build_order_spus(raw_dir: Path, name: str, orders: pd.DataFrame) -> pd.DataFrame:
    path = raw_dir / name
    columns = ["wm_order_id", "user_id", "wm_poi_id", "wm_food_spu_id"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    order_spus = _safe_read(raw_dir, name)
    spu_col = "wm_food_spu_id" if "wm_food_spu_id" in order_spus.columns else "label_spu_id"
    if spu_col not in order_spus.columns:
        return pd.DataFrame(columns=columns)
    keep = [col for col in ["wm_order_id", spu_col] if col in order_spus.columns]
    out = order_spus[keep].rename(columns={spu_col: "wm_food_spu_id"}).dropna(subset=["wm_food_spu_id"])
    if "wm_order_id" in out.columns and "wm_order_id" in orders.columns:
        out = out.merge(
            orders[["wm_order_id", "user_id", "wm_poi_id"]].drop_duplicates("wm_order_id"),
            on="wm_order_id",
            how="left",
        )
    return out[columns].dropna(subset=["user_id", "wm_poi_id", "wm_food_spu_id"]).copy()


def preprocess(raw_dir: Path, processed_dir: Path, sample_orders: int | None = 50000, seed: int = 42) -> None:
    raw_dir = Path(raw_dir)
    processed_dir = ensure_dir(processed_dir)
    rng = np.random.default_rng(seed)

    users = _safe_read(raw_dir, "users.txt")
    pois = _safe_read(raw_dir, "pois.txt")
    spus = _safe_read(raw_dir, "spus.txt")
    train = _safe_read(raw_dir, "orders_train.txt")
    test = _safe_read(raw_dir, "orders_test_poi.txt")
    raw_train_orders = len(train)
    raw_test_orders = len(test)

    require_columns(users, ["user_id"], "users")
    require_columns(pois, ["wm_poi_id"], "pois")
    require_columns(spus, ["wm_food_spu_id"], "spus")
    require_columns(train, ["user_id", "wm_poi_id"], "orders_train")
    require_columns(test, ["user_id"], "orders_test_poi")

    if sample_orders and len(train) > sample_orders:
        train = train.sample(n=sample_orders, random_state=seed).sort_index()
    train = train.dropna(subset=["user_id", "wm_poi_id"]).copy()
    test = test.dropna(subset=["user_id"]).copy()

    for frame in [train, test]:
        if "wm_order_id" not in frame.columns:
            frame["wm_order_id"] = [f"generated_{i}" for i in range(len(frame))]
        if "order_price" not in frame.columns:
            if "order_price_interval" in frame.columns:
                frame["order_price"] = frame["order_price_interval"].map(_price_interval_to_midpoint)
            else:
                frame["order_price"] = np.nan
        if "ord_period_name" not in frame.columns:
            frame["ord_period_name"] = "unknown"
        if "aoi_id" not in frame.columns:
            frame["aoi_id"] = "unknown_aoi"
        if "order_timestamp" not in frame.columns:
            frame["order_timestamp"] = np.arange(len(frame))

    if (raw_dir / "orders_poi_test_label.txt").exists():
        labels = _safe_read(raw_dir, "orders_poi_test_label.txt")
        label_cols = [c for c in ["wm_order_id", "user_id", "wm_poi_id"] if c in labels.columns]
        if "user_id" not in label_cols or "wm_poi_id" not in label_cols:
            labels = test[["wm_order_id", "user_id", "wm_poi_id"]].copy()
        else:
            labels = labels[label_cols].dropna()
    else:
        labels = test[["wm_order_id", "user_id", "wm_poi_id"]].copy()

    # Keep test users that also have train history, because offline recommenders need a profile.
    train_users = set(train["user_id"].astype(str))
    labels = labels[labels["user_id"].astype(str).isin(train_users)].copy()
    test = test[test["user_id"].astype(str).isin(set(labels["user_id"].astype(str)))].copy()
    if "wm_poi_id" not in test.columns:
        test = test.merge(labels[["wm_order_id", "wm_poi_id"]].drop_duplicates("wm_order_id"), on="wm_order_id", how="left")

    merchant_stats = (
        train.groupby("wm_poi_id")
        .agg(order_count=("wm_poi_id", "size"), avg_order_price=("order_price", "mean"))
        .reset_index()
    )
    pois = pois.merge(merchant_stats, on="wm_poi_id", how="left")
    pois["order_count"] = pois["order_count"].fillna(0).astype(int)
    pois["avg_order_price"] = pois["avg_order_price"].fillna(train["order_price"].median())
    for col in ["poi_score", "delivery_comment_avg_score", "food_comment_avg_score"]:
        if col not in pois.columns:
            pois[col] = 4.2
        pois[col] = pd.to_numeric(pois[col], errors="coerce").fillna(pois[col].median())
    tag_fallbacks = {
        "primary_first_tag_id": "primary_first_tag_name",
        "primary_second_tag_id": "primary_second_tag_name",
        "primary_third_tag_id": "primary_third_tag_name",
    }
    for col, fallback in tag_fallbacks.items():
        if col not in pois.columns:
            pois[col] = pois[fallback] if fallback in pois.columns else "unknown"
    if "aor_id" not in pois.columns:
        pois["aor_id"] = "unknown"
    for col in ["primary_first_tag_id", "primary_second_tag_id", "primary_third_tag_id", "aor_id"]:
        pois[col] = pois[col].fillna("unknown").astype(str)

    users = users[users["user_id"].astype(str).isin(train_users)].copy()
    if "avg_pay_amt" not in users.columns:
        users["avg_pay_amt"] = train.groupby("user_id")["order_price"].mean().reindex(users["user_id"]).values
    users["avg_pay_amt"] = pd.to_numeric(users["avg_pay_amt"], errors="coerce").fillna(train["order_price"].median())

    user_profile = (
        train.groupby("user_id")
        .agg(history_orders=("wm_poi_id", "size"), avg_order_price=("order_price", "mean"))
        .reset_index()
    )
    fav_cat = (
        train.merge(pois[["wm_poi_id", "primary_first_tag_id"]], on="wm_poi_id", how="left")
        .groupby(["user_id", "primary_first_tag_id"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["user_id", "cnt"], ascending=[True, False])
        .drop_duplicates("user_id")
        .rename(columns={"primary_first_tag_id": "favorite_category"})
    )
    users = users.merge(user_profile, on="user_id", how="left").merge(
        fav_cat[["user_id", "favorite_category"]], on="user_id", how="left"
    )
    users["history_orders"] = users["history_orders"].fillna(0).astype(int)
    users["avg_order_price"] = users["avg_order_price"].fillna(users["avg_pay_amt"])
    users["favorite_category"] = users["favorite_category"].fillna("unknown")

    pois = _add_geo(pois, "aor_id", seed + 7)
    user_aoi = train.groupby("user_id")["aoi_id"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
    users["aoi_id"] = users["user_id"].map(user_aoi).fillna("unknown_aoi")
    users = _add_geo(users, "aoi_id", seed + 17)

    train["split"] = "train"
    test["split"] = "test"
    session_interactions = _build_session_interactions(raw_dir, test)
    order_spus_train = _build_order_spus(raw_dir, "orders_spu_train.txt", train)
    order_spus_test = _build_order_spus(raw_dir, "orders_test_spu.txt", test)

    write_csv(users, processed_dir / "users.csv")
    write_csv(pois, processed_dir / "merchants.csv")
    write_csv(spus, processed_dir / "spus.csv")
    write_csv(train, processed_dir / "orders_train.csv")
    write_csv(test, processed_dir / "orders_test.csv")
    write_csv(labels, processed_dir / "test_interactions.csv")
    write_csv(session_interactions, processed_dir / "session_interactions.csv")
    write_csv(order_spus_train, processed_dir / "order_spus_train.csv")
    write_csv(order_spus_test, processed_dir / "order_spus_test.csv")

    data_note = {
        "source": "Takeout Recommendation Dataset (TRD), Zenodo DOI 10.5281/zenodo.8025855",
        "mode": "mock" if (raw_dir / "MOCK_DATASET").exists() else "trd",
        "train_orders": int(len(train)),
        "raw_train_orders": int(raw_train_orders),
        "test_orders": int(len(test)),
        "raw_test_orders": int(raw_test_orders),
        "test_labels": int(len(labels)),
        "session_interactions": int(len(session_interactions)),
        "order_spus_train": int(len(order_spus_train)),
        "order_spus_test": int(len(order_spus_test)),
        "users": int(len(users)),
        "merchants": int(len(pois)),
        "sample_orders": sample_orders,
        "train_sample_fraction": float(len(train) / raw_train_orders) if raw_train_orders else 0.0,
        "seed": seed,
        "rider_data": "synthetic, generated by FoodFlow simulation with fixed seed",
    }
    pd.Series(data_note).to_json(processed_dir / "data_note.json", force_ascii=False, indent=2)
