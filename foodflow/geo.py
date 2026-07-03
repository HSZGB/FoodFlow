"""Real-city geography for FoodFlow entities.

TRD 本身不含坐标，旧实现用高斯随机数合成经纬度（且同一 aor/aoi 的实体共享
同一个点），配送距离与 ETA 缺乏物理意义。本模块把用户/商家嵌入 LaDe 真实
城市的末端配送 GPS 分布：商圈映射到真实高密度簇，实体坐标从真实配送点
采样。这不等于拿到了 TRD 的真实位置——诚实口径是"TRD 订单嵌入 LaDe
真实城市空间分布"，但距离、ETA 与地图展示从此有真实街区尺度。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# 道路弯曲系数：直线距离换算道路距离的经验倍率（城市路网文献常用 1.2-1.4）。
ROAD_CIRCUITY_FACTOR = 1.3


def load_delivery_points(
    path: Path,
    max_points: int = 120_000,
    seed: int = 42,
    core_radius_km: float = 10.0,
) -> pd.DataFrame:
    """Load real delivery GPS points and keep only the densest urban core.

    LaDe 城市文件覆盖整个都市圈（烟台跨度约 200 公里），直接使用会把用户
    撒到远郊、产生几小时的"配送"。外卖是城市尺度业务，这里先粗聚类找到
    最大密度簇，再保留其质心 `core_radius_km` 公里内的点作为城市核心区。
    """
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for lng_col, lat_col in (("lng", "lat"), ("delivery_gps_lng", "delivery_gps_lat")):
        if lng_col in frame.columns and lat_col in frame.columns:
            points = frame[[lng_col, lat_col]].rename(columns={lng_col: "lng", lat_col: "lat"})
            break
    else:
        raise ValueError(f"{path} has no recognizable lng/lat columns")
    points = points.apply(pd.to_numeric, errors="coerce").dropna()
    # 剪掉坐标离群点（GPS 漂移）。
    lo = points.quantile(0.005)
    hi = points.quantile(0.995)
    points = points[
        points["lng"].between(lo["lng"], hi["lng"]) & points["lat"].between(lo["lat"], hi["lat"])
    ].reset_index(drop=True)

    if core_radius_km and len(points) > 1000:
        coarse = _cluster_points(points, n_clusters=12, seed=seed)
        core_center = points[coarse == coarse.value_counts().index[0]].mean()
        lat_rad = np.radians(float(core_center["lat"]))
        dlng_km = (points["lng"] - float(core_center["lng"])) * 111.32 * np.cos(lat_rad)
        dlat_km = (points["lat"] - float(core_center["lat"])) * 110.57
        within_core = np.sqrt(dlng_km**2 + dlat_km**2) <= core_radius_km
        points = points[within_core].reset_index(drop=True)

    if len(points) > max_points:
        points = points.sample(n=max_points, random_state=seed).reset_index(drop=True)
    return points


def _cluster_points(points: pd.DataFrame, n_clusters: int, seed: int) -> pd.Series:
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3, batch_size=4096)
    return pd.Series(model.fit_predict(points[["lng", "lat"]].to_numpy()), index=points.index)


def _sample_anchor(points: pd.DataFrame, rng: np.random.Generator, jitter_deg: float) -> tuple[float, float]:
    row = points.iloc[int(rng.integers(0, len(points)))]
    return (
        float(row["lng"] + rng.normal(0, jitter_deg)),
        float(row["lat"] + rng.normal(0, jitter_deg)),
    )


def assign_real_geography(
    users: pd.DataFrame,
    merchants: pd.DataFrame,
    points: pd.DataFrame,
    seed: int = 42,
    n_clusters: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Assign real-city coordinates to users and merchants.

    - 每个 aor_id（商圈）映射到一个真实高密度簇，商家坐标从簇内真实点采样
      （约 60m 抖动），同商圈商家在地图上聚成真实的商业区。
    - 每个 aoi_id（居住区）取一个真实锚点，用户在锚点周围约 120m 内散布。
    """
    if points.empty:
        raise ValueError("No delivery points available for geography assignment")
    rng = np.random.default_rng(seed)
    labels = _cluster_points(points, n_clusters=n_clusters, seed=seed)
    cluster_sizes = labels.value_counts()

    merchants = merchants.copy()
    users = users.copy()

    aor_ids = sorted(merchants["aor_id"].astype(str).unique()) if "aor_id" in merchants.columns else ["unknown"]
    top_clusters = cluster_sizes.index[: max(len(aor_ids), 1)].tolist()
    aor_to_cluster = {aor: top_clusters[i % len(top_clusters)] for i, aor in enumerate(aor_ids)}

    merchant_lng = np.zeros(len(merchants))
    merchant_lat = np.zeros(len(merchants))
    aor_values = merchants["aor_id"].astype(str).to_numpy() if "aor_id" in merchants.columns else np.full(len(merchants), "unknown")
    cluster_members = {cluster: points[labels == cluster] for cluster in top_clusters}
    for i, aor in enumerate(aor_values):
        members = cluster_members[aor_to_cluster[aor]]
        merchant_lng[i], merchant_lat[i] = _sample_anchor(members, rng, jitter_deg=0.0006)
    merchants["lng"] = merchant_lng
    merchants["lat"] = merchant_lat

    aoi_values = users["aoi_id"].astype(str).to_numpy() if "aoi_id" in users.columns else users.index.astype(str).to_numpy()
    aoi_anchor: dict[str, tuple[float, float]] = {}
    for aoi in pd.unique(aoi_values):
        aoi_anchor[aoi] = _sample_anchor(points, rng, jitter_deg=0.0)
    user_lng = np.zeros(len(users))
    user_lat = np.zeros(len(users))
    for i, aoi in enumerate(aoi_values):
        anchor = aoi_anchor[aoi]
        user_lng[i] = anchor[0] + rng.normal(0, 0.0012)
        user_lat[i] = anchor[1] + rng.normal(0, 0.0010)
    users["lng"] = user_lng
    users["lat"] = user_lat

    meta = {
        "geo_mode": "lade-real-city",
        "n_points": int(len(points)),
        "n_clusters": int(n_clusters),
        "merchant_groups": len(aor_ids),
        "user_groups": int(len(aoi_anchor)),
        "bbox": {
            "lng_min": float(points["lng"].min()),
            "lng_max": float(points["lng"].max()),
            "lat_min": float(points["lat"].min()),
            "lat_max": float(points["lat"].max()),
        },
        "note": (
            "TRD does not ship coordinates; entities are embedded into the real "
            "spatial distribution of LaDe last-mile delivery GPS points. Distances "
            "and ETAs are physically meaningful at city scale, but individual "
            "entity locations remain synthetic assignments."
        ),
    }
    return users, merchants, meta


def geocode_processed(
    processed_dir: Path,
    tasks_path: Path,
    seed: int = 42,
    n_clusters: int = 24,
) -> dict[str, object]:
    """Rewrite lng/lat in processed users.csv / merchants.csv with real geography."""
    processed_dir = Path(processed_dir)
    users = pd.read_csv(processed_dir / "users.csv")
    merchants = pd.read_csv(processed_dir / "merchants.csv")
    points = load_delivery_points(tasks_path, seed=seed)
    users, merchants, meta = assign_real_geography(users, merchants, points, seed=seed, n_clusters=n_clusters)
    users.to_csv(processed_dir / "users.csv", index=False)
    merchants.to_csv(processed_dir / "merchants.csv", index=False)
    meta["source"] = str(tasks_path)
    meta["seed"] = int(seed)
    (processed_dir / "geo_note.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
