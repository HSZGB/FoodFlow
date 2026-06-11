from __future__ import annotations

import math

import numpy as np
import pandas as pd


def haversine_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = values.min(), values.max()
    if hi == lo:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - lo) / (hi - lo)


def estimate_user_merchant_eta(user_row: pd.Series, merchant_row: pd.Series, period: str = "lunch") -> float:
    distance = haversine_km(
        float(user_row.get("lng", 116.40)),
        float(user_row.get("lat", 39.92)),
        float(merchant_row.get("lng", 116.40)),
        float(merchant_row.get("lat", 39.92)),
    )
    prep = 10.0 + (1.0 - float(merchant_row.get("delivery_comment_avg_score", 4.2)) / 5.0) * 8.0
    peak = 6.0 if period in {"lunch", "dinner"} else 2.0
    return float(prep + distance / 18.0 * 60.0 + peak)


def fairness_scores(merchants: pd.DataFrame) -> dict[str, float]:
    order_count = pd.to_numeric(merchants["order_count"], errors="coerce").fillna(0)
    popularity = minmax(order_count)
    quality = minmax(pd.to_numeric(merchants["poi_score"], errors="coerce").fillna(4.2))
    fair = 0.75 * (1 - popularity) + 0.25 * quality
    return dict(zip(merchants["wm_poi_id"].astype(str), fair.astype(float)))


def supply_score_for_merchant(merchant_row: pd.Series) -> float:
    order_count = float(merchant_row.get("order_count", 0) or 0)
    delivery_score = float(merchant_row.get("delivery_comment_avg_score", 4.2) or 4.2)
    return float(0.6 * (delivery_score / 5.0) + 0.4 * (1.0 / (1.0 + np.log1p(order_count))))
