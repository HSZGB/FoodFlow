from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from .data import PreparedData


KG_COLUMNS = ["head", "relation", "tail", "evidence"]


@dataclass(frozen=True)
class KGPathSummary:
    user_id: str
    merchant_id: str
    repeat_orders: int
    category: str
    category_orders: int
    area: str
    area_orders: int
    price_bucket: str
    user_price_bucket: str
    paths: list[str]
    triples: list[tuple[str, str, str]]


def _clean(value: object, default: str = "unknown") -> str:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def price_bucket(value: object) -> str:
    price = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(price):
        return "price_unknown"
    price = float(price)
    if price < 25:
        return "price_low"
    if price < 45:
        return "price_mid"
    if price < 70:
        return "price_high"
    return "price_premium"


def build_lightweight_triples(
    data: PreparedData,
    user_limit: int | None = None,
    max_history_per_user: int = 30,
) -> pd.DataFrame:
    """Build lightweight KG triples for explainable features without training a graph model."""
    rows: list[dict[str, str]] = []
    merchants = data.merchants.copy()
    merchants["wm_poi_id"] = merchants["wm_poi_id"].astype(str)
    for _, merchant in merchants.iterrows():
        merchant_id = _clean(merchant.get("wm_poi_id"))
        category = _clean(merchant.get("primary_first_tag_id"))
        area = _clean(merchant.get("aor_id"))
        bucket = price_bucket(merchant.get("avg_order_price"))
        rows.extend(
            [
                {
                    "head": f"poi:{merchant_id}",
                    "relation": "has_category",
                    "tail": f"category:{category}",
                    "evidence": "merchants.primary_first_tag_id",
                },
                {
                    "head": f"poi:{merchant_id}",
                    "relation": "located_in_area",
                    "tail": f"area:{area}",
                    "evidence": "merchants.aor_id",
                },
                {
                    "head": f"poi:{merchant_id}",
                    "relation": "has_price_range",
                    "tail": f"price:{bucket}",
                    "evidence": "merchants.avg_order_price",
                },
            ]
        )

    users = data.users.copy()
    users["user_id"] = users["user_id"].astype(str)
    if user_limit is not None:
        users = users.head(user_limit)
    for _, user in users.iterrows():
        user_id = _clean(user.get("user_id"))
        category = _clean(user.get("favorite_category"))
        bucket = price_bucket(user.get("avg_order_price", user.get("avg_pay_amt")))
        rows.extend(
            [
                {
                    "head": f"user:{user_id}",
                    "relation": "prefers_category",
                    "tail": f"category:{category}",
                    "evidence": "users.favorite_category",
                },
                {
                    "head": f"user:{user_id}",
                    "relation": "has_price_range",
                    "tail": f"price:{bucket}",
                    "evidence": "users.avg_order_price",
                },
            ]
        )

    orders = data.orders_train.copy()
    orders["user_id"] = orders["user_id"].astype(str)
    orders["wm_poi_id"] = orders["wm_poi_id"].astype(str)
    if user_limit is not None:
        allowed_users = set(users["user_id"].astype(str))
        orders = orders[orders["user_id"].isin(allowed_users)]
    if "order_timestamp" in orders.columns:
        orders = orders.sort_values(["user_id", "order_timestamp"], ascending=[True, False])
    for user_id, group in orders.groupby("user_id", sort=False):
        for _, order in group.head(max_history_per_user).iterrows():
            rows.append(
                {
                    "head": f"user:{_clean(user_id)}",
                    "relation": "ordered_poi",
                    "tail": f"poi:{_clean(order.get('wm_poi_id'))}",
                    "evidence": "orders_train.wm_poi_id",
                }
            )
            if "wm_food_spu_id" in order and not pd.isna(order.get("wm_food_spu_id")):
                rows.append(
                    {
                        "head": f"poi:{_clean(order.get('wm_poi_id'))}",
                        "relation": "sells_spu",
                        "tail": f"spu:{_clean(order.get('wm_food_spu_id'))}",
                        "evidence": "orders_train.wm_food_spu_id",
                    }
                )

    return pd.DataFrame(rows, columns=KG_COLUMNS).drop_duplicates(ignore_index=True)


def kg_path_summary(data: PreparedData, user_id: str, merchant_id: str) -> KGPathSummary:
    users = data.users.set_index("user_id", drop=False)
    merchants = data.merchants.set_index("wm_poi_id", drop=False)
    user_id = str(user_id)
    merchant_id = str(merchant_id)
    if user_id not in users.index or merchant_id not in merchants.index:
        return KGPathSummary(user_id, merchant_id, 0, "unknown", 0, "unknown", 0, "price_unknown", "price_unknown", [], [])

    user = users.loc[user_id]
    merchant = merchants.loc[merchant_id]
    history = data.orders_train[data.orders_train["user_id"].astype(str) == user_id].copy()
    merchant_history = history.merge(
        data.merchants[["wm_poi_id", "primary_first_tag_id", "aor_id"]],
        on="wm_poi_id",
        how="left",
        suffixes=("", "_merchant"),
    )
    repeat_orders = int((history["wm_poi_id"].astype(str) == merchant_id).sum())
    category = _clean(merchant.get("primary_first_tag_id"))
    area = _clean(merchant.get("aor_id"))
    category_counts = Counter(merchant_history["primary_first_tag_id"].astype(str))
    area_col = "aor_id_merchant" if "aor_id_merchant" in merchant_history.columns else "aor_id"
    area_counts = Counter(merchant_history[area_col].astype(str)) if area_col in merchant_history.columns else Counter()
    category_orders = int(category_counts.get(category, 0))
    area_orders = int(area_counts.get(area, 0))
    merchant_price_bucket = price_bucket(merchant.get("avg_order_price"))
    user_price_bucket = price_bucket(user.get("avg_order_price", user.get("avg_pay_amt")))

    paths: list[str] = []
    triples: list[tuple[str, str, str]] = []
    if repeat_orders:
        paths.append(f"user:{user_id} -[ordered_poi x{repeat_orders}]-> poi:{merchant_id}")
        triples.append((f"user:{user_id}", "ordered_poi", f"poi:{merchant_id}"))
    if category_orders:
        paths.append(
            f"user:{user_id} -[prefers_category x{category_orders}]-> category:{category} <-[has_category]- poi:{merchant_id}"
        )
        triples.extend(
            [
                (f"user:{user_id}", "prefers_category", f"category:{category}"),
                (f"poi:{merchant_id}", "has_category", f"category:{category}"),
            ]
        )
    if area_orders:
        paths.append(f"user:{user_id} -[orders_in_area x{area_orders}]-> area:{area} <-[located_in_area]- poi:{merchant_id}")
        triples.extend(
            [
                (f"user:{user_id}", "orders_in_area", f"area:{area}"),
                (f"poi:{merchant_id}", "located_in_area", f"area:{area}"),
            ]
        )
    if merchant_price_bucket == user_price_bucket and merchant_price_bucket != "price_unknown":
        paths.append(
            f"user:{user_id} -[has_price_range]-> price:{user_price_bucket} <-[has_price_range]- poi:{merchant_id}"
        )
        triples.extend(
            [
                (f"user:{user_id}", "has_price_range", f"price:{user_price_bucket}"),
                (f"poi:{merchant_id}", "has_price_range", f"price:{merchant_price_bucket}"),
            ]
        )

    deduped_triples = list(dict.fromkeys(triples))
    return KGPathSummary(
        user_id=user_id,
        merchant_id=merchant_id,
        repeat_orders=repeat_orders,
        category=category,
        category_orders=category_orders,
        area=area,
        area_orders=area_orders,
        price_bucket=merchant_price_bucket,
        user_price_bucket=user_price_bucket,
        paths=paths,
        triples=deduped_triples,
    )


def kg_explanation_parts(summary: KGPathSummary, max_paths: int = 3) -> list[str]:
    parts: list[str] = []
    if summary.repeat_orders:
        parts.append(f"KG路径显示你历史下单过该商家 {summary.repeat_orders} 次")
    if summary.category_orders:
        parts.append(f"你有 {summary.category_orders} 单历史订单连接到同品类 {summary.category}")
    if summary.area_orders:
        parts.append(f"你曾在同商圈/区域 {summary.area} 下单 {summary.area_orders} 次")
    if summary.price_bucket == summary.user_price_bucket and summary.price_bucket != "price_unknown":
        parts.append(f"商家价格段 {summary.price_bucket} 与你的消费区间一致")
    if summary.paths:
        parts.append("证据路径：" + "；".join(summary.paths[:max_paths]))
    return parts
