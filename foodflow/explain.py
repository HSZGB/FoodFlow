from __future__ import annotations

from collections import Counter

import pandas as pd

from .data import PreparedData
from .kg import kg_explanation_parts, kg_path_summary
from .rerank import estimate_user_merchant_eta, fairness_scores


def explain_recommendation(data: PreparedData, user_id: str, merchant_id: str, period: str = "lunch") -> str:
    users = data.users.set_index("user_id", drop=False)
    merchants = data.merchants.set_index("wm_poi_id", drop=False)
    if user_id not in users.index or merchant_id not in merchants.index:
        return "用户或商家不在当前样本中。"
    user = users.loc[user_id]
    merchant = merchants.loc[merchant_id]
    history = data.orders_train[data.orders_train["user_id"].astype(str) == str(user_id)]
    repeat = int((history["wm_poi_id"].astype(str) == str(merchant_id)).sum())
    category = str(merchant.get("primary_first_tag_id", "unknown"))
    user_categories = Counter(
        history.merge(data.merchants[["wm_poi_id", "primary_first_tag_id"]], on="wm_poi_id", how="left")[
            "primary_first_tag_id"
        ].astype(str)
    )
    eta = estimate_user_merchant_eta(user, merchant, period)
    fair = fairness_scores(data.merchants).get(str(merchant_id), 0.0)
    kg_summary = kg_path_summary(data, user_id, merchant_id)
    reasons = []
    reasons.extend(kg_explanation_parts(kg_summary))
    if not reasons:
        if repeat > 0:
            reasons.append(f"用户历史复购该商家 {repeat} 次")
        if user_categories.get(category, 0) > 0:
            reasons.append(f"用户常点品类与商家品类 {category} 匹配")
    reasons.append(f"预计履约时间约 {eta:.1f} 分钟")
    reasons.append(f"商家曝光补偿分 {fair:.2f}，有助于降低头部商家垄断")
    return "；".join(reasons) + "。"


def build_case_table(data: PreparedData, recommendations: dict[str, list[str]], limit: int = 8) -> pd.DataFrame:
    rows = []
    for user_id, recs in list(recommendations.items())[:limit]:
        for rank, merchant_id in enumerate(recs[:3], start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "rank": rank,
                    "wm_poi_id": merchant_id,
                    "explanation": explain_recommendation(data, user_id, merchant_id),
                }
            )
    return pd.DataFrame(rows)
