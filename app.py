from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from foodflow.data import PreparedData
from foodflow.explain import explain_recommendation
from foodflow.recommenders import OursFullRecommender
from foodflow.rider_sim import assign_order, generate_riders


st.set_page_config(page_title="FoodFlow", layout="wide")
st.title("FoodFlow 外卖三方推荐与履约仿真")

processed_dir = Path("data/processed")
if not (processed_dir / "users.csv").exists():
    st.warning("请先运行 `make mock preprocess eval simulate figures report` 或下载真实 TRD 后运行 `make preprocess`。")
    st.stop()

data = PreparedData.load(processed_dir)
users = data.user_ids
user_id = st.sidebar.selectbox("用户", users)
period = st.sidebar.selectbox("时段", ["breakfast", "lunch", "dinner", "night"], index=1)

model = OursFullRecommender().fit(data)
rec = model.recommend([user_id], 10, {user_id: period})
recs = rec.recommendations[user_id]
merchants = data.merchants.set_index("wm_poi_id", drop=False)
users_df = data.users.set_index("user_id", drop=False)

st.subheader("用户画像")
user_row = users_df.loc[user_id]
st.write(
    {
        "用户ID": user_id,
        "历史订单数": int(user_row.get("history_orders", 0)),
        "平均消费": round(float(user_row.get("avg_order_price", user_row.get("avg_pay_amt", 0))), 2),
        "常点品类": user_row.get("favorite_category", "unknown"),
    }
)

st.subheader("Top-K 推荐与解释")
rows = []
for rank, merchant_id in enumerate(recs, start=1):
    m = merchants.loc[merchant_id]
    rows.append(
        {
            "rank": rank,
            "merchant": merchant_id,
            "category": m.get("primary_first_tag_id", "unknown"),
            "score": round(rec.scores[user_id][merchant_id], 4),
            "explanation": explain_recommendation(data, user_id, merchant_id, period),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True)

chosen = st.selectbox("模拟下单商家", recs)
riders = generate_riders(data.merchants, n_riders=50, seed=7)
rider_id, eta = assign_order(users_df.loc[user_id], merchants.loc[chosen], riders, "load_aware", period, 0)
st.subheader("骑手匹配")
st.metric("匹配骑手", rider_id)
st.metric("预计送达时间", f"{eta:.1f} 分钟")

st.subheader("实验结果")
offline_path = Path("outputs/results/offline_metrics.csv")
sim_path = Path("outputs/results/simulation_metrics.csv")
col1, col2 = st.columns(2)
with col1:
    if offline_path.exists():
        st.dataframe(pd.read_csv(offline_path), use_container_width=True)
with col2:
    if sim_path.exists():
        st.dataframe(pd.read_csv(sim_path), use_container_width=True)
