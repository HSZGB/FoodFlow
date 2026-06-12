from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from foodflow.data import PreparedData
from foodflow.explain import explain_recommendation
from foodflow.recommenders import OursFullRecommender
from foodflow.rider_sim import assign_order, generate_riders


st.set_page_config(page_title="FoodFlow", layout="wide")
st.title("FoodFlow 外卖三方推荐与履约仿真")
st.caption("用户侧推荐商家，商家侧关注曝光公平，骑手侧关注订单匹配与履约效率。")

processed_dir = Path("data/processed")
if not (processed_dir / "users.csv").exists():
    st.warning("请先运行 `make mock preprocess eval simulate figures report` 或下载真实 TRD 后运行 `make preprocess`。")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_data() -> PreparedData:
    return PreparedData.load(processed_dir)


@st.cache_resource(show_spinner=False)
def load_model(_data: PreparedData) -> OursFullRecommender:
    return OursFullRecommender().fit(_data)


data = load_data()
users = data.user_ids
user_id = st.sidebar.selectbox("用户", users)
period = st.sidebar.selectbox("时段", ["breakfast", "lunch", "dinner", "night"], index=1)

model = load_model(data)
rec = model.recommend([user_id], 10, {user_id: period})
recs = rec.recommendations[user_id]
merchants = data.merchants.set_index("wm_poi_id", drop=False)
users_df = data.users.set_index("user_id", drop=False)

tab_case, tab_metrics, tab_figures = st.tabs(["三方推荐案例", "实验指标对比", "图表看板"])

with tab_case:
    st.subheader("用户画像")
    user_row = users_df.loc[user_id]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("用户ID", user_id)
    col2.metric("历史订单数", int(user_row.get("history_orders", 0)))
    col3.metric("平均消费", f"{float(user_row.get('avg_order_price', user_row.get('avg_pay_amt', 0))):.2f}")
    col4.metric("常点品类", str(user_row.get("favorite_category", "unknown")))

    st.subheader("Top-K 推荐与解释")
    rows = []
    for rank, merchant_id in enumerate(recs, start=1):
        m = merchants.loc[merchant_id]
        components = model.component_scores(user_id, merchant_id, period)
        rows.append(
            {
                "rank": rank,
                "merchant": merchant_id,
                "category": m.get("primary_first_tag_id", "unknown"),
                "final_score": round(components["final_score"], 4),
                "user_score": round(components["user_score"], 4),
                "fairness": round(components["merchant_fairness"], 4),
                "eta_min": round(components["eta_minutes"], 1),
                "supply": round(components["supply_score"], 4),
                "explanation": explain_recommendation(data, user_id, merchant_id, period),
            }
        )
    rec_df = pd.DataFrame(rows)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    chosen = st.selectbox("模拟下单商家", recs)
    riders = generate_riders(data.merchants, n_riders=50, seed=7)
    rider_id, eta = assign_order(users_df.loc[user_id], merchants.loc[chosen], riders, "load_aware", period, 0)
    st.subheader("订单-骑手匹配")
    col1, col2, col3 = st.columns(3)
    col1.metric("匹配骑手", rider_id)
    col2.metric("预计送达时间", f"{eta:.1f} 分钟")
    col3.metric("匹配策略", "LoadAware")
    st.info("这一步体现骑手侧：推荐列表产生订单后，系统把订单匹配给 ETA、负载和可靠性更合适的骑手。")

offline_path = Path("outputs/results/offline_metrics.csv")
sim_path = Path("outputs/results/simulation_metrics.csv")
figures_dir = Path("outputs/figures")

with tab_metrics:
    st.subheader("实验结果")
    if offline_path.exists():
        offline = pd.read_csv(offline_path)
        top_user = offline.sort_values("Recall@20", ascending=False).iloc[0]
        ours = offline[offline["model"].astype(str).str.contains("Ours-Full")]
        col1, col2, col3 = st.columns(3)
        col1.metric("最高 Recall@20", f"{top_user['Recall@20']:.4f}", top_user["model"])
        if not ours.empty:
            col2.metric("Ours-Full Recall@20", f"{ours.iloc[0]['Recall@20']:.4f}")
            col3.metric("Ours-Full NDCG@20", f"{ours.iloc[0]['NDCG@20']:.4f}")
        st.dataframe(offline, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(offline, x="model", y="Recall@20", title="用户侧：Recall@20"), use_container_width=True)
    if sim_path.exists():
        sim = pd.read_csv(sim_path)
        best_utility = sim.sort_values("platform_utility", ascending=False).iloc[0]
        best_eta = sim.sort_values("avg_eta").iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("最高平台效用", f"{best_utility['platform_utility']:.4f}", best_utility["policy"])
        col2.metric("最低 Avg ETA", f"{best_eta['avg_eta']:.2f}", best_eta["policy"])
        col3.metric("策略数量", len(sim))
        st.dataframe(sim, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.scatter(
                sim,
                x="avg_eta",
                y="platform_utility",
                color="policy",
                size="completed_orders",
                title="履约侧：ETA 与平台效用权衡",
            ),
            use_container_width=True,
        )

with tab_figures:
    st.subheader("图表看板")
    if not figures_dir.exists():
        st.warning("尚未生成图表，请先运行 `make figures`。")
    else:
        figure_files = sorted(figures_dir.glob("*.png"))
        if not figure_files:
            st.warning("尚未生成图表，请先运行 `make figures`。")
        for i in range(0, len(figure_files), 2):
            cols = st.columns(2)
            for col, fig in zip(cols, figure_files[i : i + 2]):
                col.image(str(fig), caption=fig.name, use_container_width=True)
