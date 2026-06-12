from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from foodflow.data import PreparedData
from foodflow.demo_support import (
    build_recommendation_frame,
    build_peak_trace,
    build_rider_policy_frame,
    clamp01,
    demo_user_cases,
    streamlit_image_width_kwargs,
    user_category_profile,
)
from foodflow.recommenders import OursBalancedRecommender, OursFullRecommender, UserOnlyRecommender
from foodflow.rider_sim import generate_riders


st.set_page_config(page_title="FoodFlow", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    .ff-title { font-size: 2rem; font-weight: 760; margin-bottom: 0.1rem; }
    .ff-subtitle { color: #4b5563; font-size: 0.98rem; margin-bottom: 1rem; }
    .ff-card {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 0.85rem 0.9rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        min-height: 255px;
    }
    .ff-card-selected { border: 2px solid #0f766e; }
    .ff-rank { color: #0f766e; font-size: 0.84rem; font-weight: 720; }
    .ff-name { color: #111827; font-size: 1.04rem; font-weight: 720; margin: 0.2rem 0 0.1rem; }
    .ff-meta { color: #64748b; font-size: 0.82rem; line-height: 1.35; }
    .ff-reason { color: #334155; font-size: 0.84rem; min-height: 2.4rem; margin: 0.35rem 0 0.55rem; }
    .ff-score { color: #0f172a; font-weight: 720; }
    .ff-bar { height: 7px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin: 0.12rem 0 0.42rem; }
    .ff-fill-user { height: 7px; background: #2563eb; }
    .ff-fill-fair { height: 7px; background: #0f766e; }
    .ff-fill-eta { height: 7px; background: #dc6803; }
    .ff-fill-supply { height: 7px; background: #7c3aed; }
    .ff-chip {
        display: inline-block;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        padding: 0.12rem 0.45rem;
        margin-right: 0.25rem;
        color: #334155;
        background: #f8fafc;
        font-size: 0.76rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="ff-title">FoodFlow 外卖三方推荐与履约仿真</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ff-subtitle">把“推荐商家”继续推进到“商家曝光”和“订单派给骑手”，用同一条链路展示用户、商家、骑手三方效果。</div>',
    unsafe_allow_html=True,
)

processed_dir = Path("data/processed")
if not (processed_dir / "users.csv").exists():
    st.warning("请先运行 `make download preprocess-full eval simulate audit figures report`，或用 `make conda-smoke` 生成 mock 数据。")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_data() -> PreparedData:
    return PreparedData.load(processed_dir)


@st.cache_resource(show_spinner=False)
def load_models(_data: PreparedData) -> dict[str, object]:
    return {
        "UserOnly": UserOnlyRecommender().fit(_data),
        "Ours-Balanced": OursBalancedRecommender().fit(_data),
        "Ours-Full": OursFullRecommender().fit(_data),
    }


def pct(value: float) -> str:
    return f"{clamp01(value) * 100:.0f}%"


def recommendation_card(row: pd.Series, selected: bool = False) -> str:
    card_class = "ff-card ff-card-selected" if selected else "ff-card"
    name = html.escape(str(row["merchant_name"]))
    category = html.escape(str(row["category"]))
    merchant_id = html.escape(str(row["merchant_id"]))
    reason = html.escape(str(row["reason"]))
    reason_chips = "".join(
        f'<span class="ff-chip">{html.escape(part)}</span>' for part in str(row["reason"]).split(" / ") if part
    )
    return f"""
    <div class="{card_class}">
      <div class="ff-rank">TOP {int(row['rank'])}</div>
      <div class="ff-name">{name}</div>
      <div class="ff-meta">
        ID {merchant_id} · 品类 {category}<br>
        评分 {float(row['poi_score']):.2f} · 均价 {float(row['avg_price']):.1f} · 距离 {float(row['distance_km']):.2f} km
      </div>
      <div class="ff-reason" aria-label="{reason}">{reason_chips}</div>
      <span class="ff-chip">总分 <span class="ff-score">{float(row['final_score']):.3f}</span></span>
      <span class="ff-chip">ETA {float(row['eta_minutes']):.1f} min</span>
      <div class="ff-meta" style="margin-top:0.55rem;">用户偏好</div>
      <div class="ff-bar"><div class="ff-fill-user" style="width:{pct(row['user_score'])};"></div></div>
      <div class="ff-meta">商家公平</div>
      <div class="ff-bar"><div class="ff-fill-fair" style="width:{pct(row['fairness'])};"></div></div>
      <div class="ff-meta">履约速度</div>
      <div class="ff-bar"><div class="ff-fill-eta" style="width:{pct(row['eta_score'])};"></div></div>
      <div class="ff-meta">供给稳定</div>
      <div class="ff-bar"><div class="ff-fill-supply" style="width:{pct(row['supply'])};"></div></div>
    </div>
    """


data = load_data()
models = load_models(data)
user_ids = data.user_ids
user_set = set(user_ids)
default_user = "8" if "8" in user_set else user_ids[0]
case_options = demo_user_cases(data.users)

with st.sidebar:
    st.header("演示参数")
    case_label = st.selectbox("快速案例", list(case_options.keys()), index=0)
    typed_user = st.text_input("手动用户 ID", value="", placeholder="留空则使用快速案例").strip()
    selected_case_user = case_options.get(case_label, default_user)
    typed_user = typed_user or selected_case_user
    if typed_user not in user_set:
        st.warning(f"用户 {typed_user} 不在当前处理数据中，已回退到 {default_user}。")
        user_id = default_user
    else:
        user_id = typed_user
    period_label = st.selectbox("时段", ["午餐高峰", "晚餐高峰", "早餐", "夜宵"], index=0)
    period_map = {"午餐高峰": "lunch", "晚餐高峰": "dinner", "早餐": "breakfast", "夜宵": "night"}
    period = period_map[period_label]
    strategy_name = st.selectbox("推荐策略", ["Ours-Full", "Ours-Balanced", "UserOnly"], index=0)
    top_k = st.slider("推荐数量", min_value=5, max_value=12, value=10, step=1)

model = models[strategy_name]
rec_result = model.recommend([user_id], top_k, {user_id: period})
recs = rec_result.recommendations[user_id]
rec_df = build_recommendation_frame(data, model, user_id, recs, period)

users_df = data.users.set_index("user_id", drop=False)
merchants = data.merchants.set_index("wm_poi_id", drop=False)
user_row = users_df.loc[user_id]
riders = generate_riders(data.merchants, n_riders=60, seed=7)

tab_case, tab_peak, tab_metrics, tab_figures = st.tabs(["推荐工作台", "高峰仿真回放", "指标故事线", "图表材料"])

with tab_case:
    profile_col, category_col = st.columns([1.05, 1.25])
    with profile_col:
        st.subheader("用户画像")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("用户 ID", user_id)
        m2.metric("历史订单", int(user_row.get("history_orders", 0)))
        m3.metric("平均消费", f"{float(user_row.get('avg_order_price', user_row.get('avg_pay_amt', 0))):.1f}")
        m4.metric("常点品类", str(user_row.get("favorite_category", "unknown")))

    with category_col:
        category_df = user_category_profile(data, user_id)
        if not category_df.empty:
            fig = px.bar(
                category_df.sort_values("orders"),
                x="orders",
                y="category",
                orientation="h",
                title="用户历史品类偏好",
                color="share",
                color_continuous_scale=["#dbeafe", "#2563eb"],
            )
            fig.update_layout(height=230, margin=dict(l=8, r=8, t=44, b=8), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{strategy_name} 推荐商家卡片")
    reason_choices = sorted({item for text in rec_df["reason"].astype(str) for item in text.split(" / ") if item})
    selected_reasons = st.multiselect("推荐理由筛选", reason_choices, default=[])
    if selected_reasons:
        card_df = rec_df[
            rec_df["reason"].astype(str).apply(lambda text: any(reason in text for reason in selected_reasons))
        ].copy()
    else:
        card_df = rec_df.copy()
    if card_df.empty:
        st.info("当前筛选条件下没有推荐商家，已展示完整推荐列表。")
        card_df = rec_df.copy()

    card_count = min(6, len(card_df))
    for start in range(0, card_count, 3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, card_df.iloc[start : start + 3].iterrows()):
            with col:
                st.markdown(recommendation_card(row), unsafe_allow_html=True)

    st.subheader("同一用户的策略对比")
    strategy_rows = []
    for name, candidate_model in models.items():
        candidate_recs = candidate_model.recommend([user_id], top_k, {user_id: period}).recommendations[user_id]
        candidate_frame = build_recommendation_frame(data, candidate_model, user_id, candidate_recs, period)
        strategy_rows.append(
            {
                "策略": name,
                "Top3 商家": " / ".join(candidate_frame["merchant_name"].head(3).astype(str).tolist()),
                "平均用户偏好": candidate_frame["user_score"].mean(),
                "平均商家公平": candidate_frame["fairness"].mean(),
                "平均 ETA": candidate_frame["eta_minutes"].mean(),
                "平均供给": candidate_frame["supply"].mean(),
                "与当前策略重合数": len(set(candidate_recs).intersection(set(recs))),
            }
        )
    strategy_df = pd.DataFrame(strategy_rows)
    compare_left, compare_right = st.columns([1.2, 1])
    with compare_left:
        st.dataframe(
            strategy_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "平均用户偏好": st.column_config.NumberColumn(format="%.3f"),
                "平均商家公平": st.column_config.NumberColumn(format="%.3f"),
                "平均 ETA": st.column_config.NumberColumn(format="%.1f"),
                "平均供给": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    with compare_right:
        strategy_chart = px.bar(
            strategy_df,
            x="策略",
            y=["平均用户偏好", "平均商家公平", "平均供给"],
            barmode="group",
            title="策略侧重点对比",
            color_discrete_sequence=["#2563eb", "#0f766e", "#7c3aed"],
        )
        strategy_chart.update_layout(height=300, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="")
        st.plotly_chart(strategy_chart, use_container_width=True)

    option_labels = [
        f"TOP {int(row.rank)} · {row.merchant_name} · ID {row.merchant_id}" for row in rec_df.itertuples(index=False)
    ]
    chosen_label = st.selectbox("模拟下单商家", option_labels)
    chosen_idx = option_labels.index(chosen_label)
    chosen_row = rec_df.iloc[chosen_idx]
    chosen_id = str(chosen_row["merchant_id"])
    merchant_row = merchants.loc[chosen_id]

    st.subheader("三方链路拆解")
    left, right = st.columns([1.05, 1.35])
    with left:
        st.markdown(recommendation_card(chosen_row, selected=True), unsafe_allow_html=True)
        rider_compare = build_rider_policy_frame(user_row, merchant_row, riders, period)
        load_aware = rider_compare[rider_compare["policy_key"] == "load_aware"].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("匹配骑手", str(load_aware["rider_id"]))
        c2.metric("预计 ETA", f"{float(load_aware['eta']):.1f} min")
        c3.metric("骑手负载", int(load_aware["load"]))

    with right:
        contrib = pd.DataFrame(
            {
                "component": ["用户偏好", "商家公平", "ETA 履约", "供给稳定"],
                "weighted_score": [
                    chosen_row["user_contrib"],
                    chosen_row["fairness_contrib"],
                    chosen_row["eta_contrib"],
                    chosen_row["supply_contrib"],
                ],
            }
        )
        contrib_fig = px.bar(
            contrib,
            x="component",
            y="weighted_score",
            color="component",
            title=f"{strategy_name} 加权分数组成",
            color_discrete_sequence=["#2563eb", "#0f766e", "#dc6803", "#7c3aed"],
        )
        contrib_fig.update_layout(showlegend=False, height=280, margin=dict(l=8, r=8, t=48, b=8))
        st.plotly_chart(contrib_fig, use_container_width=True)

        eta_fig = px.bar(
            rider_compare,
            x="policy",
            y="eta",
            color="policy",
            title="同一订单在不同骑手策略下的 ETA",
            color_discrete_sequence=["#64748b", "#2563eb", "#0f766e"],
        )
        eta_fig.update_layout(showlegend=False, height=270, margin=dict(l=8, r=8, t=48, b=8), yaxis_title="分钟")
        st.plotly_chart(eta_fig, use_container_width=True)

    flow_col, map_col = st.columns([0.9, 1.25])
    with flow_col:
        flow_fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=18,
                        thickness=18,
                        line=dict(color="#cbd5e1", width=1),
                        label=[
                            f"用户 {user_id}",
                            str(chosen_row["merchant_name"]),
                            f"骑手 {load_aware['rider_id']}",
                            f"履约 ETA {float(load_aware['eta']):.1f} min",
                        ],
                        color=["#2563eb", "#0f766e", "#dc6803", "#7c3aed"],
                    ),
                    link=dict(source=[0, 1, 2], target=[1, 2, 3], value=[1, 1, 1], color=["#bfdbfe", "#99f6e4", "#fed7aa"]),
                )
            ]
        )
        flow_fig.update_layout(title="用户 -> 商家 -> 骑手", height=340, margin=dict(l=8, r=8, t=52, b=8))
        st.plotly_chart(flow_fig, use_container_width=True)

    with map_col:
        rider_point = rider_compare[rider_compare["policy_key"] == "load_aware"].iloc[0]
        map_fig = go.Figure()
        map_fig.add_trace(
            go.Scatter(
                x=[float(user_row.get("lng", 116.40))],
                y=[float(user_row.get("lat", 39.92))],
                mode="markers+text",
                name="用户",
                text=[f"用户 {user_id}"],
                textposition="top center",
                marker=dict(size=15, color="#2563eb", symbol="circle"),
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=rec_df["lng"],
                y=rec_df["lat"],
                mode="markers",
                name="推荐商家",
                text=rec_df["merchant_name"],
                marker=dict(size=rec_df["final_score"].rank(pct=True) * 16 + 8, color="#0f766e", opacity=0.75),
                hovertemplate="%{text}<br>经度 %{x:.4f}<br>纬度 %{y:.4f}<extra></extra>",
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=[float(merchant_row.get("lng", 116.40))],
                y=[float(merchant_row.get("lat", 39.92))],
                mode="markers+text",
                name="下单商家",
                text=[str(chosen_row["merchant_name"])],
                textposition="bottom center",
                marker=dict(size=19, color="#dc6803", symbol="diamond"),
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=[float(rider_point["lng"])],
                y=[float(rider_point["lat"])],
                mode="markers+text",
                name="匹配骑手",
                text=[str(rider_point["rider_id"])],
                textposition="top center",
                marker=dict(size=17, color="#7c3aed", symbol="square"),
            )
        )
        map_fig.update_layout(
            title="空间分布：用户、推荐商家与匹配骑手",
            height=340,
            xaxis_title="经度",
            yaxis_title="纬度",
            margin=dict(l=8, r=8, t=52, b=8),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(map_fig, use_container_width=True)

    st.subheader("推荐明细")
    table_cols = [
        "rank",
        "merchant_name",
        "category",
        "final_score",
        "user_score",
        "fairness",
        "eta_minutes",
        "supply",
        "distance_km",
        "reason",
    ]
    display_df = rec_df[table_cols].rename(
        columns={
            "rank": "排名",
            "merchant_name": "商家",
            "category": "品类",
            "final_score": "总分",
            "user_score": "用户偏好",
            "fairness": "商家公平",
            "eta_minutes": "ETA",
            "supply": "供给",
            "distance_km": "距离km",
            "reason": "解释",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab_peak:
    st.subheader("午餐高峰仿真回放")
    trace_df = build_peak_trace(
        data,
        {
            "UserOnly + MinETA": (models["UserOnly"], "min_eta"),
            "Ours-Balanced": (models["Ours-Balanced"], "load_aware"),
            "Ours-Full": (models["Ours-Full"], "load_aware"),
        },
        seed=33,
        steps=6,
        requests_per_step=8,
        top_k=top_k,
    )
    if trace_df.empty:
        st.warning("当前数据没有可用于仿真的测试用户。")
    else:
        final_trace = trace_df.sort_values("step").groupby("policy", as_index=False).tail(1)
        p1, p2, p3, p4 = st.columns(4)
        best_eta_trace = final_trace.sort_values("avg_eta").iloc[0]
        best_timeout_trace = final_trace.sort_values("timeout_rate").iloc[0]
        p1.metric("回放订单数", int(final_trace["completed_orders"].max()))
        p2.metric("最低累计 Avg ETA", f"{best_eta_trace['avg_eta']:.1f}", str(best_eta_trace["policy"]))
        p3.metric("最低累计超时率", f"{best_timeout_trace['timeout_rate']:.3f}", str(best_timeout_trace["policy"]))
        p4.metric("仿真步数", int(trace_df["step"].max()))

        t1, t2 = st.columns(2)
        with t1:
            order_fig = px.line(
                trace_df,
                x="step",
                y="completed_orders",
                color="policy",
                markers=True,
                title="累计完成订单",
            )
            order_fig.update_layout(height=330, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="时间步")
            st.plotly_chart(order_fig, use_container_width=True)

            eta_line = px.line(
                trace_df,
                x="step",
                y="avg_eta",
                color="policy",
                markers=True,
                title="累计平均 ETA",
            )
            eta_line.update_layout(height=330, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="时间步", yaxis_title="分钟")
            st.plotly_chart(eta_line, use_container_width=True)

        with t2:
            timeout_line = px.line(
                trace_df,
                x="step",
                y="timeout_rate",
                color="policy",
                markers=True,
                title="累计超时率",
            )
            timeout_line.update_layout(height=330, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="时间步")
            st.plotly_chart(timeout_line, use_container_width=True)

            rider_line = px.line(
                trace_df,
                x="step",
                y="rider_load_std",
                color="policy",
                markers=True,
                title="骑手接单负载波动",
            )
            rider_line.update_layout(height=330, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="时间步")
            st.plotly_chart(rider_line, use_container_width=True)

        st.dataframe(trace_df, use_container_width=True, hide_index=True)

offline_path = Path("outputs/results/offline_metrics.csv")
sim_path = Path("outputs/results/simulation_metrics.csv")
figures_dir = Path("outputs/figures")

with tab_metrics:
    st.subheader("指标故事线")
    if offline_path.exists() and sim_path.exists():
        offline = pd.read_csv(offline_path)
        sim = pd.read_csv(sim_path)
        user_best = offline.sort_values("Recall@20", ascending=False).iloc[0]
        ours_full = offline[offline["model"] == "Ours-Full"].iloc[0]
        utility_best = sim.sort_values("platform_utility", ascending=False).iloc[0]
        eta_best = sim.sort_values("avg_eta").iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("最高 Recall@20", f"{user_best['Recall@20']:.4f}", str(user_best["model"]))
        k2.metric("Ours-Full Recall@20", f"{ours_full['Recall@20']:.4f}", "换取履约约束")
        k3.metric("最低 Avg ETA", f"{eta_best['avg_eta']:.2f}", str(eta_best["policy"]))
        k4.metric("最高平台效用", f"{utility_best['platform_utility']:.4f}", str(utility_best["policy"]))

        c1, c2 = st.columns(2)
        with c1:
            acc_fig = px.bar(
                offline,
                x="model",
                y=["Recall@20", "NDCG@20"],
                barmode="group",
                title="用户侧：推荐准确性",
                color_discrete_sequence=["#2563eb", "#0f766e"],
            )
            acc_fig.update_layout(height=360, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="")
            st.plotly_chart(acc_fig, use_container_width=True)

        with c2:
            fair_fig = px.scatter(
                offline,
                x="Recall@20",
                y="Coverage@20",
                color="model",
                size="LongTailExposure@20",
                title="用户准确性与商家覆盖的权衡",
            )
            fair_fig.update_layout(height=360, margin=dict(l=8, r=8, t=48, b=8))
            st.plotly_chart(fair_fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            sim_fig = px.scatter(
                sim,
                x="avg_eta",
                y="platform_utility",
                color="policy",
                size="completed_orders",
                title="履约侧：ETA 与平台效用",
            )
            sim_fig.update_layout(height=360, margin=dict(l=8, r=8, t=48, b=8))
            st.plotly_chart(sim_fig, use_container_width=True)

        with c4:
            timeout_fig = px.bar(
                sim.sort_values("timeout_rate"),
                x="policy",
                y="timeout_rate",
                color="platform_utility",
                title="超时率越低，平台效用越容易提升",
                color_continuous_scale=["#dc6803", "#0f766e"],
            )
            timeout_fig.update_layout(height=360, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="")
            st.plotly_chart(timeout_fig, use_container_width=True)

        with st.expander("查看完整指标表", expanded=False):
            st.dataframe(offline, use_container_width=True, hide_index=True)
            st.dataframe(sim, use_container_width=True, hide_index=True)
    else:
        st.warning("尚未找到指标表，请先运行 `make eval simulate`。")

with tab_figures:
    st.subheader("图表材料")
    image_kwargs = streamlit_image_width_kwargs(st.image)
    if not figures_dir.exists():
        st.warning("尚未生成图表，请先运行 `make figures`。")
    else:
        figure_files = sorted(figures_dir.glob("*.png"))
        if not figure_files:
            st.warning("尚未生成图表，请先运行 `make figures`。")
        for i in range(0, len(figure_files), 2):
            cols = st.columns(2)
            for col, fig in zip(cols, figure_files[i : i + 2]):
                col.image(str(fig), caption=fig.name, **image_kwargs)
