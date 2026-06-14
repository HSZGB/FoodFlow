from __future__ import annotations

import html
import os
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
from foodflow.recommenders import (
    OursBalancedRecommender,
    OursFullRecommender,
    SeqTripartiteRecommender,
    SeqXQuadRecommender,
    SeqXQuadTripartiteRecommender,
    SequentialHybridRecommender,
    UserOnlyRecommender,
)
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
def build_interactive_data(
    selected_user: str,
    max_orders: int,
    case_users: tuple[str, ...],
    _data: PreparedData,
) -> PreparedData:
    train = _data.orders_train
    if max_orders <= 0 or len(train) <= max_orders:
        return _data

    keep_users = {str(selected_user), *[str(user_id) for user_id in case_users]}
    keep_mask = train["user_id"].astype(str).isin(keep_users)
    kept = train[keep_mask]
    remaining = train[~keep_mask]
    remaining_n = max(max_orders - len(kept), 0)
    if remaining_n and len(remaining):
        sampled = remaining.sample(n=min(remaining_n, len(remaining)), random_state=42)
        train_sample = pd.concat([kept, sampled], ignore_index=True)
    else:
        train_sample = kept.copy()
    return PreparedData(
        users=_data.users,
        merchants=_data.merchants,
        orders_train=train_sample,
        orders_test=_data.orders_test,
        test_interactions=_data.test_interactions,
        spus=_data.spus,
    )


@st.cache_resource(show_spinner=False)
def load_model(strategy: str, data_key: str, _data: PreparedData) -> object:
    factories = {
        "UserOnly": UserOnlyRecommender,
        "Seq-Hybrid": SequentialHybridRecommender,
        "Seq-xQuAD": SeqXQuadRecommender,
        "Seq-Tripartite": SeqTripartiteRecommender,
        "Seq-xQuAD-Tripartite": SeqXQuadTripartiteRecommender,
        "Ours-Balanced": OursBalancedRecommender,
        "Ours-Full": OursFullRecommender,
    }
    return factories[strategy]().fit(_data)


def pct(value: float) -> str:
    return f"{clamp01(value) * 100:.0f}%"


DEMO_COLORS = {
    "Seq-xQuAD-Tripartite": "#B5121B",
    "Seq-xQuAD": "#2563EB",
    "Seq-Hybrid": "#60A5FA",
    "Seq-Tripartite": "#0F766E",
    "Ours-Full": "#DC6803",
    "Ours-Balanced": "#7C3AED",
}


def demo_color(name: object) -> str:
    text = str(name)
    for key, color in DEMO_COLORS.items():
        if key in text:
            return color
    return "#94a3b8"


def demo_color_map(values) -> dict[str, str]:
    return {str(value): demo_color(value) for value in values}


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
user_ids = data.user_ids
user_set = set(user_ids)
default_user = "8" if "8" in user_set else user_ids[0]
case_options = demo_user_cases(data.users)
demo_max_orders = int(os.environ.get("FOODFLOW_DEMO_MAX_ORDERS", "30000"))

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
    strategy_name = st.selectbox(
        "推荐策略",
        [
            "Seq-xQuAD",
            "Seq-xQuAD-Tripartite",
            "Seq-Hybrid",
            "Seq-Tripartite",
            "Ours-Full",
            "Ours-Balanced",
            "UserOnly",
        ],
        index=0,
    )
    top_k = st.slider("推荐数量", min_value=5, max_value=12, value=10, step=1)

interactive_data = build_interactive_data(user_id, demo_max_orders, tuple(case_options.values()), data)
model_data_key = f"orders={len(interactive_data.orders_train)};user={user_id};max={demo_max_orders}"
if len(interactive_data.orders_train) < len(data.orders_train):
    st.caption(
        f"交互推荐模型使用 {len(interactive_data.orders_train):,} 条训练订单；完整实验指标仍来自全量 TRD 输出。"
    )

with st.spinner(f"正在加载 {strategy_name} 推荐模型，首次运行会建立缓存..."):
    model = load_model(strategy_name, model_data_key, interactive_data)
rec_result = model.recommend([user_id], top_k, {user_id: period})
recs = rec_result.recommendations[user_id]
rec_df = build_recommendation_frame(interactive_data, model, user_id, recs, period)

users_df = data.users.set_index("user_id", drop=False)
merchants = data.merchants.set_index("wm_poi_id", drop=False)
user_row = users_df.loc[user_id]
riders = generate_riders(data.merchants, n_riders=240, seed=7)

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
    run_strategy_compare = st.checkbox("计算三策略对比", value=False)
    if run_strategy_compare:
        strategy_rows = []
        for name in [
            "UserOnly",
            "Seq-Hybrid",
            "Seq-xQuAD",
            "Seq-Tripartite",
            "Seq-xQuAD-Tripartite",
            "Ours-Balanced",
            "Ours-Full",
        ]:
            with st.spinner(f"加载 {name} 并生成对比..."):
                candidate_model = load_model(name, model_data_key, interactive_data)
                candidate_recs = candidate_model.recommend([user_id], top_k, {user_id: period}).recommendations[user_id]
                candidate_frame = build_recommendation_frame(interactive_data, candidate_model, user_id, candidate_recs, period)
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
    else:
        st.caption("为保证首屏打开速度，三策略对比默认不预计算；需要展示时勾选即可。")

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
            go.Histogram2dContour(
                x=pd.to_numeric(riders["lng"], errors="coerce"),
                y=pd.to_numeric(riders["lat"], errors="coerce"),
                colorscale=[
                    [0.0, "rgba(15, 118, 110, 0.00)"],
                    [0.35, "rgba(45, 212, 191, 0.16)"],
                    [1.0, "rgba(15, 118, 110, 0.38)"],
                ],
                contours=dict(coloring="heatmap", showlines=False),
                showscale=False,
                name="骑手密度",
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(riders["lng"], errors="coerce"),
                y=pd.to_numeric(riders["lat"], errors="coerce"),
                mode="markers",
                name="骑手供给",
                text=riders["rider_id"],
                marker=dict(
                    size=6,
                    color=pd.to_numeric(riders["load"], errors="coerce"),
                    colorscale=[[0, "#cbd5e1"], [0.5, "#94a3b8"], [1, "#475569"]],
                    opacity=0.36,
                    line=dict(width=0),
                ),
                hovertemplate="骑手 %{text}<br>经度 %{x:.4f}<br>纬度 %{y:.4f}<extra></extra>",
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=[float(rider_point["lng"]), float(merchant_row.get("lng", 116.40)), float(user_row.get("lng", 116.40))],
                y=[float(rider_point["lat"]), float(merchant_row.get("lat", 39.92)), float(user_row.get("lat", 39.92))],
                mode="lines",
                name="履约路径",
                line=dict(color="#0f766e", width=3, dash="dot"),
                hoverinfo="skip",
            )
        )
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
            title="空间供需：骑手密度、推荐商家与履约路径",
            height=340,
            xaxis_title="经度",
            yaxis_title="纬度",
            margin=dict(l=8, r=8, t=52, b=8),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        map_fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
        map_fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, scaleanchor="x", scaleratio=1)
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
    run_peak_replay = st.checkbox("运行高峰回放", value=False)
    if run_peak_replay:
        with st.spinner("正在运行轻量午餐高峰回放..."):
            trace_df = build_peak_trace(
                data,
                {
                    "UserOnly + MinETA": (load_model("UserOnly", model_data_key, interactive_data), "min_eta"),
                    "Seq-Hybrid + MinETA": (load_model("Seq-Hybrid", model_data_key, interactive_data), "min_eta"),
                    "Seq-xQuAD + MinETA": (load_model("Seq-xQuAD", model_data_key, interactive_data), "min_eta"),
                    "Seq-Tripartite": (load_model("Seq-Tripartite", model_data_key, interactive_data), "load_aware"),
                    "Seq-xQuAD-Tripartite": (
                        load_model("Seq-xQuAD-Tripartite", model_data_key, interactive_data),
                        "load_aware",
                    ),
                    "Ours-Balanced": (load_model("Ours-Balanced", model_data_key, interactive_data), "load_aware"),
                    "Ours-Full": (load_model("Ours-Full", model_data_key, interactive_data), "load_aware"),
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
                order_fig = px.bar(
                    final_trace.sort_values("completed_orders", ascending=False),
                    x="policy",
                    y="completed_orders",
                    color="policy",
                    title="最终完成订单对比",
                    color_discrete_sequence=["#2563eb", "#0f766e", "#dc6803", "#7c3aed"],
                )
                order_fig.update_layout(
                    height=330,
                    margin=dict(l=8, r=8, t=48, b=8),
                    xaxis_title="",
                    showlegend=False,
                    plot_bgcolor="#f8fafc",
                    paper_bgcolor="#ffffff",
                )
                order_fig.update_xaxes(tickangle=20, showgrid=False)
                order_fig.update_yaxes(gridcolor="#e2e8f0")
                st.plotly_chart(order_fig, use_container_width=True)

                eta_heat = trace_df.pivot(index="policy", columns="step", values="avg_eta")
                eta_fig = px.imshow(
                    eta_heat,
                    text_auto=".1f",
                    aspect="auto",
                    color_continuous_scale=["#ecfeff", "#0f766e", "#164e63"],
                    title="平均 ETA 热力图",
                )
                eta_fig.update_layout(
                    height=330,
                    margin=dict(l=8, r=8, t=48, b=8),
                    xaxis_title="时间步",
                    yaxis_title="",
                    coloraxis_colorbar=dict(title="min"),
                )
                st.plotly_chart(eta_fig, use_container_width=True)

            with t2:
                timeout_fig = px.bar(
                    final_trace.sort_values("timeout_rate"),
                    x="policy",
                    y="timeout_rate",
                    color="policy",
                    title="最终超时率对比",
                    color_discrete_sequence=["#0f766e", "#2563eb", "#dc6803", "#7c3aed"],
                )
                timeout_fig.update_layout(
                    height=330,
                    margin=dict(l=8, r=8, t=48, b=8),
                    xaxis_title="",
                    showlegend=False,
                    plot_bgcolor="#f8fafc",
                    paper_bgcolor="#ffffff",
                )
                timeout_fig.update_xaxes(tickangle=20, showgrid=False)
                timeout_fig.update_yaxes(gridcolor="#e2e8f0", tickformat=".0%")
                st.plotly_chart(timeout_fig, use_container_width=True)

                load_heat = trace_df.pivot(index="policy", columns="step", values="rider_load_std")
                load_fig = px.imshow(
                    load_heat,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale=["#f8fafc", "#f59e0b", "#7c2d12"],
                    title="骑手负载波动热力图",
                )
                load_fig.update_layout(
                    height=330,
                    margin=dict(l=8, r=8, t=48, b=8),
                    xaxis_title="时间步",
                    yaxis_title="",
                    coloraxis_colorbar=dict(title="std"),
                )
                st.plotly_chart(load_fig, use_container_width=True)

            st.dataframe(trace_df, use_container_width=True, hide_index=True)
    else:
        st.caption("高峰回放涉及多策略推荐和骑手匹配，默认不随首屏预运行；需要展示动态过程时勾选即可。")

offline_path = Path("outputs/results/offline_metrics.csv")
sim_path = Path("outputs/results/simulation_metrics.csv")
figures_dir = Path("outputs/figures")

with tab_metrics:
    st.subheader("指标故事线")
    if offline_path.exists() and sim_path.exists():
        offline = pd.read_csv(offline_path)
        sim = pd.read_csv(sim_path)
        policy_model_map = {
            "Popular + Nearest": "Popular",
            "UserOnly + Nearest": "UserOnly",
            "UserOnly + MinETA": "UserOnly",
            "Seq-Hybrid + MinETA": "Seq-Hybrid",
            "Seq-Hybrid + LoadAware": "Seq-Hybrid",
            "Seq-xQuAD + MinETA": "Seq-xQuAD",
            "Seq-Tripartite": "Seq-Tripartite",
            "Seq-xQuAD-Tripartite": "Seq-xQuAD-Tripartite",
            "Ours-Balanced": "Ours-Balanced",
            "Ours w/o Fairness": "UserOnly",
            "Ours-Full": "Ours-Full",
        }
        user_best = offline.sort_values("Recall@20", ascending=False).iloc[0]
        utility_best = sim.sort_values("platform_utility", ascending=False).iloc[0]
        eta_best = sim.sort_values("avg_eta").iloc[0]
        utility_model_name = policy_model_map.get(str(utility_best["policy"]), str(utility_best["policy"]))
        utility_model_match = offline[offline["model"].astype(str) == utility_model_name]
        utility_model = utility_model_match.iloc[0] if not utility_model_match.empty else user_best

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("最高 Recall@20", f"{user_best['Recall@20']:.4f}", str(user_best["model"]))
        k2.metric("最高效用模型 Recall@20", f"{utility_model['Recall@20']:.4f}", utility_model_name)
        k3.metric("最低 Avg ETA", f"{eta_best['avg_eta']:.2f}", str(eta_best["policy"]))
        k4.metric("最高平台效用", f"{utility_best['platform_utility']:.4f}", str(utility_best["policy"]))

        offline_index = offline.set_index("model", drop=False)
        frontier_rows = []
        for _, sim_row in sim.iterrows():
            policy = str(sim_row["policy"])
            model_name = policy_model_map.get(policy)
            if model_name not in offline_index.index:
                continue
            off_row = offline_index.loc[model_name]
            frontier_rows.append(
                {
                    "策略": policy,
                    "推荐模型": model_name,
                    "Recall@20": float(off_row["Recall@20"]),
                    "NDCG@20": float(off_row["NDCG@20"]),
                    "曝光Gini": float(off_row["ExposureGini"]),
                    "Avg ETA": float(sim_row["avg_eta"]),
                    "超时率": float(sim_row["timeout_rate"]),
                    "平台效用": float(sim_row["platform_utility"]),
                }
            )
        frontier_df = pd.DataFrame(frontier_rows).sort_values("平台效用", ascending=False)
        st.markdown("#### 三方权衡摘要")
        st.dataframe(
            frontier_df.head(7),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Recall@20": st.column_config.NumberColumn(format="%.4f"),
                "NDCG@20": st.column_config.NumberColumn(format="%.4f"),
                "曝光Gini": st.column_config.NumberColumn(format="%.4f"),
                "Avg ETA": st.column_config.NumberColumn(format="%.2f"),
                "超时率": st.column_config.NumberColumn(format="%.4f"),
                "平台效用": st.column_config.NumberColumn(format="%.4f"),
            },
        )

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
            acc_fig.update_xaxes(tickangle=25)
            st.plotly_chart(acc_fig, use_container_width=True)

        with c2:
            fair_fig = px.scatter(
                offline,
                x="Recall@20",
                y="Coverage@20",
                color="model",
                size="LongTailExposure@20",
                title="用户准确性与商家覆盖的权衡",
                color_discrete_map=demo_color_map(offline["model"]),
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
                color_discrete_map=demo_color_map(sim["policy"]),
            )
            sim_fig.update_layout(height=360, margin=dict(l=8, r=8, t=48, b=8))
            st.plotly_chart(sim_fig, use_container_width=True)

        with c4:
            timeout_sorted = sim.sort_values("timeout_rate")
            timeout_fig = px.bar(
                timeout_sorted,
                x="policy",
                y="timeout_rate",
                color="policy",
                text="timeout_rate",
                title="超时率越低，平台效用越容易提升",
                color_discrete_map=demo_color_map(timeout_sorted["policy"]),
            )
            timeout_fig.update_layout(height=360, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="")
            timeout_fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
            timeout_fig.update_xaxes(tickangle=25)
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
