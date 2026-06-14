from __future__ import annotations

import math
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
    build_rider_candidate_frame,
    build_rider_policy_frame,
    clamp01,
    demo_user_cases,
    streamlit_image_width_kwargs,
    user_category_profile,
)
from foodflow.frontier import POLICY_MODEL_MAP, build_tripartite_frontier
from foodflow.recommenders import (
    PopularRecommender,
    SeqTunedRecommender,
    SeqXQuadTripartiteRecommender,
    UserOnlyRecommender,
)
from foodflow.rider_sim import generate_riders


st.set_page_config(page_title="FoodFlow", layout="wide")

st.title("FoodFlow 外卖推荐与配送仿真")
st.caption("把一次下单拆开看：用户看到哪些商家，商家拿到多少曝光，订单最后派给哪位骑手。")

DEMO_STRATEGIES = ["UserOnly", "Seq-Tuned", "Seq-xQuAD-Tripartite"]
PEAK_POLICIES = ["Popular + Nearest", "UserOnly + MinETA", "Seq-Tuned + MinETA", "Seq-xQuAD-Tripartite"]

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
        "Popular": PopularRecommender,
        "UserOnly": UserOnlyRecommender,
        "Seq-Tuned": SeqTunedRecommender,
        "Seq-xQuAD-Tripartite": SeqXQuadTripartiteRecommender,
    }
    return factories[strategy]().fit(_data)


@st.cache_data(show_spinner=False)
def load_peak_trace(
    data_key: str,
    top_k: int,
    steps: int,
    requests_per_step: int,
    _data: PreparedData,
) -> pd.DataFrame:
    all_policies = {
        "Popular + Nearest": (load_model("Popular", data_key, _data), "nearest"),
        "UserOnly + MinETA": (load_model("UserOnly", data_key, _data), "min_eta"),
        "Seq-Tuned + MinETA": (load_model("Seq-Tuned", data_key, _data), "min_eta"),
        "Seq-xQuAD-Tripartite": (load_model("Seq-xQuAD-Tripartite", data_key, _data), "load_aware"),
    }
    policies = {name: all_policies[name] for name in PEAK_POLICIES}
    return build_peak_trace(
        _data,
        policies,
        seed=33,
        steps=steps,
        requests_per_step=requests_per_step,
        top_k=top_k,
    )


DEMO_COLORS = {
    "Popular": "#64748B",
    "Seq-Tuned": "#B45309",
    "Seq-xQuAD-Tripartite": "#B5121B",
    "UserOnly": "#2563EB",
}


def demo_color(name: object) -> str:
    text = str(name)
    for key, color in DEMO_COLORS.items():
        if key in text:
            return color
    return "#94a3b8"


def demo_color_map(values) -> dict[str, str]:
    return {str(value): demo_color(value) for value in values}


def geo_circle(lng: float, lat: float, radius_km: float, points: int = 96) -> tuple[list[float], list[float]]:
    lat_delta = radius_km / 111.0
    lng_delta = radius_km / max(111.0 * math.cos(math.radians(lat)), 1e-6)
    angles = [2.0 * math.pi * i / points for i in range(points + 1)]
    return (
        [lng + lng_delta * math.cos(angle) for angle in angles],
        [lat + lat_delta * math.sin(angle) for angle in angles],
    )


def viewport_range(values: list[float], padding_ratio: float = 0.18) -> list[float]:
    clean_values = [float(value) for value in values if pd.notna(value)]
    if not clean_values:
        return [0.0, 1.0]
    low = min(clean_values)
    high = max(clean_values)
    span = max(high - low, 0.01)
    padding = span * padding_ratio
    return [low - padding, high + padding]


def bounded_env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(min_value, min(value, max_value))


def score_bar(label: str, value: float) -> None:
    value = clamp01(value)
    st.caption(f"{label} {value:.0%}")
    st.progress(value)


def render_method_card(source: str, title: str, body: str, metric: str) -> None:
    with st.container(border=True):
        st.caption(source)
        st.write(f"**{title}**")
        st.write(body)
        st.caption(metric)


def render_peak_policy_card(row: pd.Series, rank: int) -> None:
    on_time = 1.0 - float(row["timeout_rate"])
    with st.container(border=True):
        st.caption(f"第 {rank} 名")
        st.write(f"**{row['policy']}**")
        a, b = st.columns(2)
        a.metric("累计订单", int(row["completed_orders"]))
        b.metric("准时率", f"{on_time:.1%}")
        c, d = st.columns(2)
        c.metric("Avg ETA", f"{float(row['avg_eta']):.1f} min")
        d.metric("活跃骑手", int(row["active_riders"]))


def render_recommendation_card(row: pd.Series, selected: bool = False) -> None:
    with st.container(border=True):
        st.caption("当前下单商家" if selected else f"TOP {int(row['rank'])}")
        st.write(f"**{row['merchant_name']}**")
        st.caption(
            f"ID {row['merchant_id']} · 品类 {row['category']} · "
            f"评分 {float(row['poi_score']):.2f} · 均价 {float(row['avg_price']):.1f} · "
            f"距离 {float(row['distance_km']):.2f} km"
        )
        st.write(str(row["reason"]).replace(" / ", "、"))
        m1, m2 = st.columns(2)
        m1.metric("总分", f"{float(row['final_score']):.3f}")
        m2.metric("ETA", f"{float(row['eta_minutes']):.1f} min")
        score_bar("用户偏好", float(row["user_score"]))
        score_bar("商家公平", float(row["fairness"]))
        score_bar("履约速度", float(row["eta_score"]))
        score_bar("供给稳定", float(row["supply"]))


data = load_data()
user_ids = data.user_ids
user_set = set(user_ids)
default_user = "8" if "8" in user_set else user_ids[0]
case_options = demo_user_cases(data.users)
demo_max_orders = bounded_env_int("FOODFLOW_DEMO_MAX_ORDERS", 12000, 0, 2_000_000)
demo_rider_count = bounded_env_int("FOODFLOW_DEMO_RIDERS", 1200, 120, 2400)

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
        DEMO_STRATEGIES,
        index=DEMO_STRATEGIES.index("Seq-xQuAD-Tripartite"),
    )
    top_k = st.slider("推荐数量", min_value=8, max_value=20, value=12, step=1)

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
riders = generate_riders(data.merchants, n_riders=demo_rider_count, seed=7)

if not rec_df.empty:
    top_row = rec_df.iloc[0]
    avg_eta = float(rec_df["eta_minutes"].mean())
    avg_fairness = float(rec_df["fairness"].mean())
    summary_cols = st.columns(5)
    with summary_cols[0]:
        st.metric("当前用户", str(user_id))
        st.caption(f"历史 {int(user_row.get('history_orders', 0))} 单")
    with summary_cols[1]:
        st.metric("推荐策略", strategy_name)
        st.caption("用户偏好、曝光与履约")
    with summary_cols[2]:
        st.metric("首位商家", str(top_row["merchant_name"]))
        st.caption(f"TOP1 分数 {float(top_row['final_score']):.3f}")
    with summary_cols[3]:
        st.metric("预计送达", f"{avg_eta:.1f} min")
        st.caption(f"商家公平均值 {avg_fairness:.2f}")
    with summary_cols[4]:
        st.metric("在线骑手", f"{len(riders)}")
        st.caption("合成在线骑手")

offline_path = Path("outputs/results/offline_metrics.csv")
sim_path = Path("outputs/results/simulation_metrics.csv")
figures_dir = Path("outputs/figures")

tab_case, tab_peak, tab_method, tab_metrics, tab_figures = st.tabs(
    ["推荐工作台", "高峰回放", "方法与指标", "实验结果", "图表材料"]
)

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
        st.info("这个筛选没命中，先把原来的推荐列表放回来。")
        card_df = rec_df.copy()

    card_count = min(9, len(card_df))
    for start in range(0, card_count, 3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, card_df.iloc[start : start + 3].iterrows()):
            with col:
                render_recommendation_card(row)

    st.subheader("同一用户的主线策略对比")
    run_strategy_compare = st.checkbox("计算主线策略对比", value=False)
    if run_strategy_compare:
        strategy_rows = []
        for name in DEMO_STRATEGIES:
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
                title="主线策略分项均值",
                color_discrete_sequence=["#2563eb", "#0f766e", "#7c3aed"],
            )
            strategy_chart.update_layout(height=300, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="")
            st.plotly_chart(strategy_chart, use_container_width=True)
    else:
        st.caption("开启后会临时加载三条主线策略，首次计算需要等待。")

    option_labels = [
        f"TOP {int(row.rank)} · {row.merchant_name} · ID {row.merchant_id}" for row in rec_df.itertuples(index=False)
    ]
    chosen_label = st.selectbox("模拟下单商家", option_labels)
    chosen_idx = option_labels.index(chosen_label)
    chosen_row = rec_df.iloc[chosen_idx]
    chosen_id = str(chosen_row["merchant_id"])
    merchant_row = merchants.loc[chosen_id]

    st.subheader("这单派给谁")
    left, right = st.columns([1.05, 1.35])
    with left:
        render_recommendation_card(chosen_row, selected=True)
        rider_compare = build_rider_policy_frame(user_row, merchant_row, riders, period)
        rider_candidates = build_rider_candidate_frame(user_row, merchant_row, riders, period, top_n=16)
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
            title=f"{strategy_name} 为什么把这家店排上来",
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

    candidate_left, candidate_right = st.columns([1.1, 1])
    if not rider_candidates.empty:
        candidate_view = rider_candidates.copy()
        candidate_view["status"] = [
            "已派单" if str(rider_id) == str(load_aware["rider_id"]) else "候选"
            for rider_id in candidate_view["rider_id"]
        ]
        with candidate_left:
            st.subheader("这单会优先派给谁")
            rider_table = candidate_view[
                ["rank", "status", "rider_id", "score", "eta", "pickup_distance_km", "load", "reliability", "reason"]
            ].rename(
                columns={
                    "rank": "排名",
                    "status": "状态",
                    "rider_id": "骑手",
                    "score": "派单分",
                    "eta": "ETA",
                    "pickup_distance_km": "取餐距离km",
                    "load": "当前负载",
                    "reliability": "可靠性",
                    "reason": "排序依据",
                }
            )
            st.dataframe(
                rider_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "派单分": st.column_config.NumberColumn(format="%.3f"),
                    "ETA": st.column_config.NumberColumn(format="%.1f"),
                    "取餐距离km": st.column_config.NumberColumn(format="%.2f"),
                    "可靠性": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        with candidate_right:
            candidate_plot = candidate_view.sort_values("score", ascending=True)
            score_fig = px.bar(
                candidate_plot,
                x="score",
                y="rider_id",
                orientation="h",
                color="status",
                title="骑手派单排序分",
                color_discrete_map={"已派单": "#7c3aed", "候选": "#94a3b8"},
                custom_data=["eta", "load", "reliability", "pickup_distance_km"],
            )
            score_fig.update_traces(
                hovertemplate=(
                    "骑手 %{y} | 派单分 %{x:.3f} | ETA %{customdata[0]:.1f} min"
                    " | 负载 %{customdata[1]} | 可靠性 %{customdata[2]:.2f}"
                    " | 取餐距离 %{customdata[3]:.2f} km"
                )
            )
            score_fig.update_layout(
                height=315,
                margin=dict(l=8, r=8, t=48, b=8),
                xaxis_title="负载感知得分",
                yaxis_title="",
                plot_bgcolor="#f8fafc",
                paper_bgcolor="#ffffff",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            score_fig.update_xaxes(gridcolor="#e2e8f0", range=[0, max(float(candidate_plot["score"].max()) * 1.08, 0.1)])
            score_fig.update_yaxes(showgrid=False)
            st.plotly_chart(score_fig, use_container_width=True)

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
        flow_fig.update_layout(title="这单的路径", height=340, margin=dict(l=8, r=8, t=52, b=8))
        st.plotly_chart(flow_fig, use_container_width=True)

    with map_col:
        rider_point = rider_compare[rider_compare["policy_key"] == "load_aware"].iloc[0]
        user_lng = float(user_row.get("lng", 116.40))
        user_lat = float(user_row.get("lat", 39.92))
        merchant_lng = float(merchant_row.get("lng", 116.40))
        merchant_lat = float(merchant_row.get("lat", 39.92))
        rider_lng = float(rider_point["lng"])
        rider_lat = float(rider_point["lat"])
        rider_plot = riders.copy()
        rider_plot["lng"] = pd.to_numeric(rider_plot["lng"], errors="coerce").fillna(116.40)
        rider_plot["lat"] = pd.to_numeric(rider_plot["lat"], errors="coerce").fillna(39.92)
        rider_plot["pickup_distance"] = (
            (rider_plot["lng"] - merchant_lng).pow(2) + (rider_plot["lat"] - merchant_lat).pow(2)
        )
        nearby_count = min(600, len(rider_plot))
        nearby_riders = rider_plot.nsmallest(nearby_count, "pickup_distance")
        user_plot = data.users.copy()
        user_plot["lng"] = pd.to_numeric(user_plot["lng"], errors="coerce").fillna(116.40)
        user_plot["lat"] = pd.to_numeric(user_plot["lat"], errors="coerce").fillna(39.92)
        user_plot["distance_to_user"] = (user_plot["lng"] - user_lng).pow(2) + (user_plot["lat"] - user_lat).pow(2)
        nearby_users = user_plot.nsmallest(min(180, len(user_plot)), "distance_to_user")
        merchant_plot = data.merchants.copy()
        merchant_plot["lng"] = pd.to_numeric(merchant_plot["lng"], errors="coerce").fillna(116.40)
        merchant_plot["lat"] = pd.to_numeric(merchant_plot["lat"], errors="coerce").fillna(39.92)
        merchant_plot["distance_to_store"] = (
            (merchant_plot["lng"] - merchant_lng).pow(2) + (merchant_plot["lat"] - merchant_lat).pow(2)
        )
        nearby_merchants = merchant_plot.nsmallest(min(180, len(merchant_plot)), "distance_to_store")
        merchant_circle_x, merchant_circle_y = geo_circle(merchant_lng, merchant_lat, 2.5)
        user_circle_x, user_circle_y = geo_circle(user_lng, user_lat, 2.5)

        st.caption(
            f"小蓝点是附近用户样本，小绿点是附近商家样本，灰色点是 {nearby_count} 名近场骑手；"
            "紫色编号 R1/R2... 是派单候选榜，两个圆分别是商家和用户周边 2.5km 参考范围。"
        )
        st.caption("半透明圆只是距离参考，不是等高线、热力图或真实配送边界。")
        map_fig = go.Figure()
        map_fig.add_trace(
            go.Scattergl(
                x=nearby_users["lng"],
                y=nearby_users["lat"],
                mode="markers",
                name="附近用户样本",
                text=nearby_users["user_id"].astype(str),
                marker=dict(size=5, color="#93c5fd", opacity=0.28),
                hovertemplate="用户 %{text}",
            )
        )
        map_fig.add_trace(
            go.Scattergl(
                x=nearby_merchants["lng"],
                y=nearby_merchants["lat"],
                mode="markers",
                name="附近商家样本",
                text=nearby_merchants["wm_poi_id"].astype(str),
                marker=dict(size=5, color="#86efac", opacity=0.32),
                hovertemplate="商家 %{text}",
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=merchant_circle_x,
                y=merchant_circle_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(15, 118, 110, 0.07)",
                line=dict(color="rgba(15, 118, 110, 0.42)", width=1.5),
                name="商家 2.5km 圈",
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=user_circle_x,
                y=user_circle_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.06)",
                line=dict(color="rgba(37, 99, 235, 0.38)", width=1.5),
                name="用户 2.5km 圈",
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            go.Scattergl(
                x=nearby_riders["lng"],
                y=nearby_riders["lat"],
                mode="markers",
                name="近场骑手",
                text=nearby_riders["rider_id"],
                customdata=nearby_riders[["load", "reliability"]].to_numpy(),
                marker=dict(
                    size=6.2,
                    color=pd.to_numeric(nearby_riders["load"], errors="coerce"),
                    colorscale=[[0, "#d9e2ec"], [0.5, "#94a3b8"], [1, "#334155"]],
                    opacity=0.42,
                    line=dict(width=0.4, color="#ffffff"),
                ),
                hovertemplate="骑手 %{text} | 负载 %{customdata[0]} | 可靠性 %{customdata[1]:.2f}",
            )
        )
        if not rider_candidates.empty:
            map_fig.add_trace(
                go.Scatter(
                    x=rider_candidates["lng"],
                    y=rider_candidates["lat"],
                    mode="markers+text",
                    name="Top 候选骑手",
                    text=rider_candidates["rank"].astype(int).map(lambda value: f"R{value}"),
                    customdata=rider_candidates[["rider_id", "score", "eta", "load"]].to_numpy(),
                    textposition="top center",
                    marker=dict(
                        size=(12 - rider_candidates["rank"].clip(upper=8) * 0.45).clip(lower=8),
                        color="#7c3aed",
                        opacity=0.86,
                        symbol="circle-open-dot",
                        line=dict(width=1.4, color="#7c3aed"),
                    ),
                    hovertemplate=(
                        "骑手 %{customdata[0]} | 派单分 %{customdata[1]:.3f}"
                        " | ETA %{customdata[2]:.1f} min | 负载 %{customdata[3]}"
                    ),
                )
            )
        map_fig.add_trace(
            go.Scatter(
                x=[rider_lng, merchant_lng],
                y=[rider_lat, merchant_lat],
                mode="lines",
                name="取餐段",
                line=dict(color="#7c3aed", width=3, dash="dot"),
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=[merchant_lng, user_lng],
                y=[merchant_lat, user_lat],
                mode="lines",
                name="配送段",
                line=dict(color="#dc6803", width=3),
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=[user_lng],
                y=[user_lat],
                mode="markers+text",
                name="用户",
                text=[f"用户 {user_id}"],
                textposition="top center",
                marker=dict(size=16, color="#2563eb", symbol="circle", line=dict(color="#ffffff", width=1.5)),
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=rec_df["lng"],
                y=rec_df["lat"],
                mode="markers+text",
                name="推荐商家",
                text=rec_df["rank"].astype(int).map(lambda value: f"T{value}"),
                customdata=rec_df[["merchant_name", "eta_minutes", "final_score"]].to_numpy(),
                textposition="middle center",
                marker=dict(
                    size=rec_df["final_score"].rank(pct=True) * 18 + 10,
                    color="#0f766e",
                    opacity=0.74,
                    line=dict(color="#ffffff", width=1),
                ),
                hovertemplate="%{customdata[0]} | ETA %{customdata[1]:.1f} min | 总分 %{customdata[2]:.3f}",
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=[merchant_lng],
                y=[merchant_lat],
                mode="markers+text",
                name="下单商家",
                text=[str(chosen_row["merchant_name"])],
                textposition="bottom center",
                marker=dict(size=20, color="#dc6803", symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            )
        )
        map_fig.add_trace(
            go.Scatter(
                x=[rider_lng],
                y=[rider_lat],
                mode="markers+text",
                name="匹配骑手",
                text=[str(rider_point["rider_id"])],
                textposition="top center",
                marker=dict(size=18, color="#7c3aed", symbol="square", line=dict(color="#ffffff", width=1.5)),
            )
        )
        focus_lng = [user_lng, merchant_lng, rider_lng] + rec_df["lng"].dropna().astype(float).tolist()
        focus_lat = [user_lat, merchant_lat, rider_lat] + rec_df["lat"].dropna().astype(float).tolist()
        focus_lng += nearby_users["lng"].dropna().astype(float).tolist()
        focus_lat += nearby_users["lat"].dropna().astype(float).tolist()
        focus_lng += nearby_merchants["lng"].dropna().astype(float).tolist()
        focus_lat += nearby_merchants["lat"].dropna().astype(float).tolist()
        focus_lng += nearby_riders["lng"].dropna().astype(float).tolist()
        focus_lat += nearby_riders["lat"].dropna().astype(float).tolist()
        focus_lng += rider_candidates["lng"].dropna().astype(float).tolist()
        focus_lat += rider_candidates["lat"].dropna().astype(float).tolist()
        map_fig.update_layout(
            title="当前订单附近的用户、商家和骑手",
            height=430,
            margin=dict(l=8, r=8, t=52, b=8),
            plot_bgcolor="#f6f8fb",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
            dragmode="pan",
        )
        map_fig.update_xaxes(
            visible=False,
            showgrid=False,
            zeroline=False,
            range=viewport_range(focus_lng),
        )
        map_fig.update_yaxes(
            visible=False,
            showgrid=False,
            zeroline=False,
            range=viewport_range(focus_lat),
            scaleanchor="x",
            scaleratio=1,
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
    st.subheader("午餐高峰怎么变化")
    replay_c1, replay_c2 = st.columns(2)
    with replay_c1:
        replay_steps = st.slider("回放时间步", min_value=8, max_value=32, value=16, step=2)
    with replay_c2:
        replay_requests = st.slider("每步订单请求", min_value=4, max_value=16, value=8, step=2)
    run_peak_replay = st.checkbox("运行高峰回放", value=False)
    if run_peak_replay:
        with st.spinner("正在跑午餐高峰回放，第一次会稍慢，后面会走缓存..."):
            trace_df = load_peak_trace(
                model_data_key,
                top_k,
                replay_steps,
                replay_requests,
                interactive_data,
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

            ops_board = final_trace.assign(on_time_rate=1.0 - final_trace["timeout_rate"]).sort_values(
                ["on_time_rate", "avg_eta", "completed_orders"], ascending=[False, True, False]
            )
            card_cols = st.columns(min(4, len(ops_board)))
            for col, (rank, (_, row)) in zip(card_cols, enumerate(ops_board.head(4).iterrows(), start=1)):
                with col:
                    render_peak_policy_card(row, rank)

            t1, t2 = st.columns(2)
            with t1:
                order_fig = px.bar(
                    final_trace.sort_values("completed_orders", ascending=False),
                    x="policy",
                    y="completed_orders",
                    color="policy",
                    title="最后完成了多少单",
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

                eta_heat = trace_df.pivot(index="policy", columns="step", values="step_avg_eta")
                eta_fig = px.imshow(
                    eta_heat,
                    text_auto=".1f",
                    aspect="auto",
                    color_continuous_scale=["#ecfeff", "#0f766e", "#164e63"],
                    title="每个时间段的 ETA",
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
                    title="最后超时率",
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

                load_heat = trace_df.pivot(index="policy", columns="step", values="step_timeout_rate")
                load_fig = px.imshow(
                    load_heat,
                    text_auto=".0%",
                    aspect="auto",
                    color_continuous_scale=["#f8fafc", "#f59e0b", "#7c2d12"],
                    title="每个时间段的超时率",
                )
                load_fig.update_layout(
                    height=330,
                    margin=dict(l=8, r=8, t=48, b=8),
                    xaxis_title="时间步",
                    yaxis_title="",
                    coloraxis_colorbar=dict(title="rate"),
                )
                st.plotly_chart(load_fig, use_container_width=True)

            st.dataframe(trace_df, use_container_width=True, hide_index=True)
    else:
        st.caption("开启后会连续运行多轮推荐与派单仿真，首次计算需要等待。")

with tab_method:
    st.subheader("方法与指标")
    offline_method = pd.read_csv(offline_path) if offline_path.exists() else pd.DataFrame()
    sim_method = pd.read_csv(sim_path) if sim_path.exists() else pd.DataFrame()

    def model_metric_text(model_name: str, metric: str, label: str) -> str:
        if offline_method.empty or metric not in offline_method.columns:
            return label
        matched = offline_method[offline_method["model"].astype(str) == model_name]
        if matched.empty:
            return label
        return f"{label}: {float(matched.iloc[0][metric]):.4f}"

    def policy_metric_text(policy_name: str, metric: str, label: str) -> str:
        if sim_method.empty or metric not in sim_method.columns:
            return label
        matched = sim_method[sim_method["policy"].astype(str) == policy_name]
        if matched.empty:
            return label
        return f"{label}: {float(matched.iloc[0][metric]):.4f}"

    method_rows = [
        (
            "用户画像",
            "UserOnly：用户偏好排序",
            "用品类、复购、价格、时段和商家质量做排序，是后面三方策略的用户侧底座。",
            model_metric_text("UserOnly", "Recall@20", "Recall@20"),
        ),
        (
            "短序列推荐",
            "Seq-Tuned：把复购放在更前面",
            "外卖用户常常反复点熟悉的店，所以这里更看重最近订单、复购次数和商家转移。",
            model_metric_text("Seq-Tuned", "Recall@20", "Recall@20"),
        ),
        (
            "商家侧",
            "三方重排：不只照顾用户点击",
            "把曝光公平、ETA 和供给情况一起放进排序，避免平台只推少数头部店。",
            model_metric_text("Seq-xQuAD-Tripartite", "ExposureGini", "Exposure Gini"),
        ),
        (
            "骑手侧",
            "订单派给骑手：看时间，也看负载",
            "用户选店之后继续模拟派单，用超时率和综合分看这条链路稳不稳。",
            policy_metric_text("Seq-xQuAD-Tripartite", "platform_utility", "Platform Utility"),
        ),
    ]
    for start in range(0, len(method_rows), 3):
        cols = st.columns(min(3, len(method_rows) - start))
        for col, values in zip(cols, method_rows[start : start + 3]):
            with col:
                render_method_card(*values)

    st.dataframe(
        pd.DataFrame(
            [
                {"评价对象": "用户偏好", "项目做法": "用最近订单、复购次数和店铺转移来排商家", "对应模块": "Seq-Tuned"},
                {"评价对象": "商家曝光", "项目做法": "把曝光公平、ETA 和供给情况接到重排里", "对应模块": "Seq-xQuAD-Tripartite"},
                {"评价对象": "订单履约", "项目做法": "模拟骑手候选排序和负载感知派单", "对应模块": "骑手仿真"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not offline_method.empty and not sim_method.empty:
        frontier_method = build_tripartite_frontier(offline_method, sim_method, POLICY_MODEL_MAP)
        user_best = offline_method.sort_values("Recall@20", ascending=False).iloc[0]
        ndcg_best = offline_method.sort_values("NDCG@20", ascending=False).iloc[0]
        calibration_best = (
            offline_method.sort_values("CategoryJSD@20").iloc[0]
            if "CategoryJSD@20" in offline_method.columns
            else user_best
        )
        utility_best = sim_method.sort_values("platform_utility", ascending=False).iloc[0]
        eta_best = sim_method.sort_values("avg_eta").iloc[0]
        frontier_best = (
            frontier_method[frontier_method["is_frontier"]].sort_values("platform_utility", ascending=False).iloc[0]
            if not frontier_method.empty and frontier_method["is_frontier"].any()
            else None
        )

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("最高 Recall@20", f"{user_best['Recall@20']:.4f}", str(user_best["model"]))
        p2.metric("最高 NDCG@20", f"{ndcg_best['NDCG@20']:.4f}", str(ndcg_best["model"]))
        p3.metric("最高平台综合分", f"{utility_best['platform_utility']:.4f}", str(utility_best["policy"]))
        p4.metric("最低 Avg ETA", f"{eta_best['avg_eta']:.2f} min", str(eta_best["policy"]))

        playbook_rows = [
            {
                "评价重点": "推荐准确率",
                "建议策略": str(user_best["model"]),
                "证据": f"Recall@20={float(user_best['Recall@20']):.4f}",
                "指标含义": "反映用户侧命中能力，尚未覆盖配送表现",
            },
            {
                "评价重点": "品类校准",
                "建议策略": str(calibration_best["model"]),
                "证据": f"NDCG@20={float(ndcg_best['NDCG@20']):.4f}, JSD@20={float(calibration_best.get('CategoryJSD@20', 0.0)):.4f}",
                "指标含义": "衡量推荐列表是否贴近用户历史品类分布",
            },
            {
                "评价重点": "推荐到履约",
                "建议策略": str(utility_best["policy"]),
                "证据": f"Utility={float(utility_best['platform_utility']):.4f}, Timeout={float(utility_best['timeout_rate']):.4f}",
                "指标含义": "综合比较用户满意度、准时率、负载和商家曝光",
            },
            {
                "评价重点": "前沿方案",
                "建议策略": str(frontier_best["policy"]) if frontier_best is not None else str(utility_best["policy"]),
                "证据": (
                    f"Recall@20={float(frontier_best['Recall@20']):.4f}, Utility={float(frontier_best['platform_utility']):.4f}"
                    if frontier_best is not None
                    else f"Utility={float(utility_best['platform_utility']):.4f}"
                ),
                "指标含义": "展示准确性与系统效用之间的折中边界",
            },
        ]
        st.dataframe(pd.DataFrame(playbook_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("尚未找到指标表，请先运行 `make eval simulate`。")

with tab_metrics:
    st.subheader("实验结果")
    if offline_path.exists() and sim_path.exists():
        offline = pd.read_csv(offline_path)
        sim = pd.read_csv(sim_path)
        frontier = build_tripartite_frontier(offline, sim, POLICY_MODEL_MAP)
        user_best = offline.sort_values("Recall@20", ascending=False).iloc[0]
        utility_best = sim.sort_values("platform_utility", ascending=False).iloc[0]
        eta_best = sim.sort_values("avg_eta").iloc[0]
        calibration_best = (
            offline.sort_values("CategoryJSD@20").iloc[0] if "CategoryJSD@20" in offline.columns else user_best
        )
        utility_model_name = POLICY_MODEL_MAP.get(str(utility_best["policy"]), str(utility_best["policy"]))
        utility_model_match = offline[offline["model"].astype(str) == utility_model_name]
        utility_model = utility_model_match.iloc[0] if not utility_model_match.empty else user_best

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("最高 Recall@20", f"{user_best['Recall@20']:.4f}", str(user_best["model"]))
        k2.metric("配送最优的 Recall", f"{utility_model['Recall@20']:.4f}", utility_model_name)
        k3.metric("品类最贴近历史", f"{calibration_best.get('CategoryJSD@20', 0.0):.4f}", str(calibration_best["model"]))
        k4.metric("最低 Avg ETA", f"{eta_best['avg_eta']:.2f}", str(eta_best["policy"]))
        k5.metric("最高平台综合分", f"{utility_best['platform_utility']:.4f}", str(utility_best["policy"]))

        frontier_df = frontier.rename(
            columns={
                "policy": "策略",
                "model": "推荐模型",
                "ExposureGini": "曝光Gini",
                "avg_eta": "Avg ETA",
                "timeout_rate": "超时率",
                "platform_utility": "平台综合分",
                "is_frontier": "未被压过",
            }
        ).sort_values("平台综合分", ascending=False)
        st.subheader("三方策略对比")
        st.dataframe(
            frontier_df[["策略", "推荐模型", "未被压过", "Recall@20", "NDCG@20", "曝光Gini", "Avg ETA", "超时率", "平台综合分"]].head(7),
            use_container_width=True,
            hide_index=True,
            column_config={
                "未被压过": st.column_config.CheckboxColumn(),
                "Recall@20": st.column_config.NumberColumn(format="%.4f"),
                "NDCG@20": st.column_config.NumberColumn(format="%.4f"),
                "曝光Gini": st.column_config.NumberColumn(format="%.4f"),
                "Avg ETA": st.column_config.NumberColumn(format="%.2f"),
                "超时率": st.column_config.NumberColumn(format="%.4f"),
                "平台综合分": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        pareto_fig = px.scatter(
            frontier,
            x="Recall@20",
            y="platform_utility",
            color="policy",
            symbol="is_frontier",
            size="on_time_rate",
            hover_data=["model", "NDCG@20", "ExposureGini", "avg_eta", "timeout_rate"],
            title="推荐准确性与平台效用",
            color_discrete_map=demo_color_map(frontier["policy"]),
        )
        pareto_front = frontier[frontier["is_frontier"]].sort_values("Recall@20")
        if len(pareto_front) >= 2:
            pareto_fig.add_trace(
                go.Scatter(
                    x=pareto_front["Recall@20"],
                    y=pareto_front["platform_utility"],
                    mode="lines",
                    line=dict(color="#B5121B", width=2),
                    name="折中线",
                    hoverinfo="skip",
                )
            )
        pareto_fig.update_layout(height=360, margin=dict(l=8, r=8, t=48, b=8))
        st.plotly_chart(pareto_fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            acc_fig = px.bar(
                offline,
                x="model",
                y=["Recall@20", "NDCG@20"],
                barmode="group",
                title="用户侧：推荐有没有命中",
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
                title="配送侧：时间和综合分",
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
                title="超时率对比",
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
