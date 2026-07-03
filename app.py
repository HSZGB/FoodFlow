from __future__ import annotations

import math
import os
import json
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
    enroute_opportunities,
    merchant_supply_pressure,
    recommendation_card_batches,
    streamlit_image_width_kwargs,
    top_dishes_for_user,
    user_category_profile,
)
from foodflow.frontier import POLICY_MODEL_MAP, build_tripartite_frontier
from foodflow.recommenders import (
    KGTripartiteRecommender,
    PopularRecommender,
    SessionSpuTripartiteRecommender,
    SeqTunedRecommender,
    SeqXQuadTripartiteRecommender,
    UserOnlyRecommender,
    build_learned_ltr_recommender,
    learned_ltr_model_name,
)
from foodflow.rider_sim import generate_riders


st.set_page_config(page_title="FoodFlow", layout="wide")

st.title("FoodFlow 外卖推荐与配送仿真")
st.caption("把一次下单拆开看：用户看到哪些商家，商家拿到多少曝光，订单最后派给哪位骑手。")
st.info(
    "**一句话主线：推荐不只是排商家——它改变订单的空间分布，进而影响骑手负载、送达时效和商家曝光。**  "
    "演示路径：① 推荐工作台看单个用户的推荐与派单 → ② 高峰回放看策略在高峰期的整体差异 → ③ 实验结果看离线指标与三方权衡。  "
    "三方策略在保持推荐命中的同时降低超时率、拉平商家曝光；KG-Tripartite 额外叠加知识图谱兴趣路径，可给出图谱化推荐解释。"
)

LEARNED_LTR_MODEL = learned_ltr_model_name()
LEARNED_LTR_POLICY = f"{LEARNED_LTR_MODEL} + MinETA"
DEMO_STRATEGIES = [
    "UserOnly",
    "Seq-Tuned",
    LEARNED_LTR_MODEL,
    "Seq-xQuAD-Tripartite",
    "Session-SPU-Tripartite",
    "KG-Tripartite",
]
PEAK_POLICIES = [
    "Popular + Nearest",
    "UserOnly + MinETA",
    "Seq-Tuned + MinETA",
    LEARNED_LTR_POLICY,
    "Seq-xQuAD-Tripartite + Greedy",
    "Session-SPU-Tripartite + Greedy",
    "Session-SPU-Tripartite + Batch",
    "KG-Tripartite + Batch",
]

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
    order_ids = set(train_sample["wm_order_id"].astype(str)) if "wm_order_id" in train_sample.columns else set()

    def filter_optional_orders(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not order_ids or "wm_order_id" not in frame.columns:
            return frame
        return frame[frame["wm_order_id"].astype(str).isin(order_ids)].copy()

    return PreparedData(
        users=_data.users,
        merchants=_data.merchants,
        orders_train=train_sample,
        orders_test=_data.orders_test,
        test_interactions=_data.test_interactions,
        spus=_data.spus,
        session_interactions=filter_optional_orders(_data.session_interactions),
        order_spus_train=filter_optional_orders(_data.order_spus_train),
        order_spus_test=_data.order_spus_test,
    )


@st.cache_resource(show_spinner=False)
def load_model(strategy: str, data_key: str, _data: PreparedData) -> object:
    factories = {
        "Popular": PopularRecommender,
        "UserOnly": UserOnlyRecommender,
        "Seq-Tuned": SeqTunedRecommender,
        LEARNED_LTR_MODEL: lambda: build_learned_ltr_recommender(seed=42),
        "Seq-xQuAD-Tripartite": SeqXQuadTripartiteRecommender,
        "Session-SPU-Tripartite": SessionSpuTripartiteRecommender,
        "KG-Tripartite": KGTripartiteRecommender,
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
        LEARNED_LTR_POLICY: (load_model(LEARNED_LTR_MODEL, data_key, _data), "min_eta"),
        "Seq-xQuAD-Tripartite + Greedy": (
            load_model("Seq-xQuAD-Tripartite", data_key, _data),
            "load_aware",
        ),
        "Session-SPU-Tripartite + Greedy": (
            load_model("Session-SPU-Tripartite", data_key, _data),
            "load_aware",
        ),
        "Session-SPU-Tripartite + Batch": (
            load_model("Session-SPU-Tripartite", data_key, _data),
            "load_aware",
            "batch",
        ),
        "KG-Tripartite + Batch": (
            load_model("KG-Tripartite", data_key, _data),
            "load_aware",
            "batch",
        ),
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
    "LightGBM-LTR": "#7C3AED",
    "Logistic-LTR": "#7C3AED",
    "Seq-xQuAD-Tripartite + Greedy": "#F97316",
    "Seq-xQuAD-Tripartite": "#B5121B",
    "Session-SPU-Tripartite": "#0F766E",
    "KG-Tripartite": "#9D174D",
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


# --- 地图底图源 -------------------------------------------------------------
# 高德瓦片国内加载快但使用 GCJ-02 火星坐标系，绘制 WGS84 坐标前需纠偏对齐；
# OSM/Carto 为 WGS84 但国内网络可能加载缓慢或失败。
MAP_CFG = {"gcj02": False, "style": "carto-positron"}

_AMAP_STYLE = {
    "version": 8,
    "sources": {
        "amap": {
            "type": "raster",
            "tiles": [
                f"https://webrd0{i}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={{x}}&y={{y}}&z={{z}}"
                for i in (1, 2, 3, 4)
            ],
            "tileSize": 256,
            "attribution": "© 高德地图",
        }
    },
    "layers": [{"id": "amap", "type": "raster", "source": "amap"}],
}
_OSM_STYLE = {
    "version": 8,
    "sources": {
        "osm": {
            "type": "raster",
            "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            "tileSize": 256,
            "attribution": "© OpenStreetMap contributors",
        }
    },
    "layers": [{"id": "osm", "type": "raster", "source": "osm"}],
}
MAP_TILE_SOURCES = {
    "高德（国内推荐）": {"style": _AMAP_STYLE, "gcj02": True},
    "OpenStreetMap": {"style": _OSM_STYLE, "gcj02": False},
    "Carto 浅色": {"style": "carto-positron", "gcj02": False},
}


def _wgs84_to_gcj02_arrays(lng, lat):
    """WGS84 -> GCJ-02 火星坐标纠偏（标准偏移算法，向量化）。"""
    import numpy as np

    lng = pd.to_numeric(pd.Series(list(lng)), errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(pd.Series(list(lat)), errors="coerce").to_numpy(dtype=float)
    a, ee = 6378245.0, 0.00669342162296594323
    x, y = lng - 105.0, lat - 35.0

    def _dlat(x, y):
        d = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
        d += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
        d += (20.0 * np.sin(y * np.pi) + 40.0 * np.sin(y / 3.0 * np.pi)) * 2.0 / 3.0
        d += (160.0 * np.sin(y / 12.0 * np.pi) + 320.0 * np.sin(y * np.pi / 30.0)) * 2.0 / 3.0
        return d

    def _dlng(x, y):
        d = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
        d += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
        d += (20.0 * np.sin(x * np.pi) + 40.0 * np.sin(x / 3.0 * np.pi)) * 2.0 / 3.0
        d += (150.0 * np.sin(x / 12.0 * np.pi) + 300.0 * np.sin(x / 30.0 * np.pi)) * 2.0 / 3.0
        return d

    dlat, dlng = _dlat(x, y), _dlng(x, y)
    radlat = lat / 180.0 * np.pi
    magic = 1 - ee * np.sin(radlat) ** 2
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * np.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * np.cos(radlat) * np.pi)
    return lng + dlng, lat + dlat


def map_coords(lng, lat):
    """按当前底图源把 WGS84 坐标转换为绘制坐标（高德底图时做 GCJ-02 纠偏）。"""
    if MAP_CFG["gcj02"]:
        return _wgs84_to_gcj02_arrays(lng, lat)
    return lng, lat


def map_scatter(real_map: bool, lng, lat, **kwargs):
    """构造散点轨迹：真实底图用 Scattermap（经纬度语义），离线回退到抽象画布。"""
    use_gl = kwargs.pop("use_gl", False)
    if real_map:
        marker = kwargs.get("marker")
        if isinstance(marker, dict):
            kwargs["marker"] = {key: value for key, value in marker.items() if key not in {"symbol", "line"}}
        kwargs.pop("fill", None)
        kwargs.pop("fillcolor", None)
        plot_lng, plot_lat = map_coords(lng, lat)
        return go.Scattermap(lon=plot_lng, lat=plot_lat, **kwargs)
    trace_cls = go.Scattergl if use_gl else go.Scatter
    return trace_cls(x=lng, y=lat, **kwargs)


def apply_map_layout(fig, real_map: bool, focus_lng: list[float], focus_lat: list[float], title: str, height: int = 460) -> None:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=8, r=8, t=52, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10, color="#111827")),
    )
    clean_lng = [float(v) for v in focus_lng if pd.notna(v)]
    clean_lat = [float(v) for v in focus_lat if pd.notna(v)]
    if real_map:
        if clean_lng and clean_lat:
            plot_lng, plot_lat = map_coords(clean_lng, clean_lat)
            clean_lng = [float(v) for v in plot_lng]
            clean_lat = [float(v) for v in plot_lat]
        pad_lng = max((max(clean_lng) - min(clean_lng)) * 0.08, 0.002) if clean_lng else 0.01
        pad_lat = max((max(clean_lat) - min(clean_lat)) * 0.08, 0.002) if clean_lat else 0.01
        fig.update_layout(
            map=dict(
                style=MAP_CFG["style"],
                bounds=dict(
                    west=min(clean_lng) - pad_lng if clean_lng else 121.2,
                    east=max(clean_lng) + pad_lng if clean_lng else 121.5,
                    south=min(clean_lat) - pad_lat if clean_lat else 37.4,
                    north=max(clean_lat) + pad_lat if clean_lat else 37.6,
                ),
            )
        )
    else:
        fig.update_layout(plot_bgcolor="#f6f8fb", paper_bgcolor="#ffffff", dragmode="pan")
        fig.update_xaxes(visible=False, showgrid=False, zeroline=False, range=viewport_range(clean_lng))
        fig.update_yaxes(
            visible=False, showgrid=False, zeroline=False, range=viewport_range(clean_lat),
            scaleanchor="x", scaleratio=1,
        )


def algorithm_focus(model_name: str) -> dict[str, str]:
    if model_name == "Popular":
        return {
            "focus": "全局热度基线",
            "reason": "用商家历史订单量排序，主要用于证明个性化模型相对热门榜的增益。",
            "display": "关注 Recall 和 Coverage 是否明显低于个性化模型。",
        }
    if model_name == "BPR-MF":
        return {
            "focus": "隐式反馈矩阵分解",
            "reason": "用用户和商家的潜向量点积做成对排序，冷启动用户回退热门榜。",
            "display": "关注 Recall/NDCG 与覆盖率，作为传统协同过滤对照。",
        }
    if model_name == "UserOnly":
        return {
            "focus": "用户画像可解释排序",
            "reason": "品类、复购、价格、时段和质量共同组成用户侧偏好分。",
            "display": "重点展示复购、品类和价格匹配理由。",
        }
    if model_name == "Seq-Tuned":
        return {
            "focus": "序列和复购强化",
            "reason": "固定权重组合快慢最近性、复购、商家转移、品类、热度和质量。",
            "display": "重点展示 Recall、NDCG 以及复购/转移理由。",
        }
    if model_name in {"LightGBM-LTR", "Logistic-LTR"}:
        return {
            "focus": "学习排序",
            "reason": "复用七类序列特征，由 LambdaRank 或 Logistic fallback 学习非线性排序分。",
            "display": "重点展示学习排序分、Coverage 和序列特征诊断。",
        }
    if model_name == "Seq-xQuAD-Tripartite":
        return {
            "focus": "三方重排",
            "reason": "候选内归一化用户分、商家公平、ETA 分和供给分，再用 xQuAD 做品类覆盖和长尾重排。",
            "display": "重点展示商家履约 ETA、供给稳定和曝光公平理由。",
        }
    if model_name == "Session-SPU-Tripartite":
        return {
            "focus": "会话和菜品增强",
            "reason": "在三方重排基础上加入训练期点击会话与用户/商家的 SPU 菜品类目重合。",
            "display": "重点展示会话点击、SPU 菜品类目匹配和履约 ETA。",
        }
    if model_name == "KG-Tripartite":
        return {
            "focus": "知识图谱兴趣路径",
            "reason": "在会话/SPU 三方重排之上，叠加用户对品类、商圈、价位等图谱节点的时间衰减兴趣，与商家属性做关系加权匹配（完整动态 KG 注意力模型见 kg-demo 模块）。",
            "display": "重点展示图谱路径解释（品类/区域/价位）与推荐命中的关系。",
        }
    return {"focus": "推荐模型", "reason": "按当前离线指标展示。", "display": "关注 Recall、NDCG 和覆盖率。"}


def best_simulation_text(model_name: str, sim: pd.DataFrame) -> str:
    if sim.empty:
        return ""
    policies = [policy for policy, mapped in POLICY_MODEL_MAP.items() if mapped == model_name]
    matched = sim[sim["policy"].astype(str).isin(policies)]
    if matched.empty:
        return ""
    best = matched.sort_values(["platform_utility", "on_time_rate"], ascending=[False, False]).iloc[0]
    return (
        f"{best['policy']}: ETA {float(best['avg_eta']):.1f} min, "
        f"超时率 {float(best['timeout_rate']):.1%}, 综合分 {float(best['platform_utility']):.4f}"
    )


def build_algorithm_evaluation_frame(offline: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in offline.iterrows():
        model_name = str(row["model"])
        focus = algorithm_focus(model_name)
        rows.append(
            {
                "模型": model_name,
                "展示重点": focus["focus"],
                "Recall@20": float(row.get("Recall@20", 0.0)),
                "NDCG@20": float(row.get("NDCG@20", 0.0)),
                "Coverage@20": float(row.get("Coverage@20", 0.0)),
                "曝光Gini": float(row.get("ExposureGini", 0.0)),
                "长尾曝光": float(row.get("LongTailExposure@20", 0.0)),
                "算法解释": focus["reason"],
                "Demo理由": focus["display"],
                "履约仿真证据": best_simulation_text(model_name, sim),
            }
        )
    return pd.DataFrame(rows)


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


def parse_step_matches(row: pd.Series) -> list[dict[str, object]]:
    raw = row.get("step_matches_json", row.get("batch_matches_json", ""))
    if raw is None or pd.isna(raw) or not str(raw).strip():
        return []
    try:
        matches = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return matches if isinstance(matches, list) else []


def render_step_matching_view(row: pd.Series, height: int = 430, show_table: bool = True) -> None:
    matches = parse_step_matches(row)
    if not matches:
        st.caption("该时间步没有可展示的匹配。")
        return

    display_matches = matches[:16]
    plot_df = pd.DataFrame(display_matches).copy()
    plot_df["order_no"] = range(1, len(plot_df) + 1)
    for column in [
        "user_lng",
        "user_lat",
        "merchant_lng",
        "merchant_lat",
        "rider_lng",
        "rider_lat",
        "slot_number",
        "eta",
        "score",
        "rider_load",
    ]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce").fillna(0.0)
    slot_offsets = []
    for _, match in plot_df.iterrows():
        angle = (float(match["slot_number"]) + 1.0) * 2.2
        slot_offsets.append((math.cos(angle) * 0.00045, math.sin(angle) * 0.00045))
    plot_df["slot_lng"] = plot_df["rider_lng"] + [offset[0] for offset in slot_offsets]
    plot_df["slot_lat"] = plot_df["rider_lat"] + [offset[1] for offset in slot_offsets]
    plot_df["slot_label"] = plot_df.apply(
        lambda item: f"{item['rider_id']} 槽位 {int(item['slot_number']) + 1}",
        axis=1,
    )

    fig = go.Figure()
    for _, match in plot_df.iterrows():
        eta = float(match["eta"])
        fig.add_trace(
            go.Scatter(
                x=[float(match["slot_lng"]), float(match["merchant_lng"])],
                y=[float(match["slot_lat"]), float(match["merchant_lat"])],
                mode="lines",
                name="骑手到商家",
                line=dict(color="#7c3aed", width=1.8, dash="dot"),
                opacity=0.42,
                hovertemplate=(
                    f"{match['slot_label']} -> {match.get('merchant_name', '商家')}"
                    f" | O{int(match['order_no'])}"
                ),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[float(match["slot_lng"]), float(match["user_lng"])],
                y=[float(match["slot_lat"]), float(match["user_lat"])],
                mode="lines",
                name="匹配连线",
                line=dict(color="#dc6803" if eta > 45.0 else "#0f766e", width=2.6),
                opacity=0.58,
                hovertemplate=(
                    f"O{int(match['order_no'])} -> {match['slot_label']}"
                    f" | ETA {eta:.1f} min | 边权 {float(match['score']):.3f}"
                ),
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=plot_df["merchant_lng"],
            y=plot_df["merchant_lat"],
            mode="markers",
            name="下单商家",
            text=plot_df["merchant_name"].astype(str),
            customdata=plot_df[["order_no", "merchant_id"]].to_numpy(),
            marker=dict(size=8, color="#0f766e", opacity=0.55, symbol="diamond", line=dict(color="#ffffff", width=0.8)),
            hovertemplate="O%{customdata[0]} 商家 %{text} | ID %{customdata[1]}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["user_lng"],
            y=plot_df["user_lat"],
            mode="markers+text",
            name="用户订单",
            text=plot_df["order_no"].map(lambda value: f"O{int(value)}"),
            customdata=plot_df[["user_id", "merchant_name", "eta"]].to_numpy(),
            textposition="top center",
            marker=dict(size=13, color="#2563eb", symbol="circle", line=dict(color="#ffffff", width=1.2)),
            hovertemplate="用户 %{customdata[0]} | %{customdata[1]} | ETA %{customdata[2]:.1f} min",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["slot_lng"],
            y=plot_df["slot_lat"],
            mode="markers+text",
            name="骑手容量槽位",
            text=plot_df["order_no"].map(lambda value: f"S{int(value)}"),
            customdata=plot_df[["slot_label", "rider_load", "score"]].to_numpy(),
            textposition="bottom center",
            marker=dict(size=14, color="#7c3aed", symbol="square", line=dict(color="#ffffff", width=1.2)),
            hovertemplate="%{customdata[0]} | 原负载 %{customdata[1]} | 边权 %{customdata[2]:.3f}",
        )
    )
    focus_lng = (
        plot_df["user_lng"].dropna().astype(float).tolist()
        + plot_df["merchant_lng"].dropna().astype(float).tolist()
        + plot_df["slot_lng"].dropna().astype(float).tolist()
    )
    focus_lat = (
        plot_df["user_lat"].dropna().astype(float).tolist()
        + plot_df["merchant_lat"].dropna().astype(float).tolist()
        + plot_df["slot_lat"].dropna().astype(float).tolist()
    )
    fig.update_layout(
        title=dict(
            text=(
                f"{row['policy']} · 第 {int(row['step'])} 步 {row.get('assignment_mode', 'greedy')} 地图匹配 "
                f"({int(row.get('step_matched_count', len(matches)))} / {int(row.get('step_order_count', len(matches)))})"
            ),
            font=dict(color="#111827", size=15),
        ),
        font=dict(color="#111827"),
        height=height,
        margin=dict(l=6, r=6, t=54, b=6),
        plot_bgcolor="#f6f8fb",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=10, color="#111827")),
        dragmode="pan",
    )
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False, range=viewport_range(focus_lng, 0.28))
    fig.update_yaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        range=viewport_range(focus_lat, 0.28),
        scaleanchor="x",
        scaleratio=1,
    )
    if not show_table:
        st.plotly_chart(fig, use_container_width=True)
        return
    st.caption(
        "蓝色 O 是用户订单，紫色 S 是骑手或骑手容量槽位，橙/绿实线表示该策略的匹配结果；"
        "紫色虚线表示骑手到商家的取餐段，绿色菱形是订单对应商家。Greedy 是逐单局部选择，Batch 是整批二分图匹配。"
    )
    st.plotly_chart(fig, use_container_width=True)

    match_df = pd.DataFrame(display_matches)
    match_df = match_df[
        ["order_index", "user_id", "merchant_name", "rider_id", "slot_number", "eta", "score", "rider_load"]
    ].rename(
        columns={
            "order_index": "订单序号",
            "user_id": "用户",
            "merchant_name": "商家",
            "rider_id": "骑手",
            "slot_number": "槽位",
            "eta": "ETA",
            "score": "边权",
            "rider_load": "原负载",
        }
    )
    match_df["订单序号"] = pd.to_numeric(match_df["订单序号"], errors="coerce").fillna(0).astype(int) + 1
    match_df["槽位"] = pd.to_numeric(match_df["槽位"], errors="coerce").fillna(0).astype(int) + 1
    st.dataframe(
        match_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ETA": st.column_config.NumberColumn(format="%.1f"),
            "边权": st.column_config.NumberColumn(format="%.3f"),
        },
    )


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
    uses_tripartite = bool(row.get("uses_tripartite", False))
    uses_session_spu = bool(row.get("uses_session_spu", False))
    uses_xquad = bool(row.get("uses_xquad", False))
    model_name = str(row.get("model_name", ""))
    with st.container(border=True):
        if bool(row.get("is_truth", False)):
            st.success("推荐命中")
        st.caption("当前下单商家" if selected else f"第 {int(row['rank'])} 名")
        st.write(f"**{row['merchant_name']}**")
        st.caption(
            f"ID {row['merchant_id']} · 品类 {row['category']} · "
            f"评分 {float(row['poi_score']):.2f} · 均价 {float(row['avg_price']):.1f} · "
            f"距离 {float(row['distance_km']):.2f} km"
        )
        st.write(str(row["reason"]).replace(" / ", "、"))
        m1, m2 = st.columns(2)
        if uses_xquad:
            m1.metric("列表排序分", f"{float(row['rank_score']):.3f}")
        else:
            m1.metric("总分", f"{float(row['final_score']):.3f}")
        m2.metric("ETA", f"{float(row['eta_minutes']):.1f} min")
        if uses_tripartite:
            if uses_xquad:
                score_bar("候选相关性", float(row["final_score"]))
            score_bar("用户偏好", float(row["user_score"]))
            score_bar("商家公平", float(row["fairness"]))
            score_bar("履约速度", float(row["eta_score"]))
            score_bar("供给稳定", float(row["supply"]))
            if uses_session_spu:
                score_bar("会话点击", float(row.get("session_score", 0.0)))
                score_bar("SPU菜品", float(row.get("spu_score", 0.0)))
        elif "LTR" in model_name:
            score_bar("序列特征", float(row["user_score"]))
            st.caption(f"学习排序分 {float(row['final_score']):.3f} · ETA诊断 {float(row['eta_minutes']):.1f} min")
        else:
            score_bar("模型偏好", float(row["user_score"]))
            st.caption(
                f"ETA诊断 {float(row['eta_minutes']):.1f} min · "
                f"商家公平 {float(row['fairness']):.2f} · 供给 {float(row['supply']):.2f}"
            )


data = load_data()
user_ids = data.user_ids
user_set = set(user_ids)
default_user = "8" if "8" in user_set else user_ids[0]
case_options = demo_user_cases(data)
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
    period_label = st.selectbox("下单时段（影响配送耗时）", ["午餐高峰", "晚餐高峰", "早餐", "夜宵"], index=0)
    period_map = {"午餐高峰": "lunch", "晚餐高峰": "dinner", "早餐": "breakfast", "夜宵": "night"}
    period = period_map[period_label]
    strategy_name = st.selectbox(
        "推荐策略",
        DEMO_STRATEGIES,
        index=DEMO_STRATEGIES.index("Seq-xQuAD-Tripartite"),
    )
    top_k = st.slider("推荐数量", min_value=8, max_value=20, value=12, step=1)
    map_source = st.selectbox(
        "地图底图",
        [*MAP_TILE_SOURCES.keys(), "无底图（离线画布）"],
        index=0,
        help="真实底图需联网加载瓦片：国内网络选高德（已做 GCJ-02 坐标纠偏对齐）；海外网络可选 OSM/Carto；离线请选无底图。",
    )
    use_real_map = map_source in MAP_TILE_SOURCES
    if use_real_map:
        MAP_CFG["style"] = MAP_TILE_SOURCES[map_source]["style"]
        MAP_CFG["gcj02"] = MAP_TILE_SOURCES[map_source]["gcj02"]
    preheat_clicked = st.button("预热演示缓存（全部策略 + 高峰回放）", use_container_width=True)
    st.caption("答辩前点一次：预先训练全部策略并跑一遍高峰回放，现场切换即走缓存不卡顿。")

interactive_data = build_interactive_data(user_id, demo_max_orders, tuple(case_options.values()), data)
model_data_key = f"orders={len(interactive_data.orders_train)};user={user_id};max={demo_max_orders}"

if preheat_clicked:
    preheat_progress = st.progress(0.0, text="正在预热演示缓存...")
    for preheat_index, preheat_name in enumerate(DEMO_STRATEGIES, start=1):
        preheat_progress.progress(
            preheat_index / (len(DEMO_STRATEGIES) + 1),
            text=f"预热推荐策略 {preheat_index}/{len(DEMO_STRATEGIES)}：{preheat_name}",
        )
        load_model(preheat_name, model_data_key, interactive_data)
    preheat_progress.progress(
        len(DEMO_STRATEGIES) / (len(DEMO_STRATEGIES) + 1),
        text="预热高峰回放（默认 16 步 × 每步 8 单）...",
    )
    load_peak_trace(model_data_key, top_k, 16, 8, interactive_data)
    preheat_progress.progress(1.0, text="预热完成，策略对比与高峰回放已缓存。")
if len(interactive_data.orders_train) < len(data.orders_train):
    st.caption(
        f"交互推荐模型使用 {len(interactive_data.orders_train):,} 条训练订单；实验页读取离线评估与履约仿真输出。"
    )

with st.spinner(f"正在加载 {strategy_name} 推荐模型，首次运行会建立缓存..."):
    model = load_model(strategy_name, model_data_key, interactive_data)
rec_result = model.recommend([user_id], top_k, {user_id: period})
recs = rec_result.recommendations[user_id]
rec_df = build_recommendation_frame(
    interactive_data,
    model,
    user_id,
    recs,
    period,
    rec_result.scores.get(user_id, {}),
)

users_df = data.users.set_index("user_id", drop=False)
merchants = data.merchants.set_index("wm_poi_id", drop=False)
user_row = users_df.loc[user_id]
riders = generate_riders(data.merchants, n_riders=demo_rider_count, seed=7)

if not rec_df.empty:
    top_row = rec_df.iloc[0]
    avg_eta = float(rec_df["eta_minutes"].mean())
    avg_fairness = float(rec_df["fairness"].mean())
    strategy_caption = (
        "图谱兴趣、曝光与履约"
        if strategy_name == "KG-Tripartite"
        else "会话/SPU、曝光与履约"
        if strategy_name == "Session-SPU-Tripartite"
        else "用户偏好、曝光与履约"
        if "Tripartite" in strategy_name
        else "用户偏好与排序分"
    )
    summary_cols = st.columns(5)
    with summary_cols[0]:
        st.metric("当前用户", str(user_id))
        st.caption(f"历史 {int(user_row.get('history_orders', 0))} 单")
    with summary_cols[1]:
        st.metric("推荐策略", strategy_name)
        st.caption(strategy_caption)
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
    truth_total = len(data.truth_by_user().get(str(user_id), set()))
    truth_hits = int(rec_df["is_truth"].sum()) if "is_truth" in rec_df.columns else 0
    st.caption(
        f"当前推荐 {len(rec_df)} 家，其中 {truth_hits} 家是用户后来实际下单的商家；"
        f"该用户后来一共在 {truth_total} 家商家下过单。绿色提示只用于核对推荐结果，不参与排序。"
    )
    reason_choices = sorted({item for text in rec_df["reason"].astype(str) for item in text.split(" / ") if item})
    selected_reasons = st.multiselect("推荐理由筛选", reason_choices, default=[])
    if selected_reasons:
        card_df = rec_df[
            rec_df["reason"].astype(str).apply(lambda text: any(reason in text for reason in selected_reasons))
        ].copy()
    else:
        card_df = rec_df.copy()
    if card_df.empty:
        st.info("当前筛选无匹配，已显示完整推荐列表。")
        card_df = rec_df.copy()

    st.caption(f"卡片显示 {len(card_df)}/{len(rec_df)} 家商户。")
    for batch in recommendation_card_batches(card_df, columns=3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, batch.iterrows()):
            with col:
                render_recommendation_card(row)

    st.subheader("同一用户的主线策略对比")
    run_strategy_compare = st.checkbox("计算主线策略对比", value=False)
    if run_strategy_compare:
        strategy_rows = []
        for name in DEMO_STRATEGIES:
            with st.spinner(f"加载 {name} 并生成对比..."):
                candidate_model = load_model(name, model_data_key, interactive_data)
                candidate_result = candidate_model.recommend([user_id], top_k, {user_id: period})
                candidate_recs = candidate_result.recommendations[user_id]
                candidate_frame = build_recommendation_frame(
                    interactive_data,
                    candidate_model,
                    user_id,
                    candidate_recs,
                    period,
                    candidate_result.scores.get(user_id, {}),
                )
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
        f"{'[推荐命中] ' if row.is_truth else ''}第 {int(row.rank)} 名 · {row.merchant_name} · 商家编号 {row.merchant_id}"
        for row in rec_df.itertuples(index=False)
    ]
    chosen_label = st.selectbox("模拟下单商家", option_labels)
    chosen_idx = option_labels.index(chosen_label)
    chosen_row = rec_df.iloc[chosen_idx]
    chosen_id = str(chosen_row["merchant_id"])
    merchant_row = merchants.loc[chosen_id]

    st.subheader("骑手匹配结果")
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
        if bool(chosen_row.get("uses_tripartite", False)):
            contrib_rows = [
                {"component": "用户偏好", "weighted_score": chosen_row["user_contrib"]},
                {"component": "商家公平", "weighted_score": chosen_row["fairness_contrib"]},
                {"component": "ETA 履约", "weighted_score": chosen_row["eta_contrib"]},
                {"component": "供给稳定", "weighted_score": chosen_row["supply_contrib"]},
            ]
            if bool(chosen_row.get("uses_session_spu", False)):
                contrib_rows.extend(
                    [
                        {"component": "会话点击", "weighted_score": chosen_row.get("session_contrib", 0.0)},
                        {"component": "SPU菜品", "weighted_score": chosen_row.get("spu_contrib", 0.0)},
                    ]
                )
        else:
            contrib_rows = [{"component": "模型排序分", "weighted_score": chosen_row["final_score"]}]
        contrib = pd.DataFrame(contrib_rows)
        contrib_fig = px.bar(
            contrib,
            x="component",
            y="weighted_score",
            color="component",
            title=f"{strategy_name} 分数组成",
            color_discrete_sequence=["#2563eb", "#0f766e", "#dc6803", "#7c3aed", "#0891b2", "#b45309"],
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

    st.subheader("三方视角：这一单对用户、商家、骑手分别意味着什么")
    view_user, view_merchant, view_rider = st.tabs(["用户：菜品级推荐", "商家：供给压力", "骑手：顺路单"])
    with view_user:
        dishes = top_dishes_for_user(interactive_data, user_id, chosen_id)
        if dishes.empty:
            st.info("该商家暂无菜品级订单记录（SPU 信号缺失时自动隐藏）。")
        else:
            st.caption("按该商家菜品历史销量排序，标注与用户历史口味类目（Top3）契合的菜品；名称与类目为 TRD 匿名化编号。")
            st.dataframe(dishes, use_container_width=True, hide_index=True)
    with view_merchant:
        pressure = merchant_supply_pressure(data, merchant_row, period)
        risk_color = {"高": "🔴", "中": "🟠", "低": "🟢"}[pressure["risk_level"]]
        p1, p2, p3 = st.columns(3)
        p1.metric("高峰爆单风险", f"{risk_color} {pressure['risk_level']}")
        p2.metric("需求分位", f"{pressure['demand_percentile']:.0%}")
        p3.metric("供给能力分", f"{pressure['capacity_score']:.2f}")
        st.caption(pressure["risk_advice"])
        quota = pressure["category_quota"]
        if isinstance(quota, pd.DataFrame) and not quota.empty:
            st.caption("按品类受欢迎程度给出的高峰供给配额建议（受欢迎度 × 高峰倍率）：")
            st.dataframe(quota, use_container_width=True, hide_index=True)
    with view_rider:
        matched_rider_rows = riders[riders["rider_id"].astype(str) == str(load_aware["rider_id"])]
        if matched_rider_rows.empty:
            st.info("暂无匹配骑手。")
        else:
            st.caption(
                "骑手视角：以当前配送路径（骑手→商家→用户）为基准，评估附近商家新单的边际绕行成本——"
                "顺路单只需极小的额外时间，这正是路径感知派单（RouteBatch）合并订单的依据。"
            )
            opportunities = enroute_opportunities(
                interactive_data, matched_rider_rows.iloc[0], merchant_row, user_row, period
            )
            if opportunities.empty:
                st.info("附近暂无可评估的顺路单。")
            else:
                st.dataframe(opportunities, use_container_width=True, hide_index=True)

    candidate_left, candidate_right = st.columns([1.1, 1])
    if not rider_candidates.empty:
        candidate_view = rider_candidates.copy()
        candidate_view["status"] = [
            "已派单" if str(rider_id) == str(load_aware["rider_id"]) else "候选"
            for rider_id in candidate_view["rider_id"]
        ]
        with candidate_left:
            st.subheader("负载感知派单")
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
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10, color="#111827")),
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
        flow_fig.update_layout(title="当前订单路径", height=340, margin=dict(l=8, r=8, t=52, b=8))
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
        if not use_real_map:
            st.caption("半透明圆只是距离参考，不是等高线、热力图或真实配送边界。")
        map_fig = go.Figure()
        map_fig.add_trace(
            map_scatter(
                use_real_map,
                nearby_users["lng"],
                nearby_users["lat"],
                use_gl=True,
                mode="markers",
                name="附近用户样本",
                text=nearby_users["user_id"].astype(str),
                marker=dict(size=5, color="#93c5fd", opacity=0.28),
                hovertemplate="用户 %{text}",
            )
        )
        map_fig.add_trace(
            map_scatter(
                use_real_map,
                nearby_merchants["lng"],
                nearby_merchants["lat"],
                use_gl=True,
                mode="markers",
                name="附近商家样本",
                text=nearby_merchants["wm_poi_id"].astype(str),
                marker=dict(size=5, color="#86efac", opacity=0.32),
                hovertemplate="商家 %{text}",
            )
        )
        map_fig.add_trace(
            map_scatter(
                use_real_map,
                merchant_circle_x,
                merchant_circle_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(15, 118, 110, 0.07)",
                line=dict(color="rgba(15, 118, 110, 0.42)", width=1.5),
                name="商家 2.5km 圈",
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            map_scatter(
                use_real_map,
                user_circle_x,
                user_circle_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.06)",
                line=dict(color="rgba(37, 99, 235, 0.38)", width=1.5),
                name="用户 2.5km 圈",
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            map_scatter(
                use_real_map,
                nearby_riders["lng"],
                nearby_riders["lat"],
                use_gl=True,
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
                map_scatter(
                    use_real_map,
                    rider_candidates["lng"],
                    rider_candidates["lat"],
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
            map_scatter(
                use_real_map,
                [rider_lng, merchant_lng],
                [rider_lat, merchant_lat],
                mode="lines",
                name="取餐段",
                line=dict(color="#7c3aed", width=3, dash="dot") if not use_real_map else dict(color="#7c3aed", width=3),
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            map_scatter(
                use_real_map,
                [merchant_lng, user_lng],
                [merchant_lat, user_lat],
                mode="lines",
                name="配送段",
                line=dict(color="#dc6803", width=3),
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            map_scatter(
                use_real_map,
                [user_lng],
                [user_lat],
                mode="markers+text",
                name="用户",
                text=[f"用户 {user_id}"],
                textposition="top center",
                marker=dict(size=16, color="#2563eb", symbol="circle", line=dict(color="#ffffff", width=1.5)),
            )
        )
        map_fig.add_trace(
            map_scatter(
                use_real_map,
                rec_df["lng"],
                rec_df["lat"],
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
            map_scatter(
                use_real_map,
                [merchant_lng],
                [merchant_lat],
                mode="markers+text",
                name="下单商家",
                text=[str(chosen_row["merchant_name"])],
                textposition="bottom center",
                marker=dict(size=20, color="#dc6803", symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            )
        )
        map_fig.add_trace(
            map_scatter(
                use_real_map,
                [rider_lng],
                [rider_lat],
                mode="markers+text",
                name="匹配骑手",
                text=[str(rider_point["rider_id"])],
                textposition="top center",
                marker=dict(size=18, color="#7c3aed", symbol="square", line=dict(color="#ffffff", width=1.5)),
            )
        )
        focus_lng = [user_lng, merchant_lng, rider_lng] + rec_df["lng"].dropna().astype(float).tolist()
        focus_lat = [user_lat, merchant_lat, rider_lat] + rec_df["lat"].dropna().astype(float).tolist()
        focus_lng += rider_candidates["lng"].dropna().astype(float).tolist()
        focus_lat += rider_candidates["lat"].dropna().astype(float).tolist()
        apply_map_layout(
            map_fig,
            use_real_map,
            focus_lng,
            focus_lat,
            title="当前订单附近的用户、商家和骑手（真实城市底图）" if use_real_map else "当前订单附近的用户、商家和骑手",
            height=460,
        )
        st.plotly_chart(map_fig, use_container_width=True)

    st.subheader("推荐明细")
    table_cols = [
        "rank",
        "truth_label",
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
    if "rank_score" in rec_df.columns and rec_df.get("uses_xquad", pd.Series(dtype=bool)).any():
        table_cols.insert(3, "rank_score")
    if "session_score" in rec_df.columns and rec_df["session_score"].gt(0).any():
        table_cols.insert(-2, "session_score")
    if "spu_score" in rec_df.columns and rec_df["spu_score"].gt(0).any():
        table_cols.insert(-2, "spu_score")
    score_label = "候选相关性" if "rank_score" in table_cols else "总分"
    display_df = rec_df[table_cols].rename(
        columns={
            "rank": "排名",
            "truth_label": "推荐结果",
            "merchant_name": "商家",
            "category": "品类",
            "rank_score": "列表排序分",
            "final_score": score_label,
            "user_score": "用户偏好",
            "fairness": "商家公平",
            "eta_minutes": "ETA",
            "supply": "供给",
            "session_score": "会话",
            "spu_score": "SPU菜品",
            "distance_km": "距离km",
            "reason": "解释",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

@st.cache_data(show_spinner=False)
def load_delivery_replay(data_key: str, strategy: str, dispatch: str, _data: PreparedData) -> dict:
    from foodflow.demo_support import build_delivery_replay

    model = load_model(strategy, data_key, _data)
    return build_delivery_replay(
        _data, model, dispatch=dispatch, steps=8, requests_per_step=8, n_riders=40, seed=33, top_k=10
    )


def _replay_frame_traces(frame: dict, real_map: bool) -> list:
    return [
        map_scatter(
            real_map,
            frame["leg_lng"],
            frame["leg_lat"],
            mode="lines",
            name="在途路径",
            line=dict(color="#7c3aed", width=2),
            hoverinfo="skip",
        ),
        map_scatter(
            real_map,
            frame["rider_lng"],
            frame["rider_lat"],
            mode="markers",
            name="骑手（颜色=负载）",
            text=frame["rider_id"],
            marker=dict(
                size=12,
                color=frame["rider_load"],
                colorscale=[[0.0, "#93c5fd"], [0.4, "#7c3aed"], [1.0, "#312e81"]],
                cmin=0,
                cmax=3,
            ),
            hovertemplate="骑手 %{text}",
        ),
        map_scatter(
            real_map,
            frame["m_lng"],
            frame["m_lat"],
            mode="markers",
            name="出餐中商家",
            marker=dict(size=10, color="#dc6803", opacity=0.85),
            hoverinfo="skip",
        ),
        map_scatter(
            real_map,
            frame["u_lng"],
            frame["u_lat"],
            mode="markers",
            name="待收餐用户",
            marker=dict(size=8, color="#2563eb", opacity=0.85),
            hoverinfo="skip",
        ),
    ]


with tab_peak:
    st.subheader("骑手履约动画")
    st.caption(
        "连续播放高峰期的完整履约过程：每 5 分钟到达一批订单，路径感知派单把新单插入骑手当前路径，"
        "骑手沿取餐、送达路径逐分钟移动，颜色随负载加深；紫色线为在途路径，橙点为出餐中商家，蓝点为待收餐用户。"
    )
    anim_l, anim_r = st.columns([1.6, 1.4])
    with anim_l:
        replay_dispatch_label = st.radio(
            "派单机制",
            ["顺路合单（RouteBatch）", "路径最小 ETA（RouteMinETA）"],
            horizontal=True,
            key="replay_dispatch",
        )
    with anim_r:
        run_replay_anim = st.checkbox("生成履约动画（首次约 10-30 秒）", value=False)
    if run_replay_anim:
        replay_dispatch = "min_eta" if "MinETA" in replay_dispatch_label else "detour_eta"
        with st.spinner("正在逐分钟模拟骑手履约..."):
            replay = load_delivery_replay(model_data_key, strategy_name, replay_dispatch, interactive_data)
        replay_frames = replay.get("frames", [])
        if not replay_frames:
            st.info("当前数据没有可回放的订单。")
        else:
            focus_lng = [v for f in replay_frames for v in (f["m_lng"] + f["u_lng"])] or replay_frames[0]["rider_lng"]
            focus_lat = [v for f in replay_frames for v in (f["m_lat"] + f["u_lat"])] or replay_frames[0]["rider_lat"]
            replay_fig = go.Figure(
                data=_replay_frame_traces(replay_frames[0], use_real_map),
                frames=[
                    go.Frame(
                        data=_replay_frame_traces(frame, use_real_map),
                        name=str(frame["minute"]),
                        layout=go.Layout(
                            title=(
                                f"第 {frame['minute']} 分钟 · 已派 {frame['completed']} 单 · "
                                f"忙碌骑手 {frame['busy_riders']}/{replay['n_riders']} · "
                                f"累计平均 ETA {frame['avg_eta']:.1f} min"
                            )
                        ),
                    )
                    for frame in replay_frames
                ],
            )
            apply_map_layout(
                replay_fig,
                use_real_map,
                focus_lng,
                focus_lat,
                title=f"第 0 分钟 · 已派 {replay_frames[0]['completed']} 单",
                height=560,
            )
            replay_fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        x=0.0,
                        y=1.12,
                        buttons=[
                            dict(
                                label="▶ 播放",
                                method="animate",
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=280, redraw=True),
                                        fromcurrent=True,
                                        transition=dict(duration=0),
                                    ),
                                ],
                            ),
                            dict(
                                label="⏸ 暂停",
                                method="animate",
                                args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))],
                            ),
                        ],
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        currentvalue=dict(prefix="分钟: "),
                        steps=[
                            dict(
                                label=str(frame["minute"]),
                                method="animate",
                                args=[
                                    [str(frame["minute"])],
                                    dict(mode="immediate", frame=dict(duration=0, redraw=True)),
                                ],
                            )
                            for frame in replay_frames
                        ],
                    )
                ],
            )
            st.plotly_chart(replay_fig, use_container_width=True)
            st.caption(
                "点击 ▶ 播放观看完整履约过程；顺路合单模式下可观察骑手在配送途中顺路接起附近新单（负载颜色加深、路径延长）。"
            )

    st.divider()
    st.subheader("午餐高峰回放")
    st.caption(
        "按时间步模拟午餐高峰请求流：每步抽一批用户，先推荐商家，再模拟用户选店和骑手派单。"
        "Batch 策略会把同一时间步的订单与骑手容量槽位做二分图最大权匹配。"
    )
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
                    yaxis=dict(tickfont=dict(color="#111827")),
                    xaxis=dict(tickfont=dict(color="#111827")),
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
                    yaxis=dict(tickfont=dict(color="#111827")),
                    xaxis=dict(tickfont=dict(color="#111827")),
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

            if "assignment_mode" in trace_df.columns and "step_matches_json" in trace_df.columns:
                match_rows = trace_df[
                    trace_df["step_matches_json"].fillna("").astype(str).str.len().gt(2)
                ].copy()

                if not match_rows.empty:
                    st.subheader("仿真策略相对位置")
                    match_steps = sorted(match_rows["step"].dropna().astype(int).unique().tolist())
                    selected_match_step = st.selectbox(
                        "时间步",
                        match_steps,
                        index=max(len(match_steps) - 1, 0),
                        key="peak_match_step_grid",
                    )
                    st.caption(
                        "每张小地图展示同一时间步内的批量订单匹配：蓝色 O 为用户订单，紫色 S 为骑手或骑手槽位，"
                        "绿色菱形为商家，实线为最终匹配，紫色虚线为骑手到商家的取餐段。"
                    )
                    step_map_rows = match_rows[
                        match_rows["step"].astype(int) == int(selected_match_step)
                    ].copy()
                    if step_map_rows.empty:
                        step_map_rows = match_rows.sort_values("step").groupby("policy", as_index=False).tail(1)
                    else:
                        step_map_rows = (
                            step_map_rows.sort_values("step").groupby("policy", as_index=False).tail(1)
                        )
                    map_order = {policy: index for index, policy in enumerate(ops_board["policy"].astype(str).tolist())}
                    step_map_rows = step_map_rows.assign(
                        _order=step_map_rows["policy"].astype(str).map(map_order).fillna(999)
                    ).sort_values("_order")
                    for start in range(0, len(step_map_rows), 3):
                        cols = st.columns(min(3, len(step_map_rows) - start))
                        for col, (_, map_row) in zip(cols, step_map_rows.iloc[start : start + 3].iterrows()):
                            with col:
                                render_step_matching_view(map_row, height=300, show_table=False)

            trace_display = trace_df.drop(columns=["batch_matches_json", "step_matches_json"], errors="ignore")
            st.dataframe(trace_display, use_container_width=True, hide_index=True)
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
            "学习排序",
            f"{LEARNED_LTR_MODEL}：用数据学习候选排序",
            "默认使用 LightGBM LambdaRank；如果运行环境缺少 LightGBM，会显式显示 Logistic-LTR 后备排序器。",
            model_metric_text(LEARNED_LTR_MODEL, "Coverage@20", "Coverage@20"),
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
            policy_metric_text("Seq-xQuAD-Tripartite + Greedy", "platform_utility", "Platform Utility"),
        ),
        (
            "菜品增强",
            "Session-SPU：加入会话和菜品信号",
            "用训练期点击商家和 SPU 菜品类目重合扩展候选，并保留菜品匹配证据。",
            model_metric_text("Session-SPU-Tripartite", "NDCG@20", "NDCG@20"),
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
                {"评价对象": "学习排序", "项目做法": "用同一批序列特征训练学习排序器", "对应模块": LEARNED_LTR_MODEL},
                {"评价对象": "商家曝光", "项目做法": "把曝光公平、ETA 和供给情况接到重排里", "对应模块": "Seq-xQuAD-Tripartite"},
                {"评价对象": "菜品偏好", "项目做法": "用训练期会话点击和 SPU 菜品类目重合增强候选", "对应模块": "Session-SPU-Tripartite"},
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

        st.subheader("推荐算法离线评估")
        algorithm_eval = build_algorithm_evaluation_frame(offline, sim)
        st.caption("推荐指标来自离线评估表；带 ETA、超时率和综合分的证据来自同一推荐模型对应的履约仿真策略。")
        st.dataframe(
            algorithm_eval,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Recall@20": st.column_config.NumberColumn(format="%.4f"),
                "NDCG@20": st.column_config.NumberColumn(format="%.4f"),
                "Coverage@20": st.column_config.NumberColumn(format="%.4f"),
                "曝光Gini": st.column_config.NumberColumn(format="%.4f"),
                "长尾曝光": st.column_config.NumberColumn(format="%.4f"),
            },
        )

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
                title="用户侧推荐命中",
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
