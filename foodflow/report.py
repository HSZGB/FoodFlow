from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io import ensure_dir


def _markdown_table(path: Path) -> str:
    if not path.exists():
        return "_尚未生成该结果表。_"
    df = pd.read_csv(path)
    return df.round(4).to_markdown(index=False)


def _data_mode_text(data_note_path: Path) -> str:
    if not data_note_path.exists():
        return "当前报告未读取到 `data_note.json`，请确认已运行 `make preprocess`。"
    note = json.loads(data_note_path.read_text(encoding="utf-8"))
    if note.get("mode") == "mock":
        return (
            "当前表格由 `make smoke` 的 mock TRD-like 数据生成，用于验证工程闭环。"
            "正式提交前运行 `make download preprocess eval simulate figures report`，即可用真实 TRD txt 文件刷新结果。"
        )
    return "当前表格由 TRD txt 文件生成，原始数据来源见 Zenodo DOI。"


def build_report(
    results_dir: Path,
    figures_dir: Path,
    output: Path,
    data_note_path: Path = Path("data/processed/data_note.json"),
) -> Path:
    ensure_dir(output.parent)
    offline_table = _markdown_table(results_dir / "offline_metrics.csv")
    simulation_table = _markdown_table(results_dir / "simulation_metrics.csv")
    data_mode = _data_mode_text(data_note_path)
    figure_lines = []
    if figures_dir.exists():
        for fig in sorted(figures_dir.glob("*.png")):
            figure_lines.append(f"![{fig.stem}](../outputs/figures/{fig.name})")
    figures_md = "\n\n".join(figure_lines) if figure_lines else "_尚未生成图表。_"

    text = f"""# FoodFlow：融合公平重排与履约仿真的外卖三方推荐系统

## 1. 项目目标

本项目面向外卖平台推荐，围绕用户、商家、骑手三方利益构建一个可运行的课程大作业原型。系统先用真实外卖订单数据评估用户-商家 Top-K 推荐，再把推荐结果转化为模拟订单，进入骑手匹配与高峰期履约仿真，最终用推荐指标和系统级指标共同评价策略。

核心观点是：外卖推荐不能只追求 Recall 或 NDCG。一个推荐列表如果把用户推向远距离、高拥堵、骑手供给不足的商家，平台体验可能变差。因此 FoodFlow 使用三方重排，在用户偏好之外引入商家公平、ETA 和骑手负载。

## 2. 数据来源

主数据集为 Takeout Recommendation Dataset (TRD)，Zenodo DOI：`10.5281/zenodo.8025855`。数据来自美团外卖北京 11 个商圈，时间范围为 2021-03-01 至 2021-03-28，包含用户、餐厅、菜品、训练订单、测试订单和测试标签。

本项目默认下载 TRD 的 txt 文件，不下载约 1.8GB 的 `graph.bin`，因为核心实现不依赖 DGL 图。骑手位置、在线状态、负载和收入为合成仿真数据，生成规则固定 seed，并在实验中仅作为履约约束 proxy，不声称为真实骑手数据。

{data_mode}

## 3. 方法设计

实现的推荐策略包括 Random、Popular、Repeat、ItemCF、BPR-MF、UserOnly 和 Ours-Full。Random 和 Popular 是基础对照；Repeat 反映外卖复购特征；ItemCF 和 BPR-MF 是传统协同过滤基线；UserOnly 使用品类、复购、价格、时段和商家质量；Ours-Full 在 UserOnly 上加入商家曝光公平、ETA 和供给分。

三方仿真包括 Popular + Nearest、UserOnly + Nearest、UserOnly + MinETA、Ours w/o Fairness 和 Ours-Full。每轮模拟午餐高峰的一批用户请求，推荐列表经过选择模型产生订单，再由最近骑手、最小 ETA 或负载感知策略派单。

## 4. 离线推荐结果

{offline_table}

推荐侧指标使用 Recall@K、NDCG@K、MRR@K 和 HitRate@K；商家侧指标使用 Coverage、Long-tail Exposure 和 Exposure Gini。这样既能看推荐是否命中真实下单，也能看曝光是否过度集中。

## 5. 动态履约仿真结果

{simulation_table}

履约侧指标包括完成订单数、平均 ETA、超时率、骑手负载标准差和平台综合效用。Ours-Full 的目标不是所有单项指标都最大，而是在准确性、公平性和履约可行性之间取得更适合外卖平台的折中。

## 6. 图表展示

{figures_md}

## 7. 结论与局限

FoodFlow 的答辩故事可以概括为三步：第一，公开外卖订单数据上的推荐实验证明模型不是随机的；第二，商家公平重排能够提升覆盖率并降低曝光集中；第三，履约感知重排和负载感知派单能降低 ETA、超时风险或骑手负载不均衡，从而体现多主体平台的系统级优化。

局限是骑手数据来自合成仿真，不能替代工业级派单数据；BPR-MF 是轻量实现，未追求大规模深度模型最优性能；平台效用权重是课程项目中的解释性设置，需要结合业务偏好做敏感性分析。
"""
    output.write_text(text, encoding="utf-8")
    return output
