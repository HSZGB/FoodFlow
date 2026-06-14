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


def _data_mode_text(data_note_path: Path, data_audit_path: Path | None = None) -> str:
    if not data_note_path.exists():
        return "当前报告未读取到 `data_note.json`，请确认已运行 `make preprocess`。"
    note = json.loads(data_note_path.read_text(encoding="utf-8"))
    if note.get("mode") == "mock":
        return (
            "当前表格由 `make smoke` 的 mock TRD-like 数据生成，用于验证工程闭环。"
            "正式提交前运行 `make download preprocess eval simulate figures report`，即可用真实 TRD txt 文件刷新结果。"
        )
    if data_audit_path and data_audit_path.exists():
        audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
        if audit.get("train_mode") == "full":
            return (
                "当前表格由真实 TRD txt 文件生成，训练集使用完整 `orders_train.txt`。"
                f"审计记录显示原始训练订单 {audit.get('raw_train_orders')} 条，处理后训练订单 "
                f"{audit.get('processed_train_orders')} 条。"
            )
        return (
            "当前表格由真实 TRD txt 文件生成，但训练集采用固定 seed 抽样以保证 CPU 可运行。"
            f"审计记录显示原始训练订单 {audit.get('raw_train_orders')} 条，处理后训练订单 "
            f"{audit.get('processed_train_orders')} 条，训练订单使用比例 "
            f"{float(audit.get('train_sample_fraction', 0.0)):.4f}。"
        )
    sample_orders = note.get("sample_orders")
    if sample_orders in {None, 0, "0"}:
        return "当前表格由真实 TRD txt 文件生成，训练集按 `sample_orders=0` 使用全量训练订单。"
    return (
        "当前表格由真实 TRD txt 文件生成，训练集采用固定 seed 抽样。"
        f"`data_note.json` 记录处理后训练订单 {note.get('train_orders')} 条，sample_orders={sample_orders}。"
    )


def _data_audit_summary(path: Path) -> str:
    if not path.exists():
        return "_尚未生成数据审计文件。运行 `make audit` 可生成 `outputs/results/data_audit.json` 和 `docs/DATA_AUDIT.md`。_"
    audit = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        ("必需 TRD 文件齐全", audit.get("required_raw_files_present")),
        ("训练集处理模式", audit.get("train_mode")),
        ("原始训练订单数", audit.get("raw_train_orders")),
        ("处理后训练订单数", audit.get("processed_train_orders")),
        ("训练订单使用比例", f"{float(audit.get('train_sample_fraction', 0.0)):.4f}"),
        ("骑手数据边界", "合成 proxy，非真实派单记录"),
    ]
    return pd.DataFrame(rows, columns=["项目", "值"]).to_markdown(index=False)


def build_report(
    results_dir: Path,
    figures_dir: Path,
    output: Path,
    data_note_path: Path = Path("data/processed/data_note.json"),
    data_audit_path: Path = Path("outputs/results/data_audit.json"),
) -> Path:
    ensure_dir(output.parent)
    offline_table = _markdown_table(results_dir / "offline_metrics.csv")
    simulation_table = _markdown_table(results_dir / "simulation_metrics.csv")
    data_mode = _data_mode_text(data_note_path, data_audit_path)
    data_audit = _data_audit_summary(data_audit_path)
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

### 2.1 数据审计

{data_audit}

## 3. 方法设计

默认实验只保留代表性策略集合：Popular、Repeat、BPR-MF、UserOnly、Seq-Tuned、Seq-Tuned-xQuAD、Seq-xQuAD-Tripartite 和 Ours-Full。Popular 是基础热度对照；Repeat 反映外卖复购特征；BPR-MF 是传统隐式反馈排序基线；UserOnly 使用品类、复购、价格、时段和商家质量；Seq-Tuned 借鉴 FPMC/下一篮子推荐和会话推荐思想，在同一组可解释序列特征上增强复购和商家转移信号，用作离线准确率前沿；Seq-Tuned-xQuAD 在高准确序列模型后做列表级覆盖与长尾曝光重排；Seq-xQuAD-Tripartite 把列表级覆盖、商家公平、ETA 和供给约束接到同一个重排器；Ours-Full 作为原始三方加权重排对照。

项目代码中保留了若干历史消融类，便于复现实验过程；但默认结果和 demo 不再铺开所有候选，答辩主线聚焦为：基础对照 -> 传统排序 -> 用户画像 -> 高准确序列推荐 -> 曝光/校准重排 -> 三方履约重排。

三方仿真保留 6 条代表链路：Popular + Nearest、UserOnly + MinETA、Seq-Tuned + MinETA、Seq-Tuned-xQuAD + MinETA、Seq-xQuAD-Tripartite 和 Ours-Full。每轮模拟午餐高峰的一批用户请求，推荐列表经过选择模型产生订单，再由最近骑手、最小 ETA 或负载感知策略派单。

## 4. 离线推荐结果

{offline_table}

推荐侧指标使用 Recall@K、NDCG@K、MRR@K 和 HitRate@K；商家侧指标使用 Coverage、Long-tail Exposure 和 Exposure Gini；校准侧指标使用 CategoryJSD@20，衡量推荐列表品类分布与用户历史品类分布的 Jensen-Shannon divergence，数值越低越贴近用户习惯。这样既能看推荐是否命中真实下单，也能看曝光是否过度集中，以及列表是否偏离用户长期品类偏好。全量 TRD 结果中，Seq-Tuned 通常代表离线准确率前沿，说明外卖推荐强烈受复购序列和商家转移影响；Seq-Tuned-xQuAD 在保持高 Recall 的同时改善曝光集中和品类校准；Seq-xQuAD-Tripartite 的离线准确性低于纯用户侧序列模型，但 Exposure Gini 更低，并且后续需要结合仿真指标判断其系统级收益。

## 5. 动态履约仿真结果

{simulation_table}

履约侧指标包括完成订单数、平均 ETA、超时率、骑手负载标准差和平台综合效用。多目标推荐的目标不是所有单项指标都最大，而是在准确性、公平性和履约可行性之间取得更适合外卖平台的折中。全量 TRD 仿真中，Seq-Tuned 和 Seq-Tuned-xQuAD 提升了离线推荐准确性，但进入派单后平台效用仍要与 ETA、超时率和骑手负载一起判断；Seq-xQuAD-Tripartite 展示了把列表级重排与三方履约约束结合的系统级价值。

为避免只展示单点权重，图表中额外生成 `pareto_recall_utility.png` 和 `tripartite_frontier.csv`，把 Recall@20、Exposure Gini、平均 ETA、超时率和平台效用合并为非支配前沿视角。答辩时可以用这张图说明：Seq-Tuned / Seq-Tuned-xQuAD 代表离线准确率前沿，Seq-xQuAD-Tripartite 代表系统效用前沿，它们共同构成三方推荐的权衡边界。

## 6. 图表展示

{figures_md}

## 7. 结论与局限

FoodFlow 的答辩故事可以概括为三步：第一，公开外卖订单数据上的推荐实验证明模型不是随机的，并用 Seq-Tuned 把离线排序指标继续推高；第二，显式加入商家曝光、长尾曝光、品类校准和 Gini 指标，让推荐不再只围绕用户命中率讨论；第三，Seq-xQuAD-Tripartite 和负载感知派单能降低 ETA、超时风险并提升平台效用，从而体现多主体平台的系统级优化。

局限是骑手数据来自合成仿真，不能替代工业级派单数据；BPR-MF 是轻量实现，未追求大规模深度模型最优性能；平台效用权重是课程项目中的解释性设置，需要结合业务偏好做敏感性分析。
"""
    output.write_text(text, encoding="utf-8")
    return output
