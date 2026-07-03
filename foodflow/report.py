from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io import ensure_dir
from .recommenders import learned_ltr_model_name


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
            "当前表格由 mock TRD-like 数据生成，用于验证工程闭环。"
            "正式实验请运行 `make download preprocess-full eval simulate figures report`。"
        )
    if data_audit_path and data_audit_path.exists():
        audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
        if audit.get("train_mode") == "full":
            return (
                "当前表格由真实 TRD txt 文件生成，训练集使用完整 `orders_train.txt`。"
                f"审计记录显示原始训练订单 {audit.get('raw_train_orders')} 条，"
                f"处理后训练订单 {audit.get('processed_train_orders')} 条。"
            )
        return (
            "当前表格由真实 TRD txt 文件生成，但训练集采用固定 seed 抽样以保证 CPU 可运行。"
            f"审计记录显示原始训练订单 {audit.get('raw_train_orders')} 条，"
            f"处理后训练订单 {audit.get('processed_train_orders')} 条，"
            f"训练订单使用比例 {float(audit.get('train_sample_fraction', 0.0)):.4f}。"
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
        return "_尚未生成数据审计文件。运行 `make audit` 可生成审计结果。_"
    audit = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        ("必需 TRD 文件齐全", audit.get("required_raw_files_present")),
        ("训练集处理模式", audit.get("train_mode")),
        ("原始训练订单数", audit.get("raw_train_orders")),
        ("处理后训练订单数", audit.get("processed_train_orders")),
        ("训练订单使用比例", f"{float(audit.get('train_sample_fraction', 0.0)):.4f}"),
        ("骑手数据边界", "默认合成 proxy；可用 LaDe delivery CSV 校准，非真实外卖派单记录"),
    ]
    return pd.DataFrame(rows, columns=["项目", "值"]).to_markdown(index=False)


def _figure_links(figures_dir: Path) -> str:
    if not figures_dir.exists():
        return "_尚未生成图表。_"
    lines = [f"![{fig.stem}](../outputs/figures/{fig.name})" for fig in sorted(figures_dir.glob("*.png"))]
    return "\n\n".join(lines) if lines else "_尚未生成图表。_"


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
    learned_ltr = learned_ltr_model_name()
    figures_md = _figure_links(figures_dir)

    text = f"""# FoodFlow：融合学习排序、公平重排与履约仿真的外卖推荐系统

## 1. 项目目标

FoodFlow 面向外卖平台推荐场景，先用真实外卖订单数据评估用户-商家 Top-K 推荐，再把推荐结果转化为模拟订单，进入骑手匹配与高峰期履约仿真。项目同时观察推荐准确性、商家曝光公平、ETA、超时率、骑手负载和平台综合效用。

核心观点是：外卖推荐不能只追求 Recall 或 NDCG。一个推荐列表如果把用户推向远距离、高拥堵、骑手供给不足的商家，平台体验可能变差。因此 FoodFlow 在用户偏好之外引入商家公平、ETA、供给约束和骑手负载。

## 2. 数据来源

主数据集为 Takeout Recommendation Dataset (TRD)，Zenodo DOI：`10.5281/zenodo.8025855`。数据来自美团外卖北京 11 个商圈，时间范围为 2021-03-01 至 2021-03-28，包含用户、餐厅、菜品、训练订单、测试订单和测试标签。

项目默认下载 TRD 的 txt 文件，不下载约 1.8GB 的 `graph.bin`，因为核心实现不依赖 DGL 图。TRD 不包含真实骑手状态和派单记录，因此默认骑手位置、在线状态、负载和收入为固定 seed 合成 proxy。若提供 LaDe delivery CSV，系统可从 task-accept/task-finish 时空事件估计骑手速度、服务时长和任务重叠负载，但 LaDe 不是外卖平台数据，不能声称为真实外卖派单记录。

{data_mode}

### 2.1 数据审计

{data_audit}

## 3. 方法设计

默认实验保留 8 个代表策略：Popular、BPR-MF、UserOnly、Seq-Tuned、{learned_ltr}、Seq-xQuAD-Tripartite、Session-SPU-Tripartite 和 KG-Tripartite。LightGBM 不可用时显式使用 Logistic-LTR；Session-SPU-Tripartite 只使用训练期点击会话与菜品信号，避免把测试期行为泄漏到离线排序；KG-Tripartite 在其上叠加时间衰减的知识图谱兴趣信号（品类/商圈/价位节点的关系加权匹配），是 kg-demo 子项目动态 KG 注意力模型的免训练近似。

Popular 是全局热度对照；BPR-MF 是传统隐式反馈矩阵分解；UserOnly 使用品类、复购、价格、时段和商家质量构造可解释画像分；LightGBM-LTR 复用 Seq-Tuned 的 recency、repeat、transition、category、popularity、quality 等特征，但用 LightGBM LambdaRank 学习排序函数，替代手动硬编码的 `SEQ_TUNED_WEIGHTS`；Seq-xQuAD-Tripartite 把列表级覆盖、商家公平、ETA 和供给约束接到同一个重排器；Session-SPU-Tripartite 进一步加入 TRD session 点击候选和菜品 SPU 类目偏好，用来评估更丰富的真实行为信号是否改善履约链路。

`Seq-Tuned` 保留为可解释规则基线；LightGBM 不可用时，系统使用 Logistic-LTR，而不是把规则模型伪装成学习排序。仿真共 9 条链路，并对三方重排系列同时运行逐单贪心和容量槽位批量最大权匹配。各策略共享请求流和初始骑手池，同一推荐器还共享 MNL 选择噪声，减少随机场景差异。仿真支持多随机种子重复运行（`--simulation-seeds`），结果表在均值之外报告跨种子标准差与 95% 置信区间，策略间对比以该口径为准。

轻量 KG 解释用于吸收知识图谱路线的可解释性亮点，但不引入高风险图神经网络训练。系统从训练订单、商家品类、商圈/区域和价格段构造 `user-ordered-poi`、`user-prefers-category`、`poi-has-category`、`poi-located-in-area`、`has-price-range` 等路径。`explain-case` 会输出类似 “user -> category <- poi” 的证据路径，并同时保留 ETA、曝光补偿等真实打分字段，避免空泛模板解释。

三方重排的用户分、公平分、ETA 分和供给分在候选集合内做 min-max 归一化后再加权合成，避免不同数值尺度让权重失去解释性。离线表中 `LightGBM-LTR` 更突出覆盖改善和曝光集中度下降；`Seq-xQuAD-Tripartite` 的价值则主要体现在把商家公平、ETA、供给和列表级覆盖纳入排序，并需要结合后续履约仿真判断系统级收益。

## 4. 离线推荐结果

{offline_table}

推荐侧指标使用 Recall@K、NDCG@K、MRR@K 和 HitRate@K；RepeatRecall@K 只统计测试真值中用户训练期点过的复购商家，ExploreRecall@K 只统计训练期未点过的新商家，用来拆开“复购命中”和“探索命中”。商家侧指标使用 Coverage、Long-tail Exposure 和 Exposure Gini；校准侧指标使用 CategoryJSD@20，衡量推荐列表品类分布与用户历史品类分布的 Jensen-Shannon divergence，数值越低越贴近用户习惯。这样既能看推荐是否命中真实下单，也能看曝光是否过度集中，以及列表是否偏离用户长期品类偏好。全量 TRD 结果中，Seq-Tuned 或 {learned_ltr} 通常代表离线准确率前沿，说明外卖推荐强烈受复购序列和商家转移影响；Seq-xQuAD-Tripartite 的离线准确性可能低于纯用户侧序列模型，但它把商家曝光和履约约束纳入同一条链路，需要结合仿真指标判断系统级收益。

## 5. 动态履约仿真结果

{simulation_table}

履约侧指标包括完成和未分配订单数、平均/P95 ETA、超时率、骑手负载、活跃骑手比例、骑手收入 Gini 和平台综合效用。多种子运行时，数值列为跨种子均值，并附 `_std` 与 `_ci95` 列。骑手速度、服务时长和初始负载可由外部配送任务 CSV 校准；传入 `--rider-tasks` 时会同步产出校准诊断 JSON，含逐参数来源标注（数据估计 vs 外卖默认）、经验分位数、对数正态拟合与仿真输入分布对任务数据的双样本 KS 检验。没有外部数据时继续使用固定 seed 的合成参数。

为避免只展示单点权重，图表中额外生成 `pareto_recall_utility.png` 和 `tripartite_frontier.csv`，把 Recall@20、Exposure Gini、平均 ETA、超时率和平台效用合并为非支配前沿视角。答辩时可以用这张图说明：Seq-Tuned/{learned_ltr} 代表离线准确率前沿，KG-Tripartite 等三方策略代表系统效用前沿，它们共同构成三方推荐的权衡边界。

## 6. 图表展示

{figures_md}

## 7. 结论与局限

FoodFlow 的答辩故事可以概括为三步：第一，公开外卖订单数据上的推荐实验说明模型不是随机的，并进一步利用 TRD session 点击、SPU 菜品信号和知识图谱兴趣路径；第二，引入 LightGBM-LTR 或其可解释 fallback，让序列特征权重不只依赖单一热度；第三，将推荐结果接入骑手履约仿真，并用批量二分图匹配、Session-SPU/KG 行为增强、多种子置信区间和 LaDe 可校准骑手参数说明准确性、公平性、ETA、超时率和平台效用之间的权衡。

局限是骑手数据默认来自合成仿真；LaDe 可用于校准末端配送速度、任务时长和负载分布，但仍不能替代工业级外卖派单数据。LightGBM-LTR 当前仍使用项目构造的候选集和特征，训练标签来自历史下单集合，后续可以进一步加入更丰富的上下文、真实曝光/点击标签和跨城市验证。
"""
    output.write_text(text, encoding="utf-8")
    return output
