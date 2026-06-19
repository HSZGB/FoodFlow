# 最新版调研报告路线覆盖审计

审计日期：2026-06-17

来源：`FoodFlow_最新版调研报告_融合代码核查与优化路线.docx` 抽取的 P0-P6 优化路线。

## 总体结论

当前仓库已经按“报告B工程闭环为主、报告A轻量增强为辅”的路线完成核心交付：TRD 数据处理、默认六模型离线评估、Repeat/Explore 分段指标、六条动态履约仿真、LightGBM/Logistic 学习排序入口、MNL 用户选择、批量骑手匹配、轻量 KG 解释、三方分量归一化、Pareto 前沿、报告和 NotebookLM PPT 素材包。

仍需明确边界：完整 LightGCN/KGAT/KGCN 不是当前主线实现；三方权重网格搜索保留为历史/后续分析，不作为正式指标结论；最终图片页 PPTX 受 `codex-ppt` 审批门禁约束，尚未生成。

## P0-P6 覆盖表

| 阶段 | 调研报告目标 | 当前状态 | 证据 |
|---|---|---|---|
| P0 | 修复依赖和文档一致性，统一策略、结果、图表和报告口径 | 已完成 | `requirements.txt`、`environment.yml` 均包含 `lightgbm==4.5.0`；`outputs/results/*.csv`、`report/实验报告.md`、`ppt/notebooklm/upload_pack/` 已同步六模型/六链路口径；离线表新增 RepeatRecall/ExploreRecall |
| P1 | 新增批量最大权骑手匹配，并与逐单贪心对比 | 已完成 | `foodflow/rider_sim.py` 中批量匹配；`outputs/results/simulation_metrics.csv` 同时包含 `Seq-xQuAD-Tripartite + Greedy` 和 batch 版 `Seq-xQuAD-Tripartite` |
| P2 | 新增 softmax/MNL 用户选择模型 | 已完成 | `foodflow/simulator.py` 的 `choice_model=mnl_softmax`；仿真结果表包含 `choice_model` 列 |
| P3 | 学习排序进入默认评估 | 已完成 | `foodflow/recommenders.py` 的 `build_learned_ltr_recommender()`；`tests/test_pipeline.py` 覆盖默认学习排序名称；正式结果含 `LightGBM-LTR` |
| P4 | 轻量 KG 特征与路径解释 | 已完成 | `foodflow/kg.py`；`foodflow/demo_support.py` 中 KG 路径理由；`tests/test_kg.py` |
| P5 | 多目标归一化、Pareto 前沿、权重敏感性 | 部分完成 | 三方分量候选内 min-max 归一化已在 `foodflow/recommenders.py`；`outputs/results/tripartite_frontier.csv` 和 `outputs/figures/pareto_recall_utility.png` 已生成；权重网格搜索保留为历史/后续分析 |
| P6 | LightGCN/KGAT/KGCN 增强 | 展望 | `README.md`、`docs/IMPROVEMENT_RESEARCH.md` 和报告均明确完整图模型不作为核心验收路径 |

## 当前正式主线

离线推荐默认六模型：

- `Popular`
- `BPR-MF`
- `UserOnly`
- `LightGBM-LTR`，缺少 LightGBM 时显式显示为 `Logistic-LTR`
- `Seq-Tuned`
- `Seq-xQuAD-Tripartite`

动态履约默认六链路：

- `Popular + Nearest`
- `UserOnly + MinETA`
- `Seq-Tuned + MinETA`
- `LightGBM-LTR + MinETA`
- `Seq-xQuAD-Tripartite + Greedy`
- `Seq-xQuAD-Tripartite`

## 仍需人工确认的门禁

`codex-ppt` 正式 PPT 生成需要按顺序批准：

1. `ppt/FoodFlow/outline.md` 11 页大纲。
2. 统一视觉风格。
3. 图片生成后端。
4. 1 页样张。
5. 全套图片页、讲稿和 `.pptx`。

在门禁完成前，仓库只保留大纲、审批摘要和 NotebookLM 上传包，不生成最终 `deck_spec.json`、`speech.md`、slide images 或 `.pptx`。
