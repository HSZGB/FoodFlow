# FoodFlow NotebookLM 上传素材清单

NotebookLM 上传时建议保留原文件名。图表页要优先使用 PNG 作为证据图，不要让工具重新编造数值。

## 必传提示词

| 文件 | 用途 |
|---|---|
| `prompts/source_pack.md` | 项目事实汇总，NotebookLM 的主要知识源 |
| `prompts/notebooklm_master_prompt.md` | 一次性生成 11 页 PPT 的总提示词 |
| `prompts/slide_by_slide_prompt.md` | 逐页生成、修订或补救时使用 |
| `prompts/assets_manifest.md` | 让 NotebookLM 知道每张图应该放在哪页 |
| `prompts/quality_checklist.md` | 生成后检查 |

## 必传项目来源

| 文件 | 用途 |
|---|---|
| `sources/实验报告.md` | 项目目标、数据来源、算法、指标、结果和结论 |
| `sources/DATA_SOURCE.md` | TRD 数据来源、Zenodo DOI、合成骑手边界 |
| `sources/ppt_outline.md` | 已确认的 11 页答辩结构 |
| `sources/作业要求.md` | 课程评分与展示要求 |
| `sources/硬性要求.md` | 指标展示、数据来源、讲故事、PPT 等硬性要求 |

## 指标 CSV

| 文件 | 建议用途 |
|---|---|
| `results/offline_metrics.csv` | 第 6 页离线推荐指标；必要时生成小表格 |
| `results/simulation_metrics.csv` | 第 8 页动态履约仿真指标；必要时生成小表格 |
| `results/tripartite_frontier.csv` | 第 8 或第 10 页 Pareto 前沿；必要时生成小表格 |

## 图表素材

| 图表 | 推荐页 | 使用说明 |
|---|---:|---|
| `figures/offline_recall20.png` | Slide 6 | 主图，展示 Recall@20 对比 |
| `figures/offline_ndcg20.png` | Slide 6 | 可作为补充图，展示 NDCG@20 |
| `figures/offline_coverage20.png` | Slide 6 | 可作为商家覆盖率证据 |
| `figures/offline_exposure_gini.png` | Slide 6 | 可作为曝光集中度证据 |
| `figures/offline_category_jsd20.png` | Slide 6 | 可作为品类校准证据，越低越好 |
| `figures/tradeoff_ndcg_gini.png` | Slide 6 | 推荐准确性与曝光公平 trade-off |
| `figures/tradeoff_recall_coverage.png` | Slide 6 | 推荐召回与商家覆盖率 trade-off |
| `figures/simulation_avg_eta.png` | Slide 8 | 主图，展示平均 ETA |
| `figures/simulation_timeout_rate.png` | Slide 8 | 展示超时率 |
| `figures/simulation_rider_load_std.png` | Slide 8 | 展示骑手负载均衡 |
| `figures/simulation_platform_utility.png` | Slide 8 | 主图，展示平台综合效用 |
| `figures/simulation_exposure_gini.png` | Slide 8 | 展示仿真中的商家曝光集中度 |
| `figures/tradeoff_eta_utility.png` | Slide 8 | 展示 ETA 与平台效用权衡 |
| `figures/pareto_recall_utility.png` | Slide 8 | 展示 Recall 与平台效用的 Pareto 前沿 |
| `figures/tripartite_scorecard.png` | Slide 10 | 用户、商家、平台三方指标总览 |

## 图片使用规则

- 第 6 页至少使用 `offline_recall20.png` 和 `tradeoff_ndcg_gini.png`。
- 如果第 6 页空间允许，加入 `offline_category_jsd20.png`，说明 Seq-xQuAD 兼顾命中与用户历史品类校准。
- 第 8 页至少使用 `simulation_avg_eta.png` 和 `simulation_platform_utility.png`。
- 如果第 8 页空间允许，优先加入 `pareto_recall_utility.png`，说明 Seq-Tuned / Seq-Tuned-xQuAD 是离线准确率前沿、Seq-xQuAD-Tripartite 是平台效用前沿。
- 第 10 页可以使用 `tripartite_scorecard.png` 总结三方贡献。
- 任何图表都必须保留原始坐标轴、图例、模型名称和数值含义。
- 如果 NotebookLM 重新绘制图表，必须以 CSV 数值为准，并在讲稿中说明图表来自本项目实验输出。
