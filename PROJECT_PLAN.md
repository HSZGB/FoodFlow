# FoodFlow 外卖三方推荐与动态履约仿真实施计划

## 当前状态

- [x] 选定主线：TRD 真实外卖推荐数据 + 合成骑手 + 动态履约仿真。
- [x] 评分约束：必须展示指标、说明数据集来源、讲清故事、做 PPT。
- [x] 工程骨架、数据管道、算法、仿真、报告和 PPT 大纲已实现。
- [x] 真实 TRD 必需 txt 文件已下载并通过 md5 校验；正式指标和报告已由完整 TRD `orders_train.txt` 生成。
- [x] `make smoke` 已改为隔离目录，不会覆盖真实 TRD raw/processed/outputs。
- [x] 新增 conda 环境、数据审计、Ours-Balanced 权重敏感策略和三方可视化看板。
- [x] 新增 Seq-Tuned 与 Seq-Tuned-xQuAD，当前离线 Recall@20 提升到 `0.4675`，并已重生成指标、图表、报告和 NotebookLM 上传包。
- [x] 新增可复现的 `scripts/search_seq_weights.py` 和 `docs/SEQ_TUNED_SEARCH.md`，用于说明 Seq-Tuned 权重来源。
- [x] 完成 Tuned 序列底座接入三方重排的消融记录，确认默认系统效用前沿仍为 `Seq-xQuAD-Tripartite`。
- [x] `simulate` CLI 增加策略级进度日志和耗时统计，长任务不再无反馈。
- [ ] PPT 最终图片页和 `.pptx` 等待 `codex-ppt` 审批门禁。

## 重复执行流程

每次开工：

1. 运行 `git status --short --branch`。
2. 阅读本文件的“阶段清单”和“验收命令”。
3. 只推进一个明确阶段，结束前跑对应命令。

每次收工：

1. 运行对应测试或 smoke 命令。
2. 检查输出文件和指标表是否存在且非空。
3. 更新本文件状态。
4. `git add` + `git commit` 记录里程碑。

失败两次时：

1. 记录失败信号和已尝试方案。
2. 切换不同方案，不继续做同一方向微调。
3. 用命令验证新方案是否有效。

## 阶段清单

### 阶段 1：初始化工程

- [x] `git init`
- [x] `.gitignore`
- [x] `requirements.txt`
- [x] `Makefile`
- [x] Python package skeleton
- 验收命令：`git status --short --branch`

### 阶段 2：数据闭环

- [x] TRD 下载脚本，默认跳过 `graph.bin`
- [x] 下载校验使用 Zenodo md5，防止 mock/真实文件错误续传污染。
- [x] mock 数据生成脚本
- [x] 预处理脚本
- [x] 数据字典与来源说明
- 验收命令：`make mock preprocess`

### 阶段 3：推荐与指标

- [x] Random、Popular、Repeat、ItemCF、BPR-MF、UserOnly、Ours-Full
- [x] Ours-Balanced，用于展示准确性、商家公平和履约约束的权衡路径
- [x] Recall@K、NDCG@K、MRR@K、HitRate@K
- [x] Coverage、Long-tail Exposure、Exposure Gini
- [x] Seq-Hybrid、Seq-xQuAD、Seq-Tuned、Seq-Tuned-xQuAD、Seq-Tripartite、Seq-xQuAD-Tripartite，用于展示序列复购、列表重排、校准和三方约束。
- 验收命令：`make eval`

### 阶段 4：三方仿真

- [x] 合成骑手状态
- [x] ETA、最近骑手、最小 ETA、负载感知匹配
- [x] 午餐高峰多时间步仿真
- [x] Avg ETA、Timeout Rate、Rider Load Std、Platform Utility
- 验收命令：`make simulate`

### 阶段 5：图表、报告、Demo

- [x] 指标图表
- [x] trade-off 图、三方 scorecard 与 Streamlit 三页看板
- [x] 实验报告 Markdown
- [x] Streamlit demo
- [x] 方法看板、空间供需调度图、NotebookLM 上传包已同步最新指标。
- 验收命令：`make figures report`

### 阶段 6：PPT

- [x] 按 `codex-ppt` 规则建立 `ppt/FoodFlow/outline.md` 草稿。
- [ ] 等待大纲审批。
- [ ] 确认视觉风格。
- [ ] 确认图片生成后端。
- [ ] 生成并审批样张。
- [ ] 生成全套图片页、讲稿和 PPTX。

注意：PPT skill 明确要求审批门禁。未获得大纲、风格、后端和样张确认前，不创建最终 `deck_spec.json`、`speech.md`、slide images 或 `.pptx`。

### 阶段 7：总验收

- [x] `make smoke` 通过。
- [x] `make preprocess-full eval simulate audit figures report` 已在完整 TRD 必需文件上通过。
- [x] `make conda-setup` 和 `make conda-test` 已通过。
- [x] 至少 5 种推荐策略有离线指标。
- [x] 至少 4 种三方策略有仿真指标。
- [ ] 报告包含数据来源、指标表、对比图和结果分析；PPT 最终产物等待审批后生成。

## 设计决策

- 主数据源：Takeout Recommendation Dataset (TRD), Zenodo DOI `10.5281/zenodo.8025855`。
- 不下载 `graph.bin`：体积约 1.8GB，本项目不依赖 DGL 图。
- 骑手数据：合成仿真，不伪装成真实数据；报告中说明生成规则。
- 模型边界：CPU 可运行，BPR-MF 为轻量 NumPy 实现；LightGCN/KGAT 作为调研背景，不作为核心代码验收。
