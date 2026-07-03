# FoodFlow：外卖三方推荐与动态履约仿真系统

中山大学《大数据原理》期末工程实践项目。FoodFlow 面向外卖平台推荐场景，基于公开 TRD 外卖订单数据实现用户-商家 Top-K 推荐，并进一步把推荐结果接入订单-骑手匹配与动态履约仿真，用推荐指标和系统级指标共同评价策略效果。

项目核心观点：外卖推荐不能只看 Recall 或 NDCG。推荐列表会改变订单的空间分布，进而影响商家曝光、骑手负载、预计送达时间和超时风险。因此本项目将推荐系统扩展为“用户、商家、骑手”三方闭环。

## 项目亮点

- 用户侧推荐：默认实验包含 Popular、BPR-MF、UserOnly、Seq-Tuned、LightGBM/Logistic-LTR、Seq-xQuAD-Tripartite、Session-SPU-Tripartite 和 KG-Tripartite。
- 商家侧公平：在重排中引入商家曝光公平、长尾曝光与 Exposure Gini 等指标。
- 骑手侧履约：比较最近骑手、最小 ETA、负载感知逐单派单、容量槽位批量最大权匹配，以及**路径感知顺路派单**（cheapest-insertion 边际绕行成本 + 匈牙利轮次匹配，骑手沿真实路径行进）；支持外部任务数据校准参数与运力紧张压力场景（`--n-riders`）。
- 真实城市地理：用户/商家嵌入 LaDe 真实城市配送 GPS 分布（`make geocode`），距离与 ETA 具备真实街区尺度，demo 在真实地图底图上展示配送路径与高峰派单回放。
- 动态仿真：模拟午餐高峰多时间步请求，持续更新骑手状态和订单履约结果。
- 知识图谱融合：`KG-Tripartite` 在三方重排之上叠加时间衰减的图谱兴趣信号（品类/商圈/价位节点的关系加权匹配），并从历史下单、品类、区域和价格段构造路径证据解释推荐原因；完整的动态 KG 注意力模型（torch 实现与 GPU 实验）见 `kg-demo/` 子项目。
- 统计严谨性：`simulate --simulation-seeds` 输出多种子均值与 95% 置信区间；`--rider-tasks` 校准同时产出分布诊断 JSON（对数正态拟合 + KS 检验 + 逐参数来源标注）。
- 多指标评估：同时输出 Recall、NDCG、MRR、HitRate、Repeat/Explore Recall、Coverage、Exposure Gini、Avg ETA、Timeout Rate、Rider Load Std、Platform Utility。
- 工程闭环：提供 Makefile、CLI、测试、图表、实验报告、Streamlit demo 和 NotebookLM PPT 素材包。

## 数据来源

主数据集为 Takeout Recommendation Dataset (TRD)：

- DOI：`10.5281/zenodo.8025855`
- 地址：<https://zenodo.org/records/8025855>
- 来源：美团外卖北京 11 个商圈
- 时间：2021-03-01 至 2021-03-28
- 使用文件：用户、商家、菜品、训练订单、测试订单和测试标签

项目默认下载 TRD 的 txt 文件，不下载约 1.8GB 的 `graph.bin`，因为核心实现不依赖 DGL 图文件。TRD 不包含完整骑手状态和真实派单记录，因此骑手位置、在线状态、负载、可靠性和收入为固定 seed 合成 proxy，仅用于履约仿真，不声称为真实骑手数据。若有 LaDe 等公开末端配送任务 CSV，可通过 `--rider-tasks` 校准骑手速度、服务时长和负载分布。

TRD 也不含坐标。项目通过 `make geocode GEO_TASKS=data/lade/delivery_yt.parquet` 把用户/商家嵌入 LaDe（Cainiao 末端配送数据集，Hugging Face `Cainiao-AI/LaDe-D`）真实城市的配送 GPS 密度分布——商圈映射到真实高密度簇、用户从真实配送点采样，元数据见 `data/processed/geo_note.json`。诚实口径：实体的具体位置仍是合成分配，但空间分布、距离与 ETA 具备真实城市尺度（详见 `docs/ALGORITHM_RIDER_IMPROVEMENTS.md`）。

## 方法概览

FoodFlow 的主流程如下：

```text
TRD 数据下载与清洗
  -> 用户/商家/菜品/订单特征构建
  -> 用户-商家 Top-K 推荐
  -> 三方重排：用户偏好 + 商家公平 + ETA + 供给分
  -> 模拟下单
  -> 订单-骑手匹配
  -> 午餐高峰动态履约仿真
  -> 指标、图表、报告与 demo
```

`Seq-Tuned` 用复购、时间衰减和商家转移概率构造可解释序列排序；`LightGBM-LTR` 使用同一候选和特征训练学习排序模型，不可用时回退到 `Logistic-LTR`。`Seq-xQuAD-Tripartite` 把列表覆盖、商家公平、ETA 和供给约束接入重排，并分别进入逐单和批量派单。`Session-SPU-Tripartite` 仅使用训练期点击会话与菜品画像扩展候选和解释，避免测试期信息泄漏。

三方重排分数采用可解释加权：

```text
score = 0.62 * user_score
      + 0.18 * merchant_fairness
      + 0.14 * eta_score
      + 0.06 * supply_score
```

其中 `user_score` 表示用户偏好，`merchant_fairness` 表示商家曝光补偿，`eta_score` 表示预计履约时间，`supply_score` 表示商家供给可行性。

## 项目结构

```text
foodflow/
  cli.py             # 命令行入口
  download.py        # TRD 下载
  preprocess.py      # 数据清洗与特征处理
  data.py            # 处理后数据加载
  recommenders.py    # 推荐基线、序列模型与三方重排
  rerank.py          # ETA、公平分、供给分等重排特征
  kg.py              # 轻量知识图谱三元组与路径解释
  rider_sim.py       # 骑手生成、ETA 估计、订单匹配
  simulator.py       # 动态履约仿真
  metrics.py         # 推荐与公平指标
  evaluate.py        # 离线推荐评估
  figures.py         # 图表生成
  report.py          # 实验报告生成
  explain.py         # 推荐解释

app.py               # Streamlit demo
Makefile             # 常用运行命令
outputs/             # 指标 CSV 与图表
report/              # 实验报告
ppt/notebooklm/      # NotebookLM PPT 生成素材包
tests/               # 单元测试与集成 smoke 测试
kg-demo/             # 动态知识图谱注意力推荐子项目（torch 训练版 + 网页 demo + GPU 实验结果）
```

## 快速开始

推荐使用 conda 环境：

```bash
git clone git@github.com:HSZGB/FoodFlow.git
cd FoodFlow
make conda-setup
conda activate foodflow
```

如果不使用 conda，也可以运行 `make setup` 创建 `.venv`。不要直接用系统 `python3 -m foodflow.cli` 跑完整流程，因为系统环境可能缺少 pandas、matplotlib 等依赖。

### 1. 快速 smoke 测试

使用 mock 数据完整跑通流程：

```bash
make smoke
```

该命令会执行：

```text
mock-data -> preprocess -> eval-offline -> simulate -> figures -> report -> test
```

### 2. 真实 TRD 数据实验

快速复现实验默认抽样 5 万训练订单：

```bash
make download
make preprocess
make eval
make simulate
make audit
make figures
make report
```

也可以一行运行：

```bash
make download preprocess eval simulate audit figures report
```

`make simulate` 会按策略输出进度和耗时，例如当前跑到第几个策略、完成订单数、平均 ETA、超时率和平台效用；如果只想输出最终表格，可使用：

```bash
python -m foodflow.cli simulate --quiet
```

如果有 LaDe 末端配送任务数据，可校准骑手仿真参数：

```bash
make simulate-calibrated RIDER_TASKS=/path/to/lade_delivery.csv
```

`--rider-tasks` 同时支持项目通用字段 `courier_id,accept_time,finish_time,pickup_lng,pickup_lat,delivery_lng,delivery_lat`，LaDe 分城市 delivery 字段 `courier_id,accept_time,accept_gps_lng,accept_gps_lat,delivery_time,delivery_gps_lng,delivery_gps_lat`，以及 Hugging Face 合并版 `delivery_five_cities.csv` 中的 `delivery_user_id,receipt_time,receipt_lng,receipt_lat,sign_time,poi_lng,poi_lat`。合并版坐标为投影坐标，项目会使用欧氏距离估计任务距离。LaDe 不是外卖平台数据，因此只用于骑手速度、任务时长、任务重叠负载和可靠性 proxy 的估计，不参与用户-商家推荐训练。

如果使用 LaDe 做展示，推荐同时保留两种口径：默认 `raw` 口径直接使用末端包裹配送任务估计值，适合说明跨领域压力测试；`food-scaled` 口径保留 LaDe 的负载和可靠性 proxy，但把速度和服务时长重标定到外卖 SLA 尺度，适合展示策略差异：

```bash
make simulate-calibrated RIDER_TASKS=/path/to/lade_delivery.csv RIDER_PROFILE=food-scaled

python -m foodflow.cli simulate \
  --processed-dir data/processed \
  --output outputs/results/simulation_metrics_lade_food_scaled.csv \
  --seed 42 \
  --rider-tasks /path/to/lade_delivery.csv \
  --rider-calibration-profile food-scaled
```

如果要使用完整 TRD 训练集：

```bash
make download
make preprocess-full
make eval
make simulate
make audit
make figures
make report
```

### 3. 运行测试

```bash
make test
```

### 4. 启动 demo

```bash
make demo
```

`make demo` 默认使用 conda 环境中的 Streamlit。这个命令启动的是 Web 服务，所以正常情况下不会自动结束；看到 Streamlit 地址后，保持终端开着，在浏览器访问页面即可。停止服务时在该终端按 `Ctrl+C`。
如果终端长时间停在 Streamlit 日志处，没有回到命令行，这通常代表服务正在运行，不是卡死。

默认地址：

```text
http://localhost:8501
```

首次打开页面时，系统需要加载处理后的数据并拟合当前选中的推荐模型；之后会被 Streamlit 缓存，切换用户会快很多。为保证页面响应速度，交互推荐模型默认使用约 1.2 万条训练订单，并保留当前演示用户历史；完整实验指标、报告和图表仍来自全量 TRD 输出。主线策略对比和高峰回放默认不预计算，需要时在页面里勾选。

如果想让 demo 交互模型也强制使用完整训练订单：

```bash
make demo-full
```

也可以手动指定交互训练订单上限，例如：

```bash
FOODFLOW_DEMO_MAX_ORDERS=60000 make demo
```

如果希望空间图里骑手更多或更少，也可以调整 demo 合成骑手规模，默认是 1200：

```bash
FOODFLOW_DEMO_RIDERS=1000 make demo
```

如果需要临时改用其它 Streamlit，可覆盖变量，例如：

```bash
make STREAMLIT=".venv/bin/streamlit" demo
```

如果需要指定端口：

```bash
make STREAMLIT_FLAGS="--server.port 8502" demo
```

如果页面迟迟没有变化，或者仍然看到旧版裸 HTML，先检查是否有旧的 Streamlit 进程占着端口：

```bash
make demo-check
```

demo 默认展示用户 `8`，也提供复购活跃型、高消费型、价格敏感型等快速案例；侧边栏可手动输入其它用户 ID，并在 `UserOnly`、`Seq-Tuned`、`LightGBM-LTR`、`Seq-xQuAD-Tripartite` 等主线策略之间切换。当前界面包含推荐商家、用户品类偏好、主线策略对比、分数组成、骑手候选榜、当前订单路径、午餐高峰回放、方法与指标、实验结果。
当前 demo 首屏会先给出用户、策略、Top1 商家、平均 ETA 和在线骑手数。空间图聚焦当前订单附近，展示附近用户样本、附近商家样本、近场骑手、Top 派单候选、两个 2.5km 参考圈、取餐段和配送段；这两个圆只是距离参考，不是等高线、热力图或真实配送边界。高峰回放默认 16 个时间步，也可以在页面里调长或调短。

## 主要输出

```text
outputs/results/offline_metrics.csv       # 离线推荐指标
outputs/results/simulation_metrics.csv    # 动态履约仿真指标
outputs/results/tripartite_frontier.csv   # 推荐准确性、曝光公平和履约效用的 Pareto 前沿表
outputs/results/data_audit.json           # TRD 完整性与抽样审计
outputs/figures/*.png                     # 实验图表
docs/DATA_AUDIT.md                        # 数据审计 Markdown
report/实验报告.md                        # 实验报告
ppt/notebooklm/upload_pack/               # NotebookLM PPT 上传素材包
```

重新生成 NotebookLM 上传包：

```bash
make notebooklm-pack
```

## 实验结果摘要

当前提交的结果已经使用完整 TRD `orders_train.txt`，数据审计显示原始训练订单 `1,068,495` 条，处理后训练订单 `1,068,495` 条，训练订单使用比例 `1.0000`。

离线推荐中，`Seq-Tuned` 的 Recall@10 最高，为 `0.4187`（NDCG@10 `0.3504`，HitRate@10 `0.5800`），代表纯用户侧准确率前沿。三方系列中 `KG-Tripartite` 表现最好：Recall@10 `0.4048`、NDCG@10 `0.3412`、HitRate@10 `0.5633`，在保留商家公平、ETA 和供给分量的同时接近纯精度导向的序列模型。

动态履约仿真在真实城市地理（LaDe 烟台核心区）上采用 10 个随机种子重复运行，报告均值与 95% 置信区间（完整 `_std`/`_ci95` 列见 `outputs/results/simulation_metrics.csv`）。**常规运力场景（120 骑手）**下，三方重排 + 批量槽位匹配把平均 ETA 压到约 36 分钟、超时率约 20%：

| 策略 | Avg ETA (±CI95) | Timeout Rate (±CI95) | Platform Utility (±CI95) |
|---|---:|---:|---:|
| Popular + Nearest | 59.06 ±2.28 | 0.579 ±0.023 | 0.391 ±0.007 |
| Seq-Tuned + MinETA | 40.44 ±1.21 | 0.287 ±0.041 | 0.533 ±0.012 |
| Seq-xQuAD-Tripartite + Batch | 35.90 ±1.88 | 0.195 ±0.036 | 0.553 ±0.009 |
| Session-SPU-Tripartite + Batch | 36.79 ±1.25 | 0.199 ±0.029 | 0.557 ±0.010 |
| KG-Tripartite + Batch | 36.90 ±1.04 | 0.211 ±0.033 | 0.550 ±0.011 |
| KG-Tripartite + RouteMinETA | 39.04 ±1.18 | 0.316 ±0.041 | 0.516 ±0.012 |
| KG-Tripartite + RouteBatch | 41.67 ±1.27 | 0.382 ±0.044 | 0.496 ±0.014 |

**运力紧张压力场景（30 骑手，模拟高峰爆单，`make simulate-stress`）**下，路径感知顺路派单反超所有槽位匹配策略——顺路接单率达 65.5%，平均 ETA 比最好的槽位匹配低约 11.5 分钟（远超置信区间）：

| 策略 | Avg ETA (±CI95) | Timeout Rate (±CI95) | Platform Utility (±CI95) |
|---|---:|---:|---:|
| Seq-Tuned + MinETA | 76.64 ±2.46 | 0.819 ±0.014 | 0.442 ±0.014 |
| Seq-xQuAD-Tripartite + Batch | 60.96 ±4.84 | 0.679 ±0.046 | 0.486 ±0.017 |
| KG-Tripartite + Batch | 61.47 ±2.85 | 0.685 ±0.039 | 0.481 ±0.014 |
| KG-Tripartite + RouteMinETA | **49.43 ±1.68** | **0.552 ±0.040** | **0.529 ±0.013** |
| KG-Tripartite + RouteBatch | 51.94 ±1.86 | 0.589 ±0.033 | 0.519 ±0.013 |

结论：`Seq-Tuned` 取得最强离线准确性；三方重排 + 批量匹配在常规运力下系统性改善履约；**顺路派单是高峰运力吃紧时的韧性机制**——运力充足时分散派单更快（route 家族的 ETA 诚实计入队列行进时间），运力紧张时顺路合单大幅降低 ETA 与超时率。派单机制应随供需状态切换，这是三方联合视角的核心论据。

## 图表示例

![Offline Recall@20](outputs/figures/offline_recall20.png)

![Simulation Platform Utility](outputs/figures/simulation_platform_utility.png)

![Tripartite Scorecard](outputs/figures/tripartite_scorecard.png)

## 局限与说明

- 骑手数据默认为合成 proxy；如提供 LaDe delivery 数据，可校准速度、服务时长和负载分布并提供真实城市地理，但 LaDe 仍不能替代真实外卖派单数据。
- 顺路派单不是在所有场景下更优：运力充足时槽位批量匹配的 ETA 更好，顺路合单的优势出现在运力紧张的高峰场景（见 `simulation_metrics_peak_stress.csv`），报告中按两个场景分别呈现。
- BPR-MF 是轻量实现，未追求大规模深度模型最优性能。
- 主管线的 `KG-Tripartite` 是免训练的图谱兴趣近似（时间衰减 + 关系加权匹配）；完整的动态 KG 注意力模型、LightGCN 等训练版实现与 GPU 实验结果在 `kg-demo/` 子项目中（`kg-demo/src/`、`kg-demo/outputs/`），两者共用 TRD 数据、叙事上互为轻量/深度版本。
- 平台效用权重是课程项目中的解释性设置，后续可做权重敏感性分析。
- 当前三方优化采用“推荐重排 + 下单后骑手匹配”的解耦闭环，不是工业级端到端联合调度系统。

## 课程交付材料

- 作业要求：`作业要求.md`
- 硬性要求：`硬性要求.md`
- 项目计划：`PROJECT_PLAN.md`
- 数据来源说明：`docs/DATA_SOURCE.md`
- 交付验收审计：`docs/DELIVERY_AUDIT.md`
- 调研路线覆盖审计：`docs/REPORT_ROUTE_COVERAGE.md`
- 权重敏感性分析：`docs/WEIGHT_SENSITIVITY.md`
- Seq-Tuned 权重搜索：`docs/SEQ_TUNED_SEARCH.md`
- Tuned 三方消融：`docs/TRIPARTITE_TUNED_ABLATION.md`
- 实验报告：`report/实验报告.md`
- PPT 生成材料：`ppt/notebooklm/`
