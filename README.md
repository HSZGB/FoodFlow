# FoodFlow：外卖三方推荐与动态履约仿真系统

中山大学《大数据原理》期末工程实践项目。FoodFlow 面向外卖平台推荐场景，基于公开 TRD 外卖订单数据实现用户-商家 Top-K 推荐，并进一步把推荐结果接入订单-骑手匹配与动态履约仿真，用推荐指标和系统级指标共同评价策略效果。

项目核心观点：外卖推荐不能只看 Recall 或 NDCG。推荐列表会改变订单的空间分布，进而影响商家曝光、骑手负载、预计送达时间和超时风险。因此本项目将推荐系统扩展为“用户、商家、骑手”三方闭环。

## 项目亮点

- 用户侧推荐：默认实验聚焦 Popular、BPR-MF、UserOnly、Seq-Tuned、LightGBM-LTR、Seq-xQuAD-Tripartite 六类代表性策略。
- 商家侧公平：在重排中引入商家曝光公平、长尾曝光与 Exposure Gini 等指标。
- 骑手侧履约：模拟订单-骑手匹配，比较最近骑手、最小 ETA、负载感知逐单派单和批量最大权匹配。
- 动态仿真：模拟午餐高峰多时间步请求，持续更新骑手状态和订单履约结果。
- 轻量 KG 解释：从历史下单、品类、区域和价格段构造路径证据，解释推荐原因。
- 多指标评估：同时输出 Recall、NDCG、MRR、HitRate、Coverage、Exposure Gini、Avg ETA、Timeout Rate、Rider Load Std、Platform Utility。
- 工程闭环：提供 Makefile、CLI、测试、图表、实验报告、Streamlit demo 和 NotebookLM PPT 素材包。

## 数据来源

主数据集为 Takeout Recommendation Dataset (TRD)：

- DOI：`10.5281/zenodo.8025855`
- 地址：<https://zenodo.org/records/8025855>
- 来源：美团外卖北京 11 个商圈
- 时间：2021-03-01 至 2021-03-28
- 使用文件：用户、商家、菜品、训练订单、测试订单和测试标签

项目默认下载 TRD 的 txt 文件，不下载约 1.8GB 的 `graph.bin`，因为核心实现不依赖 DGL 图文件。TRD 不包含完整骑手状态和真实派单记录，因此骑手位置、在线状态、负载、可靠性和收入为固定 seed 合成 proxy，仅用于履约仿真，不声称为真实骑手数据。

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

`Seq-Tuned` 借鉴 FPMC/下一篮子推荐和会话推荐思想，把外卖复购、最近订单时间衰减和商家转移概率加入排序，是当前离线准确率最强策略。`LightGBM-LTR` 使用同一组候选与序列特征训练学习排序模型。`Seq-xQuAD-Tripartite` 则把列表级覆盖、商家公平、ETA 和供给约束接到同一个重排器中，并进入逐单贪心与批量匹配两条仿真链路。

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
python3 scripts/prepare_notebooklm_pack.py
```

## 实验结果摘要

当前提交的结果已经使用完整 TRD `orders_train.txt`，数据审计显示原始训练订单 `1,068,495` 条，处理后训练订单 `1,068,495` 条，训练订单使用比例 `1.0000`。

离线推荐中，`Seq-Tuned` 的 Recall@20 最高，为 `0.4675`，NDCG@20 为 `0.3652`，HitRate@20 为 `0.6267`。`LightGBM-LTR` 的 Recall@20 为 `0.4424`，并把 Coverage@20 提升到 `0.4345`、Exposure Gini 降到 `0.7942`。`Seq-xQuAD-Tripartite` 的离线准确性低于纯用户侧序列模型，但能进入后续履约链路做系统级比较。

动态履约仿真中，`Seq-xQuAD-Tripartite + Greedy` 取得最低平均 ETA、最低超时率和最高平台综合效用；批量匹配版本完成订单更多，但 ETA 和平台效用相对逐单贪心有取舍：

| 策略 | Avg ETA | Timeout Rate | Platform Utility |
|---|---:|---:|---:|
| Popular + Nearest | 86.64 | 0.8537 | 0.3098 |
| UserOnly + MinETA | 53.40 | 0.7113 | 0.4240 |
| Seq-Tuned + MinETA | 56.04 | 0.7097 | 0.4147 |
| LightGBM-LTR + MinETA | 54.81 | 0.7419 | 0.3922 |
| Seq-xQuAD-Tripartite + Greedy | 46.87 | 0.5200 | 0.4767 |
| Seq-xQuAD-Tripartite | 48.98 | 0.5543 | 0.4662 |

结论不是“单一模型在所有指标上最优”，而是：`Seq-Tuned` 取得最强离线推荐准确性；`LightGBM-LTR` 带来更高覆盖和更低曝光集中度；`Seq-xQuAD-Tripartite` 牺牲一部分离线准确性，把 ETA、超时率、订单吞吐和平台效用纳入同一套系统级权衡。

## 图表示例

![Offline Recall@20](outputs/figures/offline_recall20.png)

![Simulation Platform Utility](outputs/figures/simulation_platform_utility.png)

![Tripartite Scorecard](outputs/figures/tripartite_scorecard.png)

## 局限与说明

- 骑手数据为合成 proxy，不能替代真实派单数据。
- BPR-MF 是轻量实现，未追求大规模深度模型最优性能。
- 项目实现的是轻量 KG 路径解释，没有实现 LightGCN、KGAT 或完整知识图谱训练模型，这些作为调研背景和后续增强方向。
- 平台效用权重是课程项目中的解释性设置，后续可做权重敏感性分析。
- 当前三方优化采用“推荐重排 + 下单后骑手匹配”的解耦闭环，不是工业级端到端联合调度系统。

## 课程交付材料

- 作业要求：`作业要求.md`
- 硬性要求：`硬性要求.md`
- 项目计划：`PROJECT_PLAN.md`
- 数据来源说明：`docs/DATA_SOURCE.md`
- 权重敏感性分析：`docs/WEIGHT_SENSITIVITY.md`
- Seq-Tuned 权重搜索：`docs/SEQ_TUNED_SEARCH.md`
- Tuned 三方消融：`docs/TRIPARTITE_TUNED_ABLATION.md`
- 实验报告：`report/实验报告.md`
- PPT 生成材料：`ppt/notebooklm/`
