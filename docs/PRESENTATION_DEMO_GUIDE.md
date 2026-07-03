# FoodFlow 明日展示材料整理

这份文档用于快速制作答辩 PPT，也可以作为现场 demo 的讲稿提纲。主线建议围绕一句话展开：

> 外卖推荐不是只把商家排给用户；推荐会改变订单空间分布，进而影响商家曝光、骑手负载、ETA 和超时风险。因此 FoodFlow 把用户、商家、骑手放进同一个推荐-履约闭环里评估。

## 1. 展示结论先行

PPT 开场可以先给出 3 个结论，后面逐页证明：

1. 推荐侧有效：在真实 TRD 全量训练订单上，`Seq-Tuned` 的 Recall@10 达到 `0.4187`；三方系列中 `KG-Tripartite`（叠加知识图谱兴趣信号）以 Recall@10 `0.4048` 接近纯精度模型，同时保留公平与履约分量。
2. 三方重排有效（10 种子 ±95% CI）：常规运力下 `Seq-xQuAD/Session-SPU/KG-Tripartite + Batch` 把平均 ETA 压到约 36 分钟、超时率约 20%，显著优于纯用户侧策略（约 40.4 分钟 / 29%）。
3. 顺路派单是高峰韧性机制：运力紧张场景（30 骑手）下 `KG-Tripartite + RouteMinETA` 平均 ETA `44.9` 分钟，比最好的槽位匹配低约 16 分钟、超时率 `0.46` vs `0.68`，顺路接单率 `66.9%`；常规运力下也与最优槽位匹配统计打平（36.2 vs 35.9，CI 重叠）。
4. LaDe 被双重真实使用：真实城市地理（烟台核心区配送 GPS 密度，`make geocode`）+ 骑手参数校准（校准诊断 JSON 含参数来源标注与 KS 检验）。

现场建议讲法：

> 推荐准确性是基础，但外卖平台还要在推荐后完成履约。我们的实验说明，纯用户侧最优不一定带来平台侧最优；加入商家公平、ETA 和骑手负载后，可以用小幅准确性代价换取更好的系统指标。

## 2. 数据与边界

### TRD 外卖推荐数据

主推荐数据来自 Takeout Recommendation Dataset (TRD)，Zenodo DOI `10.5281/zenodo.8025855`。当前处理结果来自真实 TRD 全量训练集，不是 mock 数据：

| 文件/对象 | 数量 |
|---|---:|
| 用户 `users.txt` | 200,000 |
| 商家 `pois.txt` | 29,072 |
| 菜品 `spus.txt` | 179,778 |
| 训练订单 `orders_train.txt` | 1,068,495 |
| 测试订单 `orders_test_poi.txt` | 230,550 |
| 测试标签 `orders_poi_test_label.txt` | 230,550 |
| Session 点击 `orders_poi_session.txt` | 230,550 |
| 训练订单菜品明细 `orders_spu_train.txt` | 3,445,180 |

可说的边界：

- 推荐侧使用真实 TRD 外卖订单、点击和菜品信号。
- TRD 不包含完整骑手状态和真实派单记录，所以默认骑手是固定 seed 的合成 proxy。
- `graph.bin` 没有下载，因为本项目没有依赖 DGL 图文件。

### LaDe 末端配送数据

LaDe 是公开工业末端配送数据。论文/数据说明中给出的量级是约 6 个月、10,677k 包裹、21k 配送员，并包含 task-accept、task-finish 一类时空事件。当前运行使用 Hugging Face `Cainiao-AI/LaDe-D` 的烟台城市文件（同时用于真实地理与骑手校准）：

```text
data/lade/delivery_yt.parquet   # 206,431 条真实配送任务，WGS84 坐标
```

实际文件统计：

| 字段 | 值 |
|---|---:|
| 原始行数 | 472,419 |
| 有效校准行数 | 469,327 |
| 配送员数 | 1,870 |
| 杭州市任务 | 185,589 |
| 上海市任务 | 165,583 |
| 重庆市任务 | 121,247 |

字段映射：

| LaDe 合并版字段 | 项目统一字段 |
|---|---|
| `delivery_user_id` | `courier_id` |
| `receipt_time` | `accept_time` |
| `sign_time` | `finish_time` |
| `receipt_lng`, `receipt_lat` | `pickup_lng`, `pickup_lat` |
| `poi_lng`, `poi_lat` | `delivery_lng`, `delivery_lat` |

注意：`delivery_five_cities.csv` 里的经纬度是投影坐标，不是 WGS84 经纬度。代码检测到坐标绝对值大于 360 后，会用欧氏距离除以 1000 估计公里数，而不是用 haversine。

## 3. 系统设计讲法

推荐用下面这个流程图作为架构页：

```text
TRD 原始 txt
  -> 数据清洗与画像构建
  -> 用户-商家 Top-K 推荐
  -> 三方重排：用户偏好 + 商家公平 + ETA + 供给稳定
  -> 模拟下单
  -> 骑手候选与派单策略
  -> 午餐高峰动态履约仿真
  -> 推荐指标 + 商家公平指标 + 骑手履约指标
```

算法模块可以这样拆：

| 模块 | 作用 | 展示重点 |
|---|---|---|
| `Seq-Tuned` | 复购、最近订单、商家转移、品类偏好加权 | 离线准确性最强 |
| `Seq-xQuAD-Tripartite` | 在序列推荐基础上做列表级覆盖与三方重排 | 平衡用户、商家、骑手 |
| `Session-SPU-Tripartite` | 额外使用下单前点击商家和菜品 SPU 类目偏好 | 说明 optional 数据也被利用 |
| 骑手匹配 | 最近骑手、最小 ETA、负载感知 | 把推荐结果接入履约仿真 |
| LaDe 校准 | 从真实末端配送任务估计速度、服务时长、负载、可靠性 | 改善骑手 proxy 的可信边界 |

三方重排的解释口径：

```text
final_score = user_preference
            + merchant_fairness
            + eta_score
            + supply_score
            + list_coverage_gain
```

其中：

- `user_preference`：复购、品类偏好、价格匹配、时段偏好、质量、新颖性。
- `merchant_fairness`：给低曝光但质量可接受的商家更多机会，降低头部垄断。
- `eta_score`：估计用户-商家空间距离和履约时间，ETA 越低越好。
- `supply_score`：商家评分、供给稳定和订单压力 proxy。
- `list_coverage_gain`：避免 Top-K 列表全是同一类或同一批头部商家。

## 4. 核心实验结果

### 离线推荐结果

| 模型 | Recall@20 | NDCG@20 | Coverage@20 | Exposure Gini | 讲法 |
|---|---:|---:|---:|---:|---|
| Popular | 0.0470 | 0.0210 | 0.0062 | 0.9952 | 热门基线，覆盖极窄 |
| BPR-MF | 0.1620 | 0.1068 | 0.1350 | 0.9516 | 传统隐式反馈基线 |
| UserOnly | 0.4287 | 0.3423 | 0.2841 | 0.8857 | 用户画像已经很强 |
| Seq-Tuned | 0.4675 | 0.3652 | 0.2884 | 0.8927 | 离线准确性最强 |
| Seq-xQuAD-Tripartite | 0.4180 | 0.3440 | 0.2847 | 0.8882 | 准确性略降，服务三方目标 |
| Session-SPU-Tripartite | 0.4233 | 0.3451 | 0.2903 | 0.8778 | 点击与菜品类目带来更好曝光公平 |

图表建议：

- `outputs/figures/offline_recall20.png`
- `outputs/figures/tradeoff_ndcg_gini.png`
- `outputs/figures/tradeoff_recall_coverage.png`

### 默认骑手仿真结果（真实烟台地理，10 种子均值 ±95% CI，120 骑手）

| 策略 | Avg ETA | Timeout Rate | Platform Utility | 讲法 |
|---|---:|---:|---:|---|
| Popular + Nearest | 59.06 ±2.28 | 0.579 | 0.391 | 只看热门和最近骑手，整体差 |
| Seq-Tuned + MinETA | 40.44 ±1.21 | 0.287 | 0.533 | 推荐准确强，不等于履约最好 |
| Seq-xQuAD-Tripartite + Batch | 35.90 ±1.88 | 0.195 | 0.553 | 三方重排+批量匹配显著改善履约 |
| Session-SPU-Tripartite + Batch | 36.79 ±1.25 | 0.199 | 0.557 | session/SPU 信号，平台效用最高 |
| KG-Tripartite + Batch | 36.90 ±1.04 | 0.211 | 0.550 | 知识图谱兴趣信号 + 离线精度最强的三方 |
| KG-Tripartite + RouteMinETA | 36.15 ±0.96 | 0.229 | 0.542 | 顺路机制常规运力下与最优打平 |

### 运力紧张压力场景（30 骑手，`make simulate-stress`）——顺路派单的主战场

| 策略 | Avg ETA | Timeout Rate | Platform Utility | 讲法 |
|---|---:|---:|---:|---|
| Seq-Tuned + MinETA | 76.64 ±2.46 | 0.819 | 0.442 | 高峰爆单时纯用户侧全面恶化 |
| Seq-xQuAD-Tripartite + Batch | 60.96 ±4.84 | 0.679 | 0.486 | 槽位匹配有帮助但仍拥堵 |
| KG-Tripartite + RouteMinETA | **44.88 ±1.55** | **0.457** | **0.544** | 顺路合单：ETA 低约 16 分钟，顺路率 66.9% |
| KG-Tripartite + RouteBatch | 47.34 ±1.66 | 0.515 | 0.525 | 绕行+ETA 等权目标，车队里程更省 |

一句话讲法：**顺路派单不是永远更快，而是高峰运力吃紧时的韧性机制**——与商家侧"高峰爆单风险"面板构成同一个故事。

图表建议：

- `outputs/figures/simulation_avg_eta.png`
- `outputs/figures/simulation_timeout_rate.png`
- `outputs/figures/simulation_platform_utility.png`
- `outputs/figures/tripartite_scorecard.png`

### LaDe 真实配送数据校准结果

LaDe 校准得到的骑手参数：

| 参数 | 值 | 含义 |
|---|---:|---|
| `speed_kmph` | 8.0 | 按配送距离/任务时长估计后再按合理范围裁剪 |
| `service_minutes` | 90.0 | 包裹末端配送任务中位服务时长较长 |
| `initial_load_lambda` | 2.5 | 同一配送员 accept-finish 区间重叠任务估计的初始负载强度 |
| `reliability_mean` | 0.9002 | 由时长分布构造的可靠性 proxy |
| `reliability_std` | 0.1 | 可靠性波动 proxy |

外卖展示口径 `food-scaled`：

| 参数 | Raw LaDe | Food-scaled | 处理 |
|---|---:|---:|---|
| `speed_kmph` | 8.0 | 20.0 | 重标定到外卖骑行速度尺度 |
| `service_minutes` | 90.0 | 10.0 | 重标定到外卖出餐/服务时长尺度 |
| `initial_load_lambda` | 2.5 | 2.5 | 保留 LaDe 任务重叠负载 |
| `reliability_mean` | 0.9002 | 0.9002 | 保留 LaDe 可靠性 proxy |
| `reliability_std` | 0.1 | 0.1 | 保留 LaDe 波动 proxy |

LaDe 分布图：

![LaDe calibration summary](../outputs/presentation/lade_calibration_summary.png)

Raw LaDe 校准后的仿真结果：

| 策略 | Avg ETA | Timeout Rate | Platform Utility |
|---|---:|---:|---:|
| Popular + Nearest | 151.94 | 1.0000 | 0.2700 |
| UserOnly + MinETA | 125.29 | 1.0000 | 0.3883 |
| Seq-Tuned + MinETA | 122.27 | 1.0000 | 0.3933 |
| LightGBM-LTR + MinETA | 123.37 | 1.0000 | 0.3953 |
| Seq-xQuAD-Tripartite | 117.61 | 0.9875 | 0.3891 |
| Seq-xQuAD-Tripartite-Batch | 116.10 | 1.0000 | 0.3879 |
| Session-SPU-Tripartite | 122.23 | 0.9882 | 0.3958 |

Food-scaled LaDe 校准后的仿真结果：

| 策略 | Avg ETA | Timeout Rate | Platform Utility |
|---|---:|---:|---:|
| Popular + Nearest | 68.31 | 0.6909 | 0.3628 |
| UserOnly + MinETA | 53.76 | 0.7356 | 0.4562 |
| Seq-Tuned + MinETA | 52.26 | 0.7474 | 0.4563 |
| LightGBM-LTR + MinETA | 52.39 | 0.7128 | 0.4683 |
| Seq-xQuAD-Tripartite | 47.82 | 0.5750 | 0.5103 |
| Seq-xQuAD-Tripartite-Batch | 47.29 | 0.5500 | 0.5154 |
| Session-SPU-Tripartite | 49.97 | 0.6000 | 0.5004 |

图表建议：

![Default vs LaDe](../outputs/presentation/default_vs_lade_eta_timeout.png)

![LaDe policy scatter](../outputs/presentation/lade_policy_utility_scatter.png)

![Food-scaled LaDe comparison](../outputs/presentation/lade_food_scaled_comparison.png)

![Raw vs food-scaled LaDe](../outputs/presentation/lade_raw_vs_food_scaled.png)

解释重点：

- LaDe 是包裹末端配送，任务中位时长 `94` 分钟，明显长于外卖即时配送。
- 项目默认用 `45` 分钟判断外卖超时，所以 Raw LaDe 直接套用后 Timeout Rate 接近 `1.0` 是合理的压力测试结果。
- Food-scaled LaDe 保留 LaDe 的任务重叠负载和可靠性 proxy，只把速度与服务时长改回外卖 SLA 尺度；这组结果更适合作为答辩中的策略对比图。
- 这不是说明 FoodFlow 策略失效，而是说明跨领域数据不能直接当作外卖时长；LaDe 适合校准负载和波动趋势，正式外卖 ETA 需要领域缩放。
- Raw LaDe 仿真输出中的 `rider_service_minutes` 为 `30.0`，是骑手生成环节为避免极端值而做的上限裁剪；完整 Raw LaDe 校准 JSON 中保留了估计值 `90.0`。Food-scaled 输出中 `rider_service_minutes` 约为 `10.0`。

## 5. PPT 结构建议

建议做 12 页，10-12 分钟比较稳。

### Slide 1. 标题

标题：FoodFlow 外卖三方推荐与动态履约仿真系统

要讲：

- 课程项目、数据来源、三方视角。
- 一句话：不只优化 Recall，而是把推荐接到履约仿真里。

视觉：

- 简单三方图：用户、商家、骑手，中间是 FoodFlow。

### Slide 2. 问题动机

核心问题：为什么外卖推荐不能只看 Recall？

要讲：

- 普通推荐关注用户点击/购买。
- 外卖推荐会影响订单地理分布。
- 订单地理分布会影响骑手派单、ETA、超时率和商家曝光。

建议话术：

> 一个推荐列表看起来命中率很高，但如果它把订单集中到少数远距离商家，就可能导致骑手负载不均和超时上升。

### Slide 3. 数据来源与边界

要讲：

- TRD 真实外卖数据：200k 用户、29k 商家、1.068M 训练订单。
- Session 点击和 SPU 菜品明细也被用于新增策略。
- 骑手侧边界：TRD 没有真实骑手派单，所以默认是仿真；LaDe 用于校准参数。

视觉：

- 使用数据统计表。
- 可引用 `docs/DATA_AUDIT.md`。

### Slide 4. 系统架构

要讲：

- 数据处理、推荐、三方重排、模拟下单、骑手匹配、动态仿真、指标输出。
- 强调所有环节都能由 Makefile/CLI 复现。

可放命令：

```bash
make preprocess eval simulate figures report
make demo
```

### Slide 5. 推荐算法

要讲：

- Popular 和 BPR-MF 是基线。
- UserOnly 使用用户画像。
- Seq-Tuned 使用复购、时间衰减、转移概率、品类偏好。
- Seq-xQuAD-Tripartite 在 Top-K 列表层加入覆盖、公平、ETA 和供给。
- Session-SPU-Tripartite 额外利用 session 点击候选和菜品 SPU 类目偏好。

视觉：

- 放打分公式和模块表。

### Slide 6. 离线推荐结果

要讲：

- `Seq-Tuned` 离线准确性最强。
- 三方策略不是单纯追求 Recall 最大，而是保留较强准确性的同时改善覆盖与公平。

视觉：

- `outputs/figures/offline_recall20.png`
- `outputs/figures/tradeoff_ndcg_gini.png`

### Slide 7. 动态履约仿真设计

要讲：

- 午餐高峰多时间步。
- 推荐列表产生模拟订单。
- 订单进入骑手候选池。
- 策略比较：最近骑手、最小 ETA、负载感知。

评价指标：

- Avg ETA
- Timeout Rate
- Rider Load Std
- Merchant Exposure Gini
- User Satisfaction
- Platform Utility

### Slide 8. 默认仿真结果

要讲：

- 纯推荐准确性强，不一定系统最优。
- `Seq-xQuAD-Tripartite-Batch` 平均 ETA 和超时率最好，Platform Utility 最高。
- `Session-SPU-Tripartite` 说明点击和菜品信号能接入三方链路。

视觉：

- `outputs/figures/simulation_avg_eta.png`
- `outputs/figures/simulation_timeout_rate.png`
- `outputs/figures/simulation_platform_utility.png`

### Slide 9. LaDe 真实配送数据校准

要讲：

- 回答“有没有用 LaDe”：有，用真实 Hugging Face `delivery_five_cities.csv`。
- 讲清楚 LaDe 的角色：校准骑手速度、服务时长、重叠负载、可靠性；不用于推荐训练。
- 强调跨领域边界：LaDe 是包裹配送，不是外卖。

视觉：

- `outputs/presentation/lade_calibration_summary.png`

### Slide 10. LaDe 双口径：压力测试与外卖尺度重标定

要讲：

- Raw LaDe 直接套进外卖 45 分钟超时阈值会导致 Timeout Rate 接近 1，这是跨领域压力测试。
- Food-scaled LaDe 保留 LaDe 的负载和可靠性，把速度/服务时长重标定到外卖尺度。
- 在 Food-scaled 结果里，`Seq-xQuAD-Tripartite-Batch` Avg ETA `47.29`、Timeout Rate `0.55`、Platform Utility `0.5154`，仍然体现三方策略优势。

视觉：

- `outputs/presentation/lade_raw_vs_food_scaled.png`
- `outputs/presentation/lade_food_scaled_comparison.png`

### Slide 11. Demo 与可解释案例

要讲：

- 用真实 TRD processed 数据中的用户 `8`。
- 策略选 `Seq-xQuAD-Tripartite`。
- 展示推荐分数组成：用户偏好、商家公平、ETA、供给。
- 展示骑手候选：ETA、到店距离、当前负载、可靠性。

视觉：

- `outputs/presentation/explainability_case.png`

案例数据：

| 项 | 值 |
|---|---|
| 用户 | `8` |
| 策略 | `Seq-xQuAD-Tripartite` |
| Top1 商家 | `#2803` |
| 推荐理由 | 复购 1 次 / 偏好品类 0 / 曝光补偿 |
| 商家 ETA | 49.8 min |
| Top1 骑手 | `r0944` |
| 骑手 ETA | 41.6 min |
| 骑手到店距离 | 0.55 km |
| 骑手负载 | 0 |

讲法：

> 这页的重点不是说某个黑箱模型给出了 Top1，而是把一次下单拆成两层解释。第一层解释为什么推荐这个商家，第二层解释为什么这个订单分给这个骑手。

### Slide 12. 总结、局限与后续工作

结论：

- 推荐侧：Seq-Tuned 最强。
- 系统侧：三方重排改善 ETA、超时和平台效用。
- 数据侧：TRD 用于真实外卖推荐，LaDe 用于真实末端配送校准。

局限：

- 没有真实外卖骑手派单记录。
- 平台效用权重是课程项目中的解释性设置。
- LightGBM 在当前环境如果未安装，会退化为 Seq-Tuned fallback。
- LaDe 跨领域，需要做时长缩放后才适合外卖 ETA 口径。

后续方向：

- 接入真实外卖骑手派单日志。
- 做 LaDe 到外卖场景的 domain scaling。
- 使用图模型或时空模型联合优化推荐与调度。

## 6. Demo 演示脚本

### 启动

```bash
make demo
```

浏览器访问：

```text
http://localhost:8501
```

如果 8501 被占用：

```bash
make STREAMLIT_FLAGS="--server.port 8502" demo
```

### 现场点击顺序

0. **答辩开始前**：点侧栏"预热演示缓存（全部策略 + 高峰回放）"，等进度条走完；之后现场切策略、开策略对比、跑高峰回放都命中缓存，不会卡顿。
1. 侧边栏选择默认用户 `8`。
2. 时段选择“午餐高峰”。
3. 策略选择 `Seq-xQuAD-Tripartite`。
4. Top-K 保持 12。
5. 先讲首屏 KPI：用户、策略、Top1 商家、平均 ETA、在线骑手数。
6. 往下看推荐商家卡片，讲 Top1 的分数组成和理由。
7. 看空间图：默认为**真实城市底图**（烟台核心区，LaDe 真实配送 GPS 分布）。侧栏"地图底图"默认高德（国内网络秒开，已做 GCJ-02 坐标纠偏对齐）；海外网络可切 OSM/Carto；彻底断网选"无底图（离线画布）"。
8. 看"三方视角"面板：用户 tab 讲菜品级推荐；商家 tab 讲高峰爆单风险与品类供给配额；骑手 tab 讲顺路单的边际绕行成本——这直接对应 RouteBatch 顺路派单策略。
9. 看骑手候选榜，讲为什么不是简单最近，而是 ETA、负载、可靠性综合。
10. 切换 `Seq-Tuned` 或 `UserOnly`，对比推荐理由和 ETA 变化；若被问知识图谱，切 `KG-Tripartite`，讲推荐理由里的品类/区域/价位图谱路径（深度版模型在 kg-demo 子项目）。
11. 切到高峰回放页：先点"生成履约动画"再按 ▶ 播放——骑手在真实地图上沿取餐、送达路径逐分钟移动、负载颜色加深、顺路接单路径延长，这是骑手侧最直观的一段；然后再勾选运行高峰回放看累计 KPI 与热力图。

### 如果现场 demo 卡住

直接使用静态备份图：

- `outputs/presentation/explainability_case.png`
- `outputs/presentation/default_vs_lade_eta_timeout.png`
- `outputs/figures/tripartite_scorecard.png`
- `outputs/figures/simulation_platform_utility.png`

备份话术：

> Demo 本质上就是把这张可解释样例图动态化。左边是推荐分数组成，右边是骑手候选排序；线上页面只是允许切用户、切策略、切时间步。

## 7. 生成的展示素材清单

新增素材：

| 文件 | 用途 |
|---|---|
| `outputs/presentation/lade_calibration_summary.png` | LaDe 校准统计页 |
| `outputs/presentation/default_vs_lade_eta_timeout.png` | 默认骑手 vs LaDe 校准对比 |
| `outputs/presentation/lade_policy_utility_scatter.png` | LaDe 校准下策略 ETA-效用关系 |
| `outputs/presentation/lade_food_scaled_comparison.png` | Food-scaled LaDe 与默认仿真对比 |
| `outputs/presentation/lade_raw_vs_food_scaled.png` | Raw LaDe 与 Food-scaled LaDe 敏感性对比 |
| `outputs/presentation/explainability_case.png` | demo 小样本解释图 |
| `outputs/presentation/explainability_case.md` | 小样本文字说明 |
| `outputs/presentation/explainability_case_recommendations.csv` | 推荐解释表 |
| `outputs/presentation/explainability_case_riders.csv` | 骑手候选解释表 |
| `outputs/presentation/presentation_assets.csv` | 素材索引 |

已有素材：

| 文件 | 用途 |
|---|---|
| `outputs/figures/offline_recall20.png` | 离线 Recall |
| `outputs/figures/offline_ndcg20.png` | 离线 NDCG |
| `outputs/figures/tradeoff_ndcg_gini.png` | 准确性-曝光公平权衡 |
| `outputs/figures/tradeoff_recall_coverage.png` | Recall-覆盖权衡 |
| `outputs/figures/simulation_avg_eta.png` | 默认仿真 Avg ETA |
| `outputs/figures/simulation_timeout_rate.png` | 默认仿真超时率 |
| `outputs/figures/simulation_platform_utility.png` | 默认仿真平台效用 |
| `outputs/figures/tripartite_scorecard.png` | 三方策略综合图 |

## 8. 答辩 Q&A 备忘

Q: 有没有用 LaDe？

A: 有。我们下载并运行了 Hugging Face 上的 `delivery_five_cities.csv`，实际读取 472,419 行、1,870 名配送员，用它估计骑手速度、服务时长、重叠负载和可靠性。LaDe 不参与用户-商家推荐训练，只用于骑手仿真参数校准。

Q: 为什么 Raw LaDe 后超时率接近 1？

A: 因为 LaDe 是包裹末端配送，中位任务时长约 94 分钟；而外卖超时阈值用 45 分钟。直接跨领域套用会自然变成压力测试。这个结果提醒我们不能把包裹配送时长当成外卖真实 ETA。

Q: 那为什么还要做 Food-scaled LaDe？

A: Raw LaDe 负责说明真实数据和外卖 SLA 的领域差异；Food-scaled LaDe 保留 LaDe 的负载和可靠性信息，只把速度/服务时长重标定到外卖尺度，用于展示策略差异。它不是伪造真实 LaDe，而是一个明确标注的敏感性实验。

Q: 你们的骑手数据真实吗？

A: 默认仿真骑手是固定 seed 合成 proxy；真实部分是 LaDe 任务表提供的校准参数。严谨说法是“骑手侧可由真实末端配送数据校准”，不是“拥有真实外卖骑手派单数据”。

Q: 为什么不只用 Recall 最高的 Seq-Tuned？

A: `Seq-Tuned` 的 Recall@20 最高，但外卖平台还要考虑履约。默认仿真里 `Seq-xQuAD-Tripartite-Batch` 的 Avg ETA、超时率和平台效用更好，说明系统级目标不等价于单一推荐准确性。

Q: Session-SPU 改进是什么意思？

A: 它使用 TRD optional 文件里的下单前点击商家序列和订单菜品 SPU 类目。点击商家用于扩展候选，SPU 类目用于衡量用户菜品偏好与商家菜品画像的匹配。

Q: 结果能复现吗？

A: 可以。TRD 结果由 `make preprocess eval simulate figures report` 生成；Raw LaDe 校准结果由 `python -m foodflow.cli simulate --rider-tasks /private/tmp/lade/delivery_five_cities.csv` 生成；Food-scaled LaDe 结果由追加 `--rider-calibration-profile food-scaled` 生成。

## 9. 最后提醒

展示时最重要的是边界清楚：

- 可以说：推荐侧用了真实 TRD 外卖订单；LaDe 被用于真实末端配送任务校准。
- 不要说：LaDe 是外卖平台骑手数据。
- 可以说：Raw LaDe 暴露包裹配送和外卖配送的领域差异；Food-scaled LaDe 用于展示保留负载/可靠性后的策略敏感性。
- 不要把 Raw LaDe 校准后的 116-152 分钟 ETA 当作真实外卖 ETA 结论。

一句收束：

> FoodFlow 的贡献不是堆一个更复杂的推荐模型，而是把推荐、商家曝光和骑手履约放进同一个可解释、可复现的实验闭环。
