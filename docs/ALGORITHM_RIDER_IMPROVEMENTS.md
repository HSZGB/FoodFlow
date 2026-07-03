# FoodFlow 算法与骑手数据增强说明

## 1. 新增算法信号

本轮新增 `Session-SPU-Tripartite` 推荐器。它复用原有 `Seq-xQuAD-Tripartite` 的三方重排逻辑，并增加两个来自 TRD optional txt 文件的信号：

- `orders_poi_session.txt`：下单前点击过的商家序列，兼容真实文件的 `#` 分隔符；只保留训练期信号后写入 `data/processed/session_interactions.csv`。
- `orders_spu_train.txt` / `orders_spu_test_label.txt`：订单中的菜品 SPU，预处理后写入 `order_spus_train.csv` 和 `order_spus_test.csv`。

推荐器会优先把训练期近期点击商家纳入候选集，并用训练期菜品类别重合度补充排序分。测试期会话不会进入模型，避免离线评估信息泄漏。

## 2. 新增骑手校准路径

TRD 不包含真实骑手状态或派单记录，因此 FoodFlow 仍然把骑手侧定义为仿真。新增的骑手校准接口允许使用 LaDe 等公开末端配送任务 CSV 来估计仿真参数，包括：

- 骑手平均速度 `speed_kmph`
- 单任务服务时长 `service_minutes`
- 初始负载强度 `initial_load_lambda`，由同一配送员 task-accept 到 task-finish 区间的重叠任务数估计
- 可靠性均值和波动

外部 CSV 可使用项目通用字段：

```text
courier_id,accept_time,finish_time,pickup_lng,pickup_lat,delivery_lng,delivery_lat
```

`accept_time` 和 `finish_time` 可以是数值分钟，也可以是可由 pandas 解析的时间字符串。

也可以直接使用 LaDe delivery 文件字段：

```text
courier_id,accept_time,accept_gps_lng,accept_gps_lat,delivery_time,delivery_gps_lng,delivery_gps_lat
```

其中 `accept_gps_*` 作为任务开始位置，`delivery_gps_*` 作为任务完成位置，`delivery_time` 映射为 `finish_time`。

Hugging Face 上的合并版 `delivery_five_cities.csv` 使用另一组字段：

```text
delivery_user_id,receipt_time,receipt_lng,receipt_lat,sign_time,poi_lng,poi_lat
```

其中 `delivery_user_id` 映射为 `courier_id`，`receipt_time` 映射为 `accept_time`，`sign_time` 映射为 `finish_time`，`receipt_lng/lat -> poi_lng/lat` 用投影坐标欧氏距离估计任务距离。

## 3. 使用方式

默认仿真不变：

```bash
make simulate
```

使用外部配送任务数据校准骑手仿真：

```bash
make simulate-calibrated RIDER_TASKS=/path/to/lade_delivery.csv
```

等价 CLI：

```bash
python -m foodflow.cli simulate \
  --processed-dir data/processed \
  --output outputs/results/simulation_metrics_calibrated.csv \
  --rider-tasks /path/to/lade_delivery.csv
```

默认 `--rider-calibration-profile raw` 会直接使用外部任务表估计出的速度、服务时长、负载与可靠性。对于 LaDe 这类包裹末端配送数据，也可以使用外卖展示口径：

```bash
python -m foodflow.cli simulate \
  --processed-dir data/processed \
  --output outputs/results/simulation_metrics_lade_food_scaled.csv \
  --rider-tasks /path/to/lade_delivery.csv \
  --rider-calibration-profile food-scaled
```

`food-scaled` 不改变 LaDe 提供的任务重叠负载和可靠性 proxy，只把速度和服务时长重标定到外卖 SLA 尺度。答辩时应把 `raw` 解释为跨领域压力测试，把 `food-scaled` 解释为保留 LaDe 结构信息后的外卖口径敏感性实验。

输出表会额外包含 `rider_speed_kmph` 与 `rider_service_minutes`，便于在报告中说明仿真参数来自固定默认值还是外部数据校准。

### 校准诊断（分布一致性与参数来源）

传入 `--rider-tasks` 时，`simulate` 会同时写出 `<output stem>_calibration.json`（或用 `--calibration-output` 指定路径），内容包括：

- `parameter_provenance`：逐参数标注来源是 `task-data-derived` 还是 `food-delivery-default`，避免"LaDe 校准"过度声称——`food-scaled` 口径下速度与服务时长是外卖默认值，只有负载与可靠性来自 LaDe；
- `empirical`：任务时长、任务速度、接单时并发负载的样本分位数（p25/p50/p75/p95）；
- `lognormal_fit`：对时长/速度拟合对数正态分布并做 KS 拟合优度检验（末端配送时长是重尾分布，仅取中位数会丢失信息）；
- `simulation_vs_data_ks`：对仿真实际使用的生成分布（与 `generate_riders` 同一采样规则）抽样，与任务数据做双样本 KS 检验，把"仿真输入分布离真实数据有多远"量化写进产物。

可靠性 proxy 的准时率现以显式 SLA 阈值（默认 45 分钟，与仿真超时口径一致）计算：`reliability_mean = 0.72 + 0.24 × P(任务时长 ≤ SLA)`。旧实现以样本 75 分位数为阈值，准时率按定义恒等于 0.75，与数据无关，已修正。

### 多种子置信区间

单种子结果是点估计，策略间平台效用差通常只有 0.01–0.05 量级。`simulate --simulation-seeds 42 43 ...` 会逐种子运行并按策略聚合，数值列输出跨种子均值，并追加 `<metric>_std` / `<metric>_ci95` / `n_seeds` 列。答辩与报告中的策略对比应引用该口径。

## 4. 真实城市地理（geocode）

TRD 不含坐标，旧实现用高斯随机数合成经纬度（同一商圈的商家甚至共享同一个点），
距离与 ETA 无物理意义。现在通过：

```bash
make geocode GEO_TASKS=data/lade/delivery_yt.parquet
```

把用户/商家嵌入 LaDe 真实城市（默认烟台）的末端配送 GPS 分布：

- 先粗聚类提取最大密度簇质心 10km 内的**城市核心区**（LaDe 城市文件覆盖整个都市圈，直接用会把用户撒到远郊）；
- 每个 aor_id 商圈映射到一个真实高密度簇，商家坐标从簇内真实点采样（约 60m 抖动）；
- 每个 aoi_id 居住区取真实锚点，用户在 120m 内散布；
- 元数据（bbox、簇数、来源与口径说明）写入 `data/processed/geo_note.json`。

诚实口径：这不是 TRD 的真实位置，而是"TRD 订单嵌入 LaDe 真实城市空间分布"；
但从此距离、ETA 与地图展示具备真实街区尺度。所有行程按直线距离 × 1.3 道路弯曲系数换算，
骑手默认速度相应校到 25 km/h（电动车市区口径）；备餐与骑手赶往商家**并行**计时
（取二者较大值），使 ETA 尺度落在 35–45 分钟的现实区间。

## 5. 顺路派单（路径感知边际插单）

骑手不再只按直线距离/单点 ETA 评估：每个骑手维护取/送航点序列（route），
新单按 **cheapest-insertion** 枚举所有 (取餐点, 送达点) 插入位置，计算边际绕行分钟数：

- `route_insertion`（RouteBatch/RouteGreedy）：目标 = 边际绕行 + ETA（等权，`ROUTE_ETA_WEIGHT`），
  批量版按匈牙利轮次匹配（每轮每骑手至多一单，应用插入后重算下一轮成本）；
- `route_min_eta`（RouteMinETA）：同一路径机制下只最小化该单 ETA，作为 route 家族的公平基线；
- 骑手随时间**沿路径行进**（每步推进 5 分钟，经过 dropoff 即完成一单），替代"派单即瞬移"；
- 新增指标：`avg_detour_minutes`（平均边际绕行）与 `enroute_pickup_share`（顺路接单率）。

关键实验结论（无免费午餐，均为 10 种子均值）：

- **运力充足**（120 骑手 vs 每步 16 单）：槽位批量匹配（Batch）ETA 更优——骑手大多空闲时，
  分散派单本来就是最优解，合单只会延迟；且 route 家族的 ETA 诚实计入队列行进时间，口径更严格。
- **运力紧张**（30 骑手，高峰爆单场景，`make simulate-stress`）：顺路合单显著胜出——
  平均 ETA 65→54 分钟、超时率 0.77→0.65、负载更均衡（rider_load_cv 下降）。

答辩表述建议：顺路派单不是"永远更快"，而是**高峰运力吃紧时的韧性机制**——
这与商家侧"高峰爆单风险"面板构成同一个故事。

## 6. 答辩边界

这项增强不能把 TRD 变成真实派单数据。更准确的表述是：

> 推荐侧使用真实 TRD 订单与点击/菜品信号；骑手侧仍是仿真，但可用 LaDe 等公开末端配送任务数据校准速度、服务时长和负载分布。LaDe 不是外卖平台数据，因此不能声称获得了真实外卖骑手派单记录。
