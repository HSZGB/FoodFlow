# FoodFlow 算法与骑手数据增强说明

## 1. 新增算法信号

本轮新增 `Session-SPU-Tripartite` 推荐器。它复用原有 `Seq-xQuAD-Tripartite` 的三方重排逻辑，并增加两个来自 TRD optional txt 文件的信号：

- `orders_poi_session.txt`：下单前点击过的商家序列，预处理后写入 `data/processed/session_interactions.csv`。
- `orders_spu_train.txt` / `orders_test_spu.txt`：订单中的菜品 SPU，预处理后写入 `order_spus_train.csv` 和 `order_spus_test.csv`。

推荐器会优先把近期点击商家纳入候选集，并用用户历史菜品类别与商家历史菜品类别的重合度补充排序分。这样算法改进仍然来自同一个 TRD 数据源，不需要额外伪造用户行为。

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

## 4. 答辩边界

这项增强不能把 TRD 变成真实派单数据。更准确的表述是：

> 推荐侧使用真实 TRD 订单与点击/菜品信号；骑手侧仍是仿真，但可用 LaDe 等公开末端配送任务数据校准速度、服务时长和负载分布。LaDe 不是外卖平台数据，因此不能声称获得了真实外卖骑手派单记录。
