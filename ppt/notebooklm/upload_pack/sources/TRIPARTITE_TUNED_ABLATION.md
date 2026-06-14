# Tuned 序列底座接入三方重排的消融记录

## 验证问题

`Seq-Tuned` 已经把离线推荐 Recall@20 提升到 `0.4675`。一个自然问题是：是否应该把 Tuned 序列底座直接接入三方重排，替换当前的 `Seq-xQuAD-Tripartite`？

本次消融的目的不是继续追求单一 Recall，而是验证：

- 高准确率序列底座是否能同步提升平台效用。
- 增加 ETA 权重后，是否能进一步降低履约时间和超时率。
- 当前选择 `Seq-xQuAD-Tripartite` 作为系统效用前沿是否仍然成立。

## 候选策略

使用与正式仿真相同的数据、seed、用户请求流、选择模型、骑手生成和负载感知派单逻辑，只替换推荐器：

| 策略 | 说明 |
|---|---|
| Seq-Tuned-Tripartite | 使用 `Seq-Tuned` 序列权重 + 轻量三方分数 |
| Seq-Tuned-xQuAD-Tripartite | 使用 `Seq-Tuned-xQuAD` 序列权重 + xQuAD 三方列表重排 |
| Seq-Tuned-xQuAD-Tripartite-ETA | 在上一个策略上提高 ETA 与供给权重 |
| Seq-xQuAD-Tripartite | 当前默认系统效用前沿 |
| Ours-Full | 原始三方重排强对照 |

这些候选没有纳入默认 `build_recommenders()`，因为验证目标是判断是否值得替换默认主线。

## 离线指标

| 模型 | Recall@20 | NDCG@20 | HitRate@20 | Coverage@20 | ExposureGini | CategoryJSD@20 |
|---|---:|---:|---:|---:|---:|---:|
| Seq-Tuned-Tripartite | 0.4073 | 0.3453 | 0.5633 | 0.2992 | 0.8795 | 0.0148 |
| Seq-Tuned-xQuAD-Tripartite | 0.4029 | 0.3395 | 0.5567 | 0.4228 | 0.8314 | 0.0137 |
| Seq-Tuned-xQuAD-Tripartite-ETA | 0.4029 | 0.3396 | 0.5567 | 0.4305 | 0.8240 | 0.0137 |
| Seq-xQuAD-Tripartite | 0.4180 | 0.3440 | 0.5733 | 0.2847 | 0.8882 | 0.0158 |

观察：

- Tuned 三方候选的离线 Recall 没有超过当前 `Seq-xQuAD-Tripartite`。
- Tuned-xQuAD 三方候选能显著降低 Exposure Gini 和 CategoryJSD，但准确率下降更多。
- 这说明“高准确率 Tuned 底座”接入三方约束后，会被公平、ETA、供给和列表覆盖重新排序，不能简单等同于继续保持用户侧 Recall 前沿。

## 仿真指标

| 策略 | Avg ETA | Timeout Rate | On-time Rate | User Satisfaction | Platform Utility |
|---|---:|---:|---:|---:|---:|
| Seq-Tuned-Tripartite | 54.42 | 0.6737 | 0.3263 | 0.8842 | 0.4981 |
| Seq-Tuned-xQuAD-Tripartite | 54.54 | 0.7400 | 0.2600 | 0.9010 | 0.4854 |
| Seq-Tuned-xQuAD-Tripartite-ETA | 49.04 | 0.5543 | 0.4457 | 0.8505 | 0.5166 |
| Seq-xQuAD-Tripartite | 50.33 | 0.5319 | 0.4681 | 0.8479 | 0.5264 |
| Ours-Full | 50.85 | 0.5714 | 0.4286 | 0.8973 | 0.5246 |

观察：

- `Seq-Tuned-xQuAD-Tripartite-ETA` 的平均 ETA 最低，为 `49.04`，说明提高 ETA 权重确实能压低平均履约时间。
- 但它的超时率仍高于 `Seq-xQuAD-Tripartite`，平台效用也低于 `0.5264`。
- `Ours-Full` 用户满意度最高，平台效用接近最优，是三方重排的强对照。
- 当前默认的 `Seq-xQuAD-Tripartite` 仍是平台效用最高策略。

## 结论

本次消融支持当前主线：

- `Seq-Tuned` 负责证明用户侧离线准确率可以显著提升。
- `Seq-Tuned-xQuAD` 负责证明高准确率可以兼顾曝光公平与品类校准。
- `Seq-xQuAD-Tripartite` 负责证明三方履约约束能带来最高平台效用。

因此不把 Tuned 三方候选纳入默认策略，不是因为没有尝试，而是因为它们没有在系统级仿真中超过当前 Pareto 前沿。
