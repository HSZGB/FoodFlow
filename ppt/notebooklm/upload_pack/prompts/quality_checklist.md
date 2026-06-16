# FoodFlow PPT 生成后检查清单

用 NotebookLM 或其他工具生成 PPT 后，按这份清单逐页检查。每一项都要能在页面或讲稿中找到证据。

## 全局检查

- [ ] 11 页，16:9，中文，课程大作业答辩风格。
- [ ] 标题为 FoodFlow 或包含“外卖三方推荐与动态履约仿真”。
- [ ] 没有把骑手数据说成真实派单数据。
- [ ] 没有把 DHRD、LightGCN、KGAT 写成核心实现。
- [ ] 每页文字不过度堆叠，图表坐标轴与图例清晰。
- [ ] 指标数值来自 `offline_metrics.csv` 和 `simulation_metrics.csv`。

## 三方推荐检查

- [ ] 用户侧：明确是给用户推荐商家/菜品，评价 Recall、NDCG、MRR、HitRate。
- [ ] 商家侧：明确商家通过推荐获得曝光，评价 Coverage、Long-tail Exposure、Exposure Gini。
- [ ] 骑手侧：明确下单后的订单会被推荐/匹配给骑手，策略包括 Nearest、MinETA、LoadAware。
- [ ] 第 9 页案例能串起“用户画像 -> 商家推荐 -> 下单 -> 骑手匹配 -> ETA/超时风险”。

## 关键页检查

- [ ] Slide 3 写明 TRD、Zenodo DOI `10.5281/zenodo.8025855`、北京 11 个商圈、2021-03-01 至 2021-03-28。
- [ ] Slide 5 写明 Seq-xQuAD-Tripartite = 序列偏好 + 列表级覆盖 + 商家公平 + ETA + 供给分。
- [ ] Slide 6 出现 Recall@20/NDCG@20/Exposure Gini/Coverage 至少两个图或一个图加表。
- [ ] Slide 8 出现 Avg ETA、Timeout Rate、Rider Load Std、Platform Utility 至少两个图或一个图加表。
- [ ] Slide 10 同时写结论和局限，不只写优点。

## 必须保留的关键数字

- [ ] `UserOnly`：Recall@20 = `0.4287`，NDCG@20 = `0.3423`，HitRate@20 = `0.5733`。
- [ ] `Seq-Tuned`：Recall@20 = `0.4675`，NDCG@20 = `0.3652`，HitRate@20 = `0.6267`。
- [ ] `LightGBM-LTR`：Recall@20 = `0.4424`，NDCG@20 = `0.3243`，HitRate@20 = `0.5900`。
- [ ] `Seq-xQuAD-Tripartite`：Recall@20 = `0.4180`，NDCG@20 = `0.3440`，HitRate@20 = `0.5733`。
- [ ] `BPR-MF`：Recall@20 = `0.1620`，NDCG@20 = `0.1068`，HitRate@20 = `0.2433`。
- [ ] `Popular`：Recall@20 = `0.0470`，NDCG@20 = `0.0210`，HitRate@20 = `0.0900`。
- [ ] `Popular + Nearest`：Avg ETA = `86.64`，Timeout Rate = `0.8537`，Utility = `0.3098`。
- [ ] `UserOnly + MinETA`：Avg ETA = `53.40`，Timeout Rate = `0.7113`，Utility = `0.4240`。
- [ ] `Seq-Tuned + MinETA`：Avg ETA = `56.04`，Timeout Rate = `0.7097`，Utility = `0.4147`。
- [ ] `LightGBM-LTR + MinETA`：Avg ETA = `54.81`，Timeout Rate = `0.7419`，Utility = `0.3922`。
- [ ] `Seq-xQuAD-Tripartite + Greedy`：Avg ETA = `46.87`，Timeout Rate = `0.5200`，Utility = `0.4767`。
- [ ] `Seq-xQuAD-Tripartite`：Avg ETA = `48.98`，Timeout Rate = `0.5543`，Utility = `0.4662`。
