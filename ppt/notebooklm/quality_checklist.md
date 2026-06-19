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
- [ ] `Seq-xQuAD-Tripartite`：Recall@20 = `0.4180`，NDCG@20 = `0.3439`，HitRate@20 = `0.5733`。
- [ ] `Session-SPU-Tripartite`：Recall@20 = `0.4147`，NDCG@20 = `0.3431`，HitRate@20 = `0.5700`。
- [ ] `BPR-MF`：Recall@20 = `0.1620`，NDCG@20 = `0.1068`，HitRate@20 = `0.2433`。
- [ ] `Popular`：Recall@20 = `0.0470`，NDCG@20 = `0.0210`，HitRate@20 = `0.0900`。
- [ ] `Popular + Nearest`：Avg ETA = `89.68`，Timeout Rate = `0.8025`，Utility = `0.3218`。
- [ ] `UserOnly + MinETA`：Avg ETA = `52.14`，Timeout Rate = `0.6897`，Utility = `0.4060`。
- [ ] `Seq-Tuned + MinETA`：Avg ETA = `50.51`，Timeout Rate = `0.6163`，Utility = `0.4205`。
- [ ] `LightGBM-LTR + MinETA`：Avg ETA = `53.96`，Timeout Rate = `0.6854`，Utility = `0.3903`。
- [ ] `Seq-xQuAD-Tripartite + Greedy`：Avg ETA = `49.33`，Timeout Rate = `0.5604`，Utility = `0.4568`。
- [ ] `Seq-xQuAD-Tripartite + Batch`：Avg ETA = `48.32`，Timeout Rate = `0.5275`，Utility = `0.4581`。
- [ ] `Session-SPU-Tripartite + Greedy`：Avg ETA = `46.47`，Timeout Rate = `0.4941`，Utility = `0.4694`。
