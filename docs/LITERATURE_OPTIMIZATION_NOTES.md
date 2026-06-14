# FoodFlow 论文调研与可落地优化点

## 调研结论

FoodFlow 当前最值得增强的方向不是直接堆大模型，而是把外卖场景的“重复消费、短序列下一单、多目标重排、即时履约”讲清楚并落到可运行代码。外卖推荐和电影/新闻推荐不同，用户经常复购同一商家，也会受最近一次订单、午晚高峰、距离和骑手供给影响。

## 可借鉴论文与项目落点

| 方向 | 代表工作 | 可迁移想法 | FoodFlow 落地 |
|---|---|---|---|
| 隐式反馈排序 | Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, 2012 | 用隐式行为做个性化排序，而不是显式评分预测 | 保留 `BPR-MF` 作为传统排序基线 |
| 下一篮子推荐 | Rendle et al., Factorizing Personalized Markov Chains for Next-Basket Recommendation, WWW 2010 | 用户长期偏好 + 最近一次行为共同决定下一次消费 | 新增 `Seq-Hybrid`，融合复购、最近订单、商家转移概率 |
| 会话/短序列推荐 | Neural Attentive Session-based Recommendation, CIKM 2017；SR-GNN, AAAI 2019 | 最近行为在短期意图中权重更高 | `Seq-Hybrid` 使用快/慢两个时间衰减复购分 |
| 时间上下文推荐 | Déjà vu: Contextualized Temporal Attention for Sequential Recommendation, 2020 | 重复消费需要显式建模时间与上下文 | 加入时段、最近订单和时间衰减特征 |
| 多目标推荐 | Multi-Objective Recommender Systems: Survey and Challenges, 2022 | 准确率、覆盖、公平、履约并非单一目标 | 保留 `Seq-Hybrid` 作为准确性上界，同时用 `Ours-Full` 讲系统级收益 |
| 校准推荐 | Calibrated Recommendations, RecSys 2018 | 推荐列表的品类分布应贴近用户历史偏好，避免看似多样但偏离兴趣 | 新增 `CategoryJSD@20`，用 Jensen-Shannon divergence 衡量推荐品类校准 |
| 曝光公平 | Burke, Multisided Fairness for Recommendation, 2017；Joint Multisided Exposure Fairness, SIGIR 2022 | 推荐平台要同时考虑消费者和供给侧曝光 | `Ours-Full` 和图表继续展示 Coverage、Long-tail Exposure、Exposure Gini |
| 多样性重排 | YouTube DPP reranking, CIKM 2018；MMR/xQuAD 思路 | 排序后可做多样性、覆盖和供给约束重排 | 新增 `Seq-xQuAD`，在序列相关性后做品类覆盖和长尾曝光的列表级重排 |
| 即时配送派单 | Matching Algorithm with Reinforcement Learning and Decoupling Strategy for Order Dispatching in On-Demand Food Delivery, TST 2023 | 推荐产生订单后，派单和 ETA 会反过来影响平台体验 | 新增 `Seq-xQuAD-Tripartite`，把列表级重排和 ETA/供给/负载感知派单串成闭环 |

## 本轮实现：Seq-Hybrid 与 Seq-xQuAD

`Seq-Hybrid` 是一个轻量的序列混合模型：

```text
score =
  0.25 * fast_recency
+ 0.12 * slow_recency
+ 0.30 * repeat_frequency
+ 0.23 * transition_score
+ 0.05 * category_preference
+ 0.03 * merchant_popularity
+ 0.02 * merchant_quality
```

其中：

- `fast_recency`：最近几单快速衰减，捕捉短期意图。
- `slow_recency`：较慢衰减，保留稳定复购习惯。
- `repeat_frequency`：外卖高频复购信号。
- `transition_score`：用户最近商家到候选商家的全局转移概率，借鉴 FPMC/下一篮子思路。
- `category_preference`、`merchant_popularity`、`merchant_quality`：保留可解释业务特征。

在此基础上继续新增 `Seq-xQuAD`。它不替换序列模型，而是在 `Seq-Hybrid` 候选分上做贪心列表级重排：

```text
list_score =
  relevance_weight * normalized_seq_score
+ diversity_weight * uncovered_category_gain
+ tail_weight * long_tail_gain
```

其中 `uncovered_category_gain` 借鉴 xQuAD/MMR 的覆盖思想，让推荐列表不要被单一品类挤满；`long_tail_gain` 给订单量较低的商家一点曝光补偿。这个版本的定位不是工业级 DPP，而是课程项目中可解释、可运行、指标有提升的列表重排模块。

随后加入 `Seq-xQuAD-Tripartite`：先用序列偏好、曝光公平、ETA 和供给分得到三方相关性，再把这个相关性送入 xQuAD 式列表重排。它的意义不在于继续刷新离线 Recall，而是把“用户喜欢什么”和“系统能否履约”放在同一个可解释排序器里。

最新一轮继续加入 `Seq-Tuned` 和 `Seq-Tuned-xQuAD`。这不是换一套黑盒模型，而是在 `Seq-Hybrid` 的同一批可解释特征上做轻量权重重分配：提高外卖场景最关键的复购频率、最近商家转移和品类偏好权重，降低全局流行度权重。`Seq-Tuned` 用于追求离线准确率前沿，`Seq-Tuned-xQuAD` 则在该底座上继续做列表级覆盖和长尾曝光重排。

## 当前指标收益

完整 TRD 处理结果上，`Seq-Tuned` 成为离线准确率最强策略；`Seq-Tuned-xQuAD` 在几乎保持同等 Recall 的同时，取得更低 Exposure Gini 和更低 CategoryJSD@20，说明列表更不集中，也更贴近用户历史品类偏好：

| 模型 | Recall@20 | NDCG@20 | HitRate@20 | CategoryJSD@20 |
|---|---:|---:|---:|---:|
| UserOnly | 0.4287 | 0.3423 | 0.5733 | 0.0156 |
| Seq-Hybrid | 0.4441 | 0.3513 | 0.5933 | 0.0152 |
| Seq-xQuAD | 0.4447 | 0.3538 | 0.5967 | 0.0151 |
| Seq-Tuned | 0.4675 | 0.3652 | 0.6267 | 0.0152 |
| Seq-Tuned-xQuAD | 0.4670 | 0.3613 | 0.6200 | 0.0140 |
| Seq-xQuAD-Tripartite | 0.4180 | 0.3440 | 0.5733 | 0.0158 |

本轮继续实现了 `Seq-Tripartite`：在 `Seq-Hybrid` 的序列偏好底座上轻量加入公平、ETA 和供给约束。它的 NDCG@20 和 HitRate@20 高于 UserOnly，仿真平台效用也高于 `Seq-Hybrid + MinETA`，说明序列准确性可以和三方约束结合。

仿真结果中，`Seq-xQuAD-Tripartite` 取得当前最高平台效用 `0.5264`、最低平均 ETA `50.33` 和最低超时率 `0.5319`。答辩故事因此更自然：`Seq-Tuned` 证明推荐准确性可以继续提升；`Seq-Tuned-xQuAD` 证明高准确序列模型可以兼顾曝光与校准；`Seq-xQuAD-Tripartite` 证明高准确序列模型接入三方约束后可以改善履约；`Ours-Full` 作为原始三方重排对照，说明外卖平台不能只看准确性，还要看 ETA、超时率、商家曝光和骑手负载。

## 下一步可继续尝试

1. 对三方权重做网格搜索或贝叶斯优化，输出 Pareto frontier，而不是只给单点权重。
2. 用真实地图底图或 hexbin 密度图替换普通散点，让 demo 的空间分布更像业务看板。
3. 把 `Seq-xQuAD` 的多样性项从品类覆盖升级为距离、商家层级、菜品标签的多维覆盖。
