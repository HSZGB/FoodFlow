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
| 曝光公平 | Burke, Multisided Fairness for Recommendation, 2017；Joint Multisided Exposure Fairness, SIGIR 2022 | 推荐平台要同时考虑消费者和供给侧曝光 | `Ours-Full` 和图表继续展示 Coverage、Long-tail Exposure、Exposure Gini |
| 多样性重排 | YouTube DPP reranking, CIKM 2018；MMR/xQuAD 思路 | 排序后可做多样性、覆盖和供给约束重排 | 后续可把当前公平分升级成 DPP/xQuAD 风格的显式列表级重排 |
| 即时配送派单 | Matching Algorithm with Reinforcement Learning and Decoupling Strategy for Order Dispatching in On-Demand Food Delivery, TST 2023 | 推荐产生订单后，派单和 ETA 会反过来影响平台体验 | 当前履约仿真比较最近骑手、最小 ETA、负载感知派单 |

## 本轮实现：Seq-Hybrid

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

## 当前指标收益

完整 TRD 处理结果上，`Seq-Hybrid` 成为离线推荐指标最强策略：

| 模型 | Recall@20 | NDCG@20 | HitRate@20 |
|---|---:|---:|---:|
| UserOnly | 0.4287 | 0.3423 | 0.5733 |
| Seq-Hybrid | 0.4441 | 0.3513 | 0.5933 |

仿真结果中，`Ours-Full` 仍取得最高平台效用。答辩故事因此更自然：`Seq-Hybrid` 证明推荐准确性可以继续提升；`Ours-Full` 证明外卖平台不能只看准确性，还要看 ETA、超时率、商家曝光和骑手负载。

## 下一步可继续尝试

1. 把 `Seq-Hybrid` 作为 `Ours-Full` 的用户偏好底座，形成 `Seq-Tripartite`。
2. 引入 DPP/xQuAD 的列表级重排，让商家覆盖和品类多样性更直观。
3. 对三方权重做网格搜索或贝叶斯优化，输出 Pareto frontier，而不是只给单点权重。
4. 用真实地图底图或 hexbin 密度图替换普通散点，让 demo 的空间分布更像业务看板。
