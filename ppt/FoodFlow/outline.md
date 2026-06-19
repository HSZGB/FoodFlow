# FoodFlow 答辩 PPT 大纲（草稿，待审批）

Slide 1: FoodFlow：外卖三方推荐与动态履约仿真系统
- Key points: 项目名称、课程大作业、用户 × 商家 × 骑手三方视角。
- Visual idea: 外卖平台三方关系图，中心为推荐系统。
- Layout role and intent: cover。
- Required source images: none。

Slide 2: 为什么外卖推荐不能只看 Recall
- Key points: 推荐会影响订单空间分布；订单进一步影响骑手、ETA、超时和商家曝光；只看命中率会漏掉履约风险。
- Visual idea: “推荐 -> 下单 -> 派单 -> 配送状态更新”的链路。
- Layout role and intent: problem framing。
- Required source images: none。

Slide 3: 数据来源与可复现边界
- Key points: TRD 来自 Zenodo；包含用户、商家、菜品、订单和测试标签；骑手数据为固定 seed 合成 proxy；不下载 1.8GB graph.bin。
- Visual idea: 数据表关系图和来源引用条。
- Layout role and intent: data evidence。
- Required source images: none。

Slide 4: 系统总体架构
- Key points: 数据处理、召回排序、三方重排、骑手匹配、动态仿真、指标评估；所有模块可由 Makefile 重复执行。
- Visual idea: 横向流水线架构图。
- Layout role and intent: architecture。
- Required source images: none。

Slide 5: 推荐算法与三方重排
- Key points: 七个默认模型覆盖热门、矩阵分解、用户画像、序列排序、学习排序、三方重排和训练期 Session/SPU 增强；解释只引用真实参与打分的特征。
- Visual idea: 打分公式拆解和模型对照表。
- Layout role and intent: method comparison。
- Required source images: none。

Slide 6: 离线推荐指标结果
- Key points: Seq-Tuned 的 Recall@20 = 0.4675、NDCG@20 = 0.3652，为离线准确率前沿，ExploreRecall@20 = 0.1246 也最高；LightGBM-LTR 的 Recall@20 = 0.4424、Coverage@20 = 0.4345、Exposure Gini = 0.7942，展示学习排序和覆盖改善；Seq-xQuAD-Tripartite 的 Recall@20 = 0.4180、NDCG@20 = 0.3439，牺牲一部分离线准确性换取系统级约束。
- Visual idea: 指标柱状图 + trade-off 散点图。
- Layout role and intent: data evidence。
- Required source images:
  - Main evidence figure; strict input asset after experiment generation; preserve labels and values.

    ![Offline Recall](../../outputs/figures/offline_recall20.png)

  - Accuracy-fairness trade-off; strict input asset after experiment generation; preserve axes and legends.

    ![Tradeoff](../../outputs/figures/tradeoff_ndcg_gini.png)

Slide 7: 动态履约仿真设计
- Key points: 午餐高峰多时间步；推荐列表经过 softmax/MNL 选择产生模拟订单；最近骑手、最小 ETA、负载感知逐单派单与批量最大权匹配；骑手状态随订单更新。
- Visual idea: 离散时间仿真循环图。
- Layout role and intent: process explanation。
- Required source images: none。

Slide 8: 三方策略的系统级指标
- Key points: 七条链路对比；同一批 91 个订单上，批量三方匹配将 Avg ETA 从 49.33 降至 48.32、Timeout Rate 从 0.5604 降至 0.5275；Session-SPU-Tripartite + Greedy 的 Avg ETA = 46.47、Timeout Rate = 0.4941、Platform Utility = 0.4694，为当前系统效用前沿。
- Visual idea: 仿真指标四联图。
- Layout role and intent: data evidence。
- Required source images:
  - Simulation ETA and platform utility figures; strict input assets after experiment generation; preserve values.

    ![Simulation ETA](../../outputs/figures/simulation_avg_eta.png)

    ![Platform Utility](../../outputs/figures/simulation_platform_utility.png)

Slide 9: 案例解释：一次推荐如何兼顾三方
- Key points: 用户画像；Top-K 商家；推荐解释；匹配骑手；预计送达时间；说明推荐不是黑箱。
- Visual idea: 左侧用户画像，中间推荐列表，右侧骑手匹配卡片。
- Layout role and intent: case study。
- Required source images: none。

Slide 10: 结论、局限与改进方向
- Key points: Seq-Tuned 证明序列偏好能提升离线推荐；LightGBM-LTR 补上学习排序与覆盖改善；Seq-xQuAD-Tripartite 把商家公平和履约感知纳入系统效用；局限是骑手为合成 proxy；后续可接入真实派单或图模型。
- Visual idea: 三个结论卡片 + 一个局限说明。
- Layout role and intent: summary。
- Required source images: none。

Slide 11: Q&A
- Key points: 数据可复现、指标可对比、模块可运行。
- Visual idea: 简洁收束页。
- Layout role and intent: Q&A。
- Required source images: none。
