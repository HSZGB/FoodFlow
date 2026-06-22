# FoodFlow 答辩 PPT 大纲

目标：围绕 `docs/RECOMMENDER_FORMULAS.md` 中的推荐方法、三方重排和履约仿真，形成一套可直接制作 PPT 的答辩大纲。主线是：外卖推荐不能只看用户点击，还要同时解释商家曝光、履约 ETA 和骑手负载。

## Slide 1: FoodFlow：外卖三方推荐与动态履约仿真系统
- Key points: 项目名称；用户、商家、骑手三方视角；从推荐列表延伸到订单履约。
- Speaker intent: 先定义 FoodFlow 不是单纯 Top-K 推荐 demo，而是“推荐 - 选店 - 派单 - 仿真评估”的闭环系统。
- Visual idea: 三方关系图，用户、商家、骑手围绕平台推荐系统连接。
- Layout role and intent: cover。
- Required source images: none。

## Slide 2: 为什么外卖推荐不能只看 Recall
- Key points: 推荐商家会改变订单空间分布；更远的商家会改变取餐距离和 ETA；曝光集中会影响商家公平；骑手负载会影响超时。
- Speaker intent: 引出系统级目标：用户满意、商家曝光、履约效率和骑手负载需要一起看。
- Visual idea: “推荐列表 -> 用户选店 -> 骑手派单 -> ETA/超时/负载/曝光”的链路图。
- Layout role and intent: problem framing。
- Required source images: none。

## Slide 3: 数据来源与可复现边界
- Key points: 数据来自 Takeout Recommendation Dataset (TRD)，使用用户、商家、菜品、训练订单、测试订单和测试标签；骑手状态、骑手轨迹和派单记录在公开数据中不存在，因此用固定 seed 合成 proxy。
- Speaker intent: 明确哪些是真实数据，哪些是仿真变量，避免把合成骑手误讲成真实轨迹。
- Visual idea: 数据表关系图：users、pois、orders、spus、session、riders proxy。
- Layout role and intent: data evidence。
- Required source images: none。

## Slide 4: 系统总体架构
- Key points: 数据预处理；候选召回与排序；三方重排；用户选择仿真；骑手派单；离线指标和动态仿真指标。
- Speaker intent: 让听众知道后续算法都落在同一条工程链路里。
- Visual idea: 横向流水线：TRD -> Recommenders -> Rerank -> MNL Choice -> Rider Assignment -> Metrics。
- Layout role and intent: architecture。
- Required source images: none。

## Slide 5: 当前默认推荐器全景
- Key points: 默认评估包含 7 个模型：Popular、BPR-MF、UserOnly、LightGBM/Logistic-LTR、Seq-Tuned、Seq-xQuAD-Tripartite、Session-SPU-Tripartite。
- Speaker intent: 先把模型族谱讲清楚：从基线、个性化、序列、学习排序，到三方重排和会话/菜品增强。
- Visual idea: 模型对照表，列为“模型、核心信号、是否个性化、是否含履约、解释重点”。
- Layout role and intent: method overview。
- Required source images: none。

## Slide 6: 基线模型：Popular 与 BPR-MF
- Key points: Popular 用商家历史订单量排序，分数为 `score_pop(u,m)=c_m`；BPR-MF 用隐式反馈矩阵分解，预测分数为用户向量和商家向量点积。
- Speaker intent: Popular 是热门榜对照，BPR-MF 是传统协同过滤对照，用来说明后续模型不是只和随机或热门榜比较。
- Formula callout:
  - `score_pop(u,m)=c_m`
  - `x_hat_ui = p_u^T q_i`
  - BPR 优化 `log sigma(x_hat_ui - x_hat_uj) - lambda ||Theta||^2`
- Visual idea: 左右两栏：热门计数排序 vs 用户/商家向量空间。
- Layout role and intent: baseline explanation。
- Required source images: none。

## Slide 7: UserOnly：可解释用户画像排序
- Key points: 使用品类偏好、复购、价格匹配、时段热度、商家质量和新颖性；复购权重最高，符合外卖常点老店的行为。
- Speaker intent: 说明第一类可解释推荐如何把用户历史转成可展示理由。
- Formula callout:
  - `repeat(u,m)=log(1+cnt_um)/log 5`
  - `score_user = 0.20 category + 0.52 repeat + 0.10 price + 0.06 period + 0.07 quality + 0.05 novelty`
- Visual idea: 用户画像雷达图或分数组成条形图。
- Layout role and intent: explainable model。
- Required source images: none。

## Slide 8: Seq-Tuned：外卖短序列偏好
- Key points: 组合快慢最近性、复购、商家转移、品类、热度和质量；权重固定、可解释、速度快；离线准确率表现强。
- Speaker intent: 讲清楚 Seq-Tuned 为什么在外卖场景有效：最近点过什么、常复购什么、上一个商家可能转移到哪里。
- Formula callout:
  - `r_fast=e^{-age/6}`, `r_slow=e^{-age/12}`
  - `trans(u,m)=max_k 0.85^k T(s_{t-k},m)`
  - 权重：repeat 0.412158、transition 0.247023、fast_recency 0.142601、slow_recency 0.093624、category 0.091250。
- Visual idea: 用户最近 5 单序列 -> 候选商家得分。
- Layout role and intent: sequence method。
- Required source images: none。

## Slide 9: LightGBM-LTR：从人工权重到学习排序
- Key points: LightGBM-LTR 复用 Seq-Tuned 的候选和 7 个序列特征，但不用硬编码权重；用 LambdaRank 学习排序函数；缺少 LightGBM 时回退 Logistic-LTR。
- Speaker intent: 说明项目既有规则可解释基线，也引入学习排序模型进行对比。
- Formula callout:
  - `x_um=[r_fast,r_slow,repeat,transition,category,popularity,quality]`
  - `s_hat_um=f_LGBM(x_um)`
  - 训练目标：`objective=lambdarank`, `metric=ndcg`
- Visual idea: 特征向量进入 LightGBM 排序器，输出 Top-K。
- Layout role and intent: learned ranking。
- Required source images: none。

## Slide 10: Seq-xQuAD-Tripartite：用户、商家、履约的三方重排
- Key points: 在序列用户分基础上加入商家公平、ETA 和供给可行性；候选内 min-max 归一化；xQuAD 在列表层面增加品类覆盖和长尾曝光。
- Speaker intent: 这是项目从“用户推荐”走向“三方推荐”的关键页，重点解释为什么三方重排会牺牲部分离线命中但改善系统约束。
- Formula callout:
  - `fair(m)=0.75(1-pop_norm(m))+0.25 quality_norm(m)`
  - `eta_score=1-min(ETA_hat/70,1)`
  - `supply=0.6 delivery_score/5 + 0.4/(1+log(1+count_m))`
  - `score_tri=(0.93 user + 0.025 fair + 0.03 eta + 0.015 supply)/sum(w)`
  - xQuAD: `0.84 relevance + 0.12 category_gain + 0.04 tail`
- Visual idea: 四个分量汇入三方重排器，再经过 xQuAD 逐步选商家。
- Layout role and intent: core method。
- Required source images: none。

## Slide 11: Session-SPU-Tripartite：会话点击与菜品画像增强
- Key points: 复用三方重排和 xQuAD；额外使用训练期 `orders_poi_session` 点击序列和 `orders_spu_train` 菜品 SPU 类目；测试期信息不进入模型，避免泄漏。
- Speaker intent: 讲清楚 Session-SPU 的新增价值：召回侧加入“下单前看过的店”和“菜品类目相近的店”，解释侧保留 session/SPU 证据。
- Formula callout:
  - `session(u,m)=exp(-(rank_u(m)-1)/4)` if clicked
  - `spu(u,m)=overlap(user_spu_categories, merchant_spu_categories)`
  - 权重：user 0.86、fair 0.025、eta 0.03、supply 0.015、session 0.055、spu 0.015。
- Visual idea: session 点击序列和 SPU 菜品类目共同扩展候选池。
- Layout role and intent: model enhancement。
- Required source images: none。

## Slide 12: 轻量 KG 路径解释
- Key points: 项目没有训练 LightGCN/KGAT，而是把结构化字段转为可解释路径；路径包括复购、品类、区域和价格段。
- Speaker intent: 说明 demo 里的解释不是空泛文案，而是引用真实参与推荐或诊断的字段。
- KG paths:
  - `user --ordered_poi--> poi`
  - `user --prefers_category--> category <--has_category-- poi`
  - `user --orders_in_area--> area <--located_in_area-- poi`
  - `user --has_price_range--> price <--has_price_range-- poi`
- Visual idea: 小型知识路径图。
- Layout role and intent: explainability。
- Required source images: none。

## Slide 13: 动态履约仿真：从推荐列表到订单流
- Key points: 仿真策略包含 Popular + Nearest、UserOnly + MinETA、Seq-Tuned + MinETA、LightGBM-LTR + MinETA、Seq-xQuAD-Tripartite + Greedy/Batch、Session-SPU-Tripartite + Greedy/Batch；用户选择使用 softmax/MNL。
- Speaker intent: 说明仿真不是只把 Top1 当订单，而是用推荐分、排序位置和不下单选项生成订单流。
- Formula callout:
  - `P(choice=m|u,t)=exp(V_umt)/(exp(V_none)+sum_j exp(V_ujt))`
- Visual idea: 每个时间步抽用户 -> 推荐 Top-K -> MNL 选择 -> 进入派单池。
- Layout role and intent: simulation process。
- Required source images: none。

## Slide 14: 骑手策略：最近、最小 ETA 与负载感知
- Key points: 骑手侧包含距离、速度、服务半径、接单率、可靠性、当前负载和可用时间；负载感知不是只看最近距离。
- Speaker intent: 解释 demo 里“骑手到商家虚线”和“商家到用户配送段”的意义，回应 load-aware 为什么必须考虑商家位置。
- Formula callout:
  - `ETA = wait + prep + peak + dist(rider,merchant)/speed*60 + dist(merchant,user)/(1.08 speed)*60 + 5 load`
  - `score_rider = 0.50 eta_score + 0.20 reliability + 0.15 load_score + 0.15 accept_prob`
- Visual idea: 用户、商家、骑手三点地图；虚线为取餐段，实线为配送段。
- Layout role and intent: fulfillment method。
- Required source images: none。

## Slide 15: Batch 二分图匹配：从逐单贪心到整体最优
- Key points: Batch 策略把同一时间步内订单集合 `O` 与骑手容量槽位集合 `S` 构造成二分图；一个骑手可按剩余容量展开多个槽位；用最大权匹配减少局部最优。
- Speaker intent: 直观说明 Batch 不是“当前订单找当前最好骑手”，而是“这一批订单整体分配”。
- Formula callout:
  - `W_os=score_rider(o,r(s))-0.20 timeout_risk`
  - `max sum_o sum_s W_os x_os`
  - constraints: 每个订单最多匹配一个槽位，每个槽位最多接一个订单。
- Visual idea: 小地图散点图：蓝色用户订单、紫色骑手槽位、绿色商家，实线为订单匹配，虚线为取餐段。
- Layout role and intent: assignment algorithm。
- Required source images: none。

## Slide 16: 离线推荐指标结果
- Key points: Seq-Tuned 取得离线准确率前沿；LightGBM-LTR 展示学习排序与覆盖改善；三方重排在 Recall 上有所让步，但引入商家公平、ETA 和供给约束。
- Speaker intent: 讲清楚离线指标用于比较推荐模型本身，不包含后缀派单策略。
- Metrics to cite:
  - Seq-Tuned: Recall@20 = 0.4675, NDCG@20 = 0.3652。
  - LightGBM-LTR: Recall@20 = 0.4424, Coverage@20 = 0.4345。
  - Seq-xQuAD-Tripartite: Recall@20 = 0.4180, NDCG@20 = 0.3439。
- Visual idea: Recall/NDCG 柱状图 + 覆盖/曝光权衡图。
- Layout role and intent: data evidence。
- Required source images:
  - ![Offline Recall](../../outputs/figures/offline_recall20.png)
  - ![Offline NDCG](../../outputs/figures/offline_ndcg20.png)
  - ![Tradeoff NDCG Gini](../../outputs/figures/tradeoff_ndcg_gini.png)

## Slide 17: 动态履约仿真结果
- Key points: 端到端指标比较推荐策略进入履约后的表现；Batch 用于比较同一推荐模型下逐单与批量匹配；Session-SPU-Tripartite 在平台综合效用上表现突出。
- Speaker intent: 区分“推荐准确率最优”和“系统效用最优”不是同一个问题。
- Metrics to cite:
  - Seq-xQuAD-Tripartite + Greedy: Avg ETA 约 49.33, Timeout Rate 约 0.5604。
  - Seq-xQuAD-Tripartite + Batch: Avg ETA 约 48.32, Timeout Rate 约 0.5275。
  - Session-SPU-Tripartite + Greedy/Batch 为系统效用前沿，具体数值以 `outputs/results/simulation_metrics.csv` 为准。
- Visual idea: ETA、超时率、平台综合分三联图。
- Layout role and intent: system evaluation。
- Required source images:
  - ![Simulation ETA](../../outputs/figures/simulation_avg_eta.png)
  - ![Simulation Timeout](../../outputs/figures/simulation_timeout_rate.png)
  - ![Platform Utility](../../outputs/figures/simulation_platform_utility.png)

## Slide 18: Demo 展示脚本
- Key points: 推荐工作台展示用户画像、Top-K 推荐、模型分数组成和解释；高峰回放展示不同策略下 ETA、超时率、相对位置地图；Batch 地图匹配展示订单与骑手槽位的整体匹配。
- Speaker intent: 给答辩现场一个可执行的 demo 路线，避免只展示静态图。
- Demo path:
  1. 选择用户和推荐策略，展示 UserOnly、Seq-Tuned、LTR、Seq-xQuAD、Session-SPU 的差异。
  2. 选择 Session-SPU-Tripartite，强调会话点击和 SPU 菜品理由。
  3. 运行高峰回放，切到 Session-SPU-Tripartite + Batch，展示地图匹配图。
- Visual idea: demo 截图占主视觉，旁边列步骤。
- Layout role and intent: live demo guide。
- Required source images: optional screenshot from running Streamlit demo。

## Slide 19: 关键结论
- Key points: Seq-Tuned 证明短序列和复购对外卖推荐很强；LightGBM-LTR 用学习排序补充规则权重；Seq-xQuAD-Tripartite 把公平、ETA 和供给纳入推荐；Session-SPU 进一步利用训练期会话与菜品信号；Batch 匹配改善逐单派单的局部最优问题。
- Speaker intent: 用 5 条结论收束，不把所有指标混成一个“唯一最好”。
- Visual idea: 五个结论卡片。
- Layout role and intent: summary。
- Required source images: none。

## Slide 20: 局限与后续改进
- Key points: TRD 没有真实骑手轨迹和真实派单记录；当前 ETA 和骑手状态是启发式 proxy；离线评估默认不是全测试用户全量评估；后续可接入真实配送日志、校准 ETA、扩展图模型或在线 A/B。
- Speaker intent: 主动说明边界，避免过度声明。
- Visual idea: “当前边界 -> 下一步”的路线图。
- Layout role and intent: limitations and future work。
- Required source images: none。

## Slide 21: Q&A
- Key points: 数据可复现、指标可复算、demo 可运行、算法解释可追溯。
- Speaker intent: 简洁收束。
- Visual idea: 项目名称 + Git/Makefile/Streamlit demo 提示。
- Layout role and intent: Q&A。
- Required source images: none。
