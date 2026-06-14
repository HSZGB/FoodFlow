# FoodFlow 外卖三方推荐与动态履约仿真：NotebookLM 源材料

这份材料是给 NotebookLM 上传使用的压缩版事实源。生成 PPT 时只依据这里和上传的报告、CSV、PNG 图表，不要自行编造实验结果。

## 1. 课程要求

课程大作业展示需要说明：

- 实现过程
- 技术挑战
- 解决方案
- 实验结果分析
- 算法设计思路、算法任务、实验结果、结果分析、代码说明与创新性

硬性要求：

- 需要有指标展示，不要只有结果或效果展示。
- 要讲好故事。
- 要说明数据集来源。
- 需要做出 PPT。

## 2. 项目一句话

FoodFlow 是一个面向外卖平台的三方推荐与动态履约仿真系统。它先在公开外卖订单数据上做用户到商家的 Top-K 推荐，再把推荐结果转化为模拟订单，进入骑手匹配和午餐高峰动态仿真，最后同时评价用户命中、商家曝光公平、骑手负载与平台履约效率。

核心观点：外卖推荐不能只看 Recall 或 NDCG。推荐列表会改变订单空间分布，订单又影响骑手调度、ETA、超时率和商家曝光。因此 FoodFlow 的主线是“先证明推荐有效，再证明三方重排能用较小准确率代价换取更好的履约和平台效用”。

## 3. 三方推荐如何体现

FoodFlow 的“三方”不是口号，而是三个对象都进入了算法与评价：

| 参与方 | 推荐/决策对象 | 本项目中的体现 | 指标 |
|---|---|---|---|
| 用户 | 给用户推荐商家/菜品 | Top-K 商家推荐，考虑复购、品类偏好、价格匹配、时段偏好和商家质量 | Recall@K、NDCG@K、MRR@K、HitRate@K |
| 商家 | 商家被推荐给合适用户 | 三方重排加入曝光公平分，缓解头部商家垄断，让长尾商家获得合理曝光 | Coverage、Long-tail Exposure、Exposure Gini |
| 骑手 | 订单推荐/匹配给骑手 | 用户下单后，将订单分配给候选骑手，比较最近骑手、最小 ETA、负载感知策略 | Avg ETA、Timeout Rate、Rider Load Std、Platform Utility |

答辩时可以这样解释：用户侧是“把商家推荐给用户”；商家侧是“把曝光机会推荐给更需要补偿且质量合格的商家”；骑手侧是“把订单推荐/匹配给最适合承接的骑手”。

## 4. 数据来源与边界

主数据集为 Takeout Recommendation Dataset (TRD)。

- DOI：`10.5281/zenodo.8025855`
- URL：`https://zenodo.org/records/8025855`
- License：CC-BY-4.0
- 来源：美团外卖北京 11 个商圈
- 时间：2021-03-01 至 2021-03-28
- 使用文件：`users.txt`、`pois.txt`、`spus.txt`、`orders_train.txt`、`orders_test_poi.txt`、`orders_poi_test_label.txt`
- 不下载约 1.8GB 的 `graph.bin`，因为核心实现不依赖 DGL 图文件

TRD 不包含完整骑手状态和真实派单记录。因此 FoodFlow 使用固定随机种子合成骑手位置、在线状态、负载、可靠性和收入。合成骑手只用于履约可行性的 proxy 仿真，不能描述成真实骑手数据。

## 5. 系统流程

系统模块：

1. 数据下载：下载 TRD txt 文件并跳过 `graph.bin`。
2. 数据处理：清洗用户、商家、菜品、训练订单、测试订单和测试标签。
3. 特征工程：生成用户画像、商家画像、菜品/品类特征、复购特征和时段特征。
4. 离线推荐：默认评估 Popular、BPR-MF、UserOnly、Seq-Tuned、Seq-xQuAD-Tripartite 五个代表策略。
5. 三方重排：在用户偏好外加入列表级覆盖、商家公平、ETA 和供给分。
6. 动态履约仿真：默认比较 Popular + Nearest、UserOnly + MinETA、Seq-Tuned + MinETA、Seq-xQuAD-Tripartite 四条链路。
7. 指标与图表：输出 CSV、PNG 图表、报告和 Streamlit demo。

可重复执行接口：

- `make preprocess`
- `make eval`
- `make simulate`
- `make figures`
- `make report`
- `make smoke`
- `make test`
- `make demo`

## 6. 推荐算法

答辩主线只展开 5 个代表策略：

- `Popular`：按训练订单热门商家排序，作为朴素热度基线。
- `BPR-MF`：轻量矩阵分解排序基线，用来代表传统隐式反馈排序。
- `UserOnly`：用户偏好排序，特征包括品类偏好、复购、价格匹配、时段热度、商家质量、新颖性。
- `Seq-Tuned`：在同一批可解释序列特征上提高复购、商家转移和品类偏好权重，是当前离线准确率最强策略。
- `Seq-xQuAD-Tripartite`：把列表级覆盖、商家公平、ETA 和供给约束放进同一个重排器，是当前仿真平台效用最强策略。

项目代码中保留了一些历史消融类，方便追溯实验过程；PPT 不需要逐一展开，避免主线臃肿。

三方重排打分：

```text
score = user_preference
      + merchant_fairness
      + eta_score
      + supply_score
      + list_coverage_gain
```

其中：

- `user_preference`：用户品类偏好、复购、价格匹配、时段偏好、商家质量、新颖性。
- `merchant_fairness`：由商家历史订单热度和质量构成，偏向给低曝光但质量较好的商家补偿。
- `eta_score`：估计用户到商家的履约时间，ETA 越低得分越高。
- `supply_score`：商家配送评分与订单压力的供给可行性 proxy。
- `list_coverage_gain`：列表级覆盖增益，避免推荐列表过度集中。

可解释输出只引用真实参与打分的特征，例如复购、品类偏好、价格匹配、商家评分、ETA、曝光补偿。

## 7. 动态履约仿真

默认仿真策略：

- `Popular + Nearest`：热门推荐 + 最近骑手，作为朴素履约基线。
- `UserOnly + MinETA`：用户偏好推荐 + 最小 ETA 骑手。
- `Seq-Tuned + MinETA`：短序列推荐 + 最小 ETA 骑手。
- `Seq-xQuAD-Tripartite`：列表级三方重排 + 负载感知骑手。

仿真流程：

1. 午餐高峰多时间步生成用户请求。
2. 推荐器给每个用户生成 Top-K 商家列表。
3. 用户选择模型优先选择真实测试标签中出现的商家，也允许按排序概率选择。
4. 生成订单后，从当前可用骑手中选择候选。
5. 骑手策略包括最近骑手、最小 ETA、负载感知。
6. 完成派单后更新骑手位置、负载、可用时间、收入和接单数。

骑手侧 ETA 估计包括骑手到店距离、商家到用户距离、出餐时间、高峰期惩罚、骑手当前负载和等待时间。

## 8. 离线推荐指标结果

推荐侧指标：Recall@K、NDCG@K、MRR@K、HitRate@K。

商家侧扩展指标：Coverage@20、Exposure Gini、Long-tail Exposure@20。

校准指标：CategoryJSD@20，衡量推荐列表品类分布与用户历史品类分布的 Jensen-Shannon divergence，越低表示越贴近用户长期品类偏好。

关键结果以 `results/offline_metrics.csv` 为准。当前核心结论：

- `Seq-Tuned` 是离线命中最强策略，Recall@20 = `0.4675`，NDCG@20 = `0.3652`，HitRate@20 = `0.6267`。
- `UserOnly` Recall@20 = `0.4287`，NDCG@20 = `0.3423`，说明用户画像特征已经明显优于热门基线。
- `Seq-xQuAD-Tripartite` Recall@20 = `0.4180`，NDCG@20 = `0.3440`，不是追求单一 Recall 最大，而是在准确性、商家曝光和履约之间折中。
- `BPR-MF` 和 `Popular` 用于证明传统基线和朴素热度基线的差距。

因为三方策略还要考虑商家公平和履约，不能只用 Recall 判断优劣。

## 9. 动态履约仿真指标结果

履约指标：completed_orders、avg_eta、timeout_rate、on_time_rate、rider_load_std、merchant_exposure_gini、user_satisfaction、platform_utility。

关键结果以 `results/simulation_metrics.csv` 为准。当前核心结论：

- `Popular + Nearest`：Avg ETA = `84.71`，Timeout Rate = `0.7903`，Utility = `0.3365`。
- `UserOnly + MinETA`：Avg ETA = `55.33`，Timeout Rate = `0.6701`，Utility = `0.4831`。
- `Seq-Tuned + MinETA`：Avg ETA = `54.74`，Timeout Rate = `0.7320`，Utility = `0.4664`。
- `Seq-xQuAD-Tripartite`：Avg ETA = `50.33`，Timeout Rate = `0.5319`，Utility = `0.5264`。

结果解读：

- 只看热门推荐和最近骑手会造成较高 ETA 和超时率。
- `UserOnly + MinETA` 显著降低 ETA，说明订单推荐给骑手时需要考虑 ETA。
- `Seq-xQuAD-Tripartite` 平均 ETA 最低、超时率最低、平台综合效用最高。
- 因此项目结论不是“单一模型在所有指标上最大”，而是“多方重排改善系统级结果”。

## 10. 平台效用定义

平台效用是课程项目中的解释性综合指标，用来把多主体目标合并展示。它考虑：

- 用户满意度
- 准时率
- 完成率
- 商家曝光公平
- 骑手负载均衡
- 超时惩罚

注意：平台效用权重不是工业生产参数，报告中应说明它是课程实验中的可解释设置。

## 11. PPT 工作流要求

根据 Codex PPT 技能，图片页 PPT 的标准流程需要审批门禁：确认大纲、确认视觉风格、确认图片后端、生成并确认 1 页样张，然后再生成整套图片页和讲稿。当前 NotebookLM 方案是备用生成路径：请先用这里的提示词产出 11 页内容、视觉布局和讲稿，再由 PPT 工具或 NotebookLM 生成页面。

`Image to Editable PPT` 技能不用于从零创作本答辩 PPT。它只适用于后续已有图片页、PDF 或截图后，需要把它们重建成对象级可编辑 PPT 的场景；那时应按 `editppt prepare -> page worker -> record -> finalize` 的流程处理。

## 12. 答辩故事线

推荐 PPT 的主线：

1. 外卖推荐不是普通电商推荐，因为推荐会影响空间订单分布和配送压力。
2. 使用 TRD 公开数据先完成用户-商家 Top-K 推荐，并用标准指标证明推荐有效。
3. 加入商家公平、ETA 和供给分，构成三方重排。
4. 把推荐结果接入动态履约仿真，比较不同派单策略。
5. 实验显示 Seq-xQuAD-Tripartite 牺牲一部分离线准确性，但换来更低 ETA、更低超时率和更高平台效用。
6. 局限是骑手数据为合成 proxy，后续可以接入真实派单数据或更复杂图模型。

## 13. PPT 页数与结构

建议 11 页：

1. 封面：FoodFlow 外卖三方推荐与动态履约仿真系统。
2. 问题：为什么外卖推荐不能只看 Recall。
3. 数据：TRD 数据来源与可复现边界。
4. 架构：系统总体架构。
5. 方法：推荐算法与三方重排。
6. 实验：离线推荐指标结果。
7. 仿真：动态履约仿真设计。
8. 结果：三方策略的系统级指标。
9. 案例：一次推荐如何兼顾三方。
10. 总结：结论、局限与答辩亮点。
11. Q&A。
