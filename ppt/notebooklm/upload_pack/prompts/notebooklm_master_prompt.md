# NotebookLM 总提示词：生成 FoodFlow 答辩 PPT

将本文件内容复制到 NotebookLM。复制前请先上传 `upload_pack` 中的 Markdown、CSV 和 PNG 图表。

---

你是一个中文课程大作业答辩 PPT 生成专家。请严格依据我上传的资料，为项目 **FoodFlow：外卖三方推荐与动态履约仿真系统** 生成一套 16:9 中文答辩 PPT。

## 目标

生成 11 页 PPT，用于数据挖掘/推荐系统课程大作业展示。PPT 必须讲清楚：

1. 为什么外卖推荐不能只看 Recall。
2. TRD 数据来源与可复现边界。
3. 用户、商家、骑手三方如何进入推荐与评价。
4. 系统实现流程、推荐算法、三方重排、动态履约仿真。
5. 离线推荐指标和系统级仿真指标。
6. 结论、局限与后续工作。

## 必须遵守的事实

- 主数据集是 Takeout Recommendation Dataset (TRD)，Zenodo DOI 为 `10.5281/zenodo.8025855`。
- TRD 来自美团外卖北京 11 个商圈，时间为 2021-03-01 至 2021-03-28。
- 使用用户、商家、菜品、训练订单、测试订单和测试标签。
- 不下载约 1.8GB 的 `graph.bin`，因为本项目核心实现不依赖 DGL 图文件。
- 骑手位置、在线状态、负载、可靠性和收入是固定 seed 合成 proxy，只用于仿真，不能说成真实骑手数据。
- 答辩主线只展开 5 个代表推荐策略：Popular、BPR-MF、UserOnly、Seq-Tuned、Seq-xQuAD-Tripartite。
- 项目代码中保留历史消融类，PPT 不要逐一展开，避免主线臃肿。
- Seq-Tuned = 复购、商家转移和品类偏好权重增强的短序列推荐。
- Seq-xQuAD-Tripartite = 序列偏好 + 列表级覆盖 + 商家公平 + ETA + 供给分。
- 答辩主线只展开 4 条仿真链路：Popular + Nearest、UserOnly + MinETA、Seq-Tuned + MinETA、Seq-xQuAD-Tripartite。
- 三方推荐必须体现为：
  - 用户侧：给用户推荐商家/菜品。
  - 商家侧：通过公平重排让商家获得更合理曝光。
  - 骑手侧：用户下单后，把订单推荐/匹配给合适骑手。
- Codex PPT 工作流要求：确认大纲、确认视觉风格、确认图片后端、确认 1 页样张后，才能生成全套图片页和讲稿。
- Image to Editable PPT 不用于从零创作本 PPT；它只用于后续把已有图片页、PDF 或截图重建成对象级可编辑 PPT。

## 必须保留的关键数值

离线推荐：

- `UserOnly`：Recall@20 = `0.4287`，NDCG@20 = `0.3423`，HitRate@20 = `0.5733`。
- `Seq-Tuned`：Recall@20 = `0.4675`，NDCG@20 = `0.3652`，HitRate@20 = `0.6267`。
- `Seq-xQuAD-Tripartite`：Recall@20 = `0.4180`，NDCG@20 = `0.3440`，HitRate@20 = `0.5733`。
- `BPR-MF`：Recall@20 = `0.1620`，NDCG@20 = `0.1068`，HitRate@20 = `0.2433`。
- `Popular`：Recall@20 = `0.0470`，NDCG@20 = `0.0210`，HitRate@20 = `0.0900`。

动态履约仿真：

- `Popular + Nearest`：Avg ETA = `84.71`，Timeout Rate = `0.7903`，Platform Utility = `0.3365`。
- `UserOnly + MinETA`：Avg ETA = `55.33`，Timeout Rate = `0.6701`，Platform Utility = `0.4831`。
- `Seq-Tuned + MinETA`：Avg ETA = `54.74`，Timeout Rate = `0.7320`，Platform Utility = `0.4664`。
- `Seq-xQuAD-Tripartite`：Avg ETA = `50.33`，Timeout Rate = `0.5319`，Platform Utility = `0.5264`。

结论必须表达为：Seq-Tuned 是离线推荐指标最强策略；Seq-xQuAD-Tripartite 牺牲一部分离线准确性，但换来更好的履约效率、更低超时风险和最高平台综合效用。

## 视觉风格

采用“科研答辩风”：

- 白色或极浅灰背景。
- 主色使用深学术蓝和研究蓝，可少量使用正式红强调关键结论。
- 图表、流程图、指标卡为主，避免营销风、卡通风和无关装饰。
- 页面信息密度中等偏高，但必须清晰。
- 每页标题明确，正文短句化。
- 不要添加真实机构 logo、水印或无关图片。
- 图表页要优先使用我上传的 PNG 图表，保留图例、坐标轴和模型名称。

## 11 页结构

请按以下结构生成：

1. **FoodFlow：外卖三方推荐与动态履约仿真系统**  
   封面。突出用户 × 商家 × 骑手三方视角。

2. **为什么外卖推荐不能只看 Recall**  
   说明推荐会改变订单空间分布，进而影响 ETA、超时、骑手负载和商家曝光。

3. **数据来源与可复现边界**  
   写明 TRD、Zenodo DOI、北京 11 个商圈、时间范围、使用文件、合成骑手边界。

4. **系统总体架构**  
   展示数据处理 -> 推荐召回排序 -> 三方重排 -> 骑手匹配 -> 动态仿真 -> 指标评估。

5. **推荐算法与三方重排**  
   对比 Popular、BPR-MF、UserOnly、Seq-Tuned、Seq-xQuAD-Tripartite；展示 Seq-xQuAD-Tripartite 的三方重排拆解。

6. **离线推荐指标结果**  
   使用 `offline_recall20.png`、`tradeoff_ndcg_gini.png` 和 `offline_category_jsd20.png`，也可引用 `offline_metrics.csv` 生成小表。说明标准推荐指标、商家曝光指标与品类校准指标。

7. **动态履约仿真设计**  
   展示午餐高峰多时间步、推荐列表产生订单、订单推荐/匹配给骑手、状态更新闭环。

8. **三方策略的系统级指标**  
   使用 `simulation_avg_eta.png`、`simulation_platform_utility.png` 和 `pareto_recall_utility.png`，也可补充超时率和骑手负载。强调 Seq-xQuAD-Tripartite 的系统级优势，同时说明 Seq-Tuned 是离线准确率前沿。

9. **案例解释：一次推荐如何兼顾三方**  
   用一个案例串起用户画像、Top-K 商家推荐、推荐解释、下单、骑手匹配、ETA/超时风险。

10. **结论、局限与答辩亮点**  
    三个结论：推荐有效、三方重排可解释、履约感知提升系统级指标。局限：骑手为合成 proxy，后续接入真实派单或图模型。

11. **Q&A**  
    简洁收束，保留数据可复现、指标可对比、模块可运行三个关键词。

## 输出要求

如果你可以直接生成 PPT，请生成可下载的 PPT 文件。如果不能直接生成 PPT，请输出可以复制到 PPT 工具中的逐页内容，每页包含：

- 页面标题
- 3-5 个页面要点
- 图表或流程图布局
- 需要使用的图片文件名
- 演讲备注，约 80-120 字

不要编造上传资料之外的新实验数据。不要把本项目包装成工业生产系统。不要只写概念，必须展示指标和结果分析。
