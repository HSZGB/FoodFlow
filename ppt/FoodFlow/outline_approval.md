# FoodFlow PPT 大纲审批摘要

状态：草稿待审批。当前没有生成 `deck_spec.json`、`speech.md`、slide prompt jobs、最终 slide images 或 `.pptx`。

## 大纲文件

- 草稿路径：`ppt/FoodFlow/outline.md`
- 页数：11 页
- 目标：中文课程大作业答辩，讲清“用户推荐 - 商家曝光 - 骑手履约”的三方闭环。
- 主线：六个离线推荐模型 + 六条动态履约仿真链路。

## 11 页结构

1. FoodFlow：外卖三方推荐与动态履约仿真系统
2. 为什么外卖推荐不能只看 Recall
3. 数据来源与可复现边界
4. 系统总体架构
5. 推荐算法与三方重排
6. 离线推荐指标结果
7. 动态履约仿真设计
8. 三方策略的系统级指标
9. 案例解释：一次推荐如何兼顾三方
10. 结论、局限与改进方向
11. Q&A

## 严格输入图片映射

Slide 6 使用正式离线结果图：

- `outputs/figures/offline_recall20.png`
- `outputs/figures/tradeoff_ndcg_gini.png`

Slide 8 使用正式履约仿真图：

- `outputs/figures/simulation_avg_eta.png`
- `outputs/figures/simulation_platform_utility.png`

生成图片页时必须保留图表数据、坐标轴、图例和数值口径；不得把历史消融数字混入正式主线。

## 进入下一阶段前需确认

- 是否批准当前 11 页大纲。
- 是否保持 16:9 中文科研答辩风。
- 是否允许使用内置图片生成工具生成 1 页样张。

根据 `codex-ppt` 工作流，只有大纲获批后才能进入视觉风格确认；只有风格和图片后端获批后才能生成 1 页样张；只有样张获批后才能生成全套图片页、讲稿和 `.pptx`。
