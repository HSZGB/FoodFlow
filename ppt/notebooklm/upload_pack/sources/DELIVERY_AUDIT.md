# FoodFlow 交付验收审计

审计日期：2026-06-19

## 课程硬性要求映射

| 要求 | 当前状态 | 证据 |
|---|---|---|
| 指标展示，不要只有效果图 | 已满足 | `report/实验报告.md` 第 4、5 节；`outputs/results/offline_metrics.csv`；`outputs/results/simulation_metrics.csv`；`outputs/figures/*.png` 共 18 张；离线表含 RepeatRecall/ExploreRecall 分段指标 |
| 讲好故事 | 已满足 | `README.md` 的项目主线；`report/实验报告.md` 第 3、7 节；`ppt/FoodFlow/outline.md` 的 11 页答辩结构 |
| 说明数据集来源 | 已满足 | `docs/DATA_SOURCE.md`；`docs/DATA_AUDIT.md`；`outputs/results/data_audit.json`；报告第 2 节 |
| 需要做出 PPT | 部分满足，等待审批门禁 | `ppt/FoodFlow/outline.md`、`ppt/FoodFlow/outline_approval.md`、`ppt/notebooklm/upload_pack/` 已准备；最终图片页和 `.pptx` 需先批准大纲、风格、图片后端和样张 |

路线覆盖另见 `docs/REPORT_ROUTE_COVERAGE.md`，该文件逐项对照最新版调研报告中的 P0-P6 优化路线，避免把历史消融或展望能力误写成正式主线。

## 关键实验证据

- 数据审计显示必需 TRD 原始文件齐全，训练集处理模式为 `full`，原始训练订单和处理后训练订单均为 `1,068,495` 条。
- 离线评估默认包含 7 个模型：Popular、BPR-MF、UserOnly、Seq-Tuned、LightGBM/Logistic-LTR、Seq-xQuAD-Tripartite、Session-SPU-Tripartite，并补充 RepeatRecall/ExploreRecall。
- 动态履约仿真默认包含 7 条链路，包括学习排序、三方逐单/批量匹配和 Session-SPU 增强策略。
- 当前核心结论：`Seq-Tuned` 是离线准确率前沿；批量三方匹配相较逐单版本降低 ETA 和超时率；`Session-SPU-Tripartite + Greedy` 取得当前最高平台效用。

## 已验证命令

```bash
./.venv/bin/python -m pytest -q
```

结果：`24 passed`。

```bash
test -s report/实验报告.md \
  && test -s outputs/results/offline_metrics.csv \
  && test -s outputs/results/simulation_metrics.csv \
  && test -s outputs/results/data_audit.json \
  && test -s ppt/notebooklm/upload_pack/UPLOAD_INDEX.md
```

结果：`ok`。

```bash
find outputs/figures -maxdepth 1 -name '*.png' | sort | wc -l
```

结果：`18`。

```bash
find ppt/notebooklm/upload_pack -type f | sort | wc -l
```

结果：`38`。

## 剩余门禁

`codex-ppt` 正式生成 PPTX 前仍需按顺序确认：

1. 批准 `ppt/FoodFlow/outline.md` 的 11 页大纲。
2. 确认统一视觉风格。
3. 确认图片生成后端。
4. 生成并批准 1 页样张。
5. 生成全套图片页、讲稿和 `.pptx`。

在这些门禁完成前，不应创建最终 `deck_spec.json`、`speech.md`、slide prompt jobs、最终 slide images 或 `.pptx`。
