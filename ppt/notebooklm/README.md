# FoodFlow NotebookLM PPT 生成包

这个目录用于把本项目已经完成的报告、指标、图表和 PPT 设计要求整理成 NotebookLM 可以直接使用的素材包。它的目的不是在本地生成图片页，而是让 NotebookLM 或其他在线 PPT 工具按同一套证据和提示词生成答辩 PPT。

## 文件说明

- `source_pack.md`：压缩后的项目事实材料，适合直接上传给 NotebookLM。
- `notebooklm_master_prompt.md`：一键生成整套 11 页 PPT 的总提示词。
- `slide_by_slide_prompt.md`：逐页生成或修订时使用的详细提示词。
- `assets_manifest.md`：图表、CSV、报告、交付验收审计、调研路线覆盖审计、PPT 大纲审批摘要和课程要求文件的上传清单。
- `quality_checklist.md`：生成后逐页检查清单，防止指标、三方逻辑和数据来源被漏掉。

## 一键准备上传包

在项目根目录执行：

```bash
python3 scripts/prepare_notebooklm_pack.py
```

脚本会生成：

```text
ppt/notebooklm/upload_pack/
```

建议把 `upload_pack` 中的 Markdown、CSV 和 PNG 图表全部上传到 NotebookLM。上传后，把 `notebooklm_master_prompt.md` 的内容粘贴给 NotebookLM；如果整套生成不稳定，就改用 `slide_by_slide_prompt.md` 按页生成。

## 生成时的硬性要求

- 生成中文 16:9 答辩 PPT，共 11 页。
- 必须说明 TRD 数据来源、Zenodo DOI、骑手数据为固定 seed 合成 proxy。
- 必须说明当前结果已经通过 `data_audit.json` 证明使用完整 TRD 训练订单。
- 必须展示标准推荐指标：Recall@K、NDCG@K、MRR@K、HitRate@K。
- 必须展示三方扩展指标：Coverage、Long-tail Exposure、Exposure Gini、Avg ETA、Timeout Rate、Rider Load Std、Platform Utility。
- 必须体现三方推荐：用户获得商家推荐，商家获得公平曝光，订单被推荐/匹配给骑手。
- 不要把合成骑手数据描述成真实骑手派单数据。
- 不要把 LightGCN、KGAT 或 DHRD 写成核心已实现方案，它们只能作为调研背景或后续工作。

## 推荐 NotebookLM 流程

1. 新建 NotebookLM 项目。
2. 上传 `upload_pack/prompts/`、`upload_pack/sources/`、`upload_pack/results/` 和 `upload_pack/figures/` 下的文件。
3. 先粘贴 `upload_pack/prompts/notebooklm_master_prompt.md`，要求生成完整 PPT。
4. 如果 NotebookLM 只输出文字大纲，就让它按 `slide_by_slide_prompt.md` 逐页输出“页面内容 + 视觉布局 + 讲稿”。
5. 生成后用 `quality_checklist.md` 检查每页，尤其是第 3、6、8、9 页。

## 与两个 PPT 技能的关系

- `codex-ppt` 的正式流程需要先确认大纲、视觉风格、图片后端和 1 页样张，再生成全套图片页、讲稿和 `.pptx`。当前 NotebookLM 包是因为本地暂时不能生成图片时的备用路径，不跳过这些门禁去伪造本地成品。
- `image-to-editable-ppt` 不用于从零创作本 PPT。它适合后续已经有图片页、PDF、截图或图片式 PPT 后，再通过 `editppt prepare -> page worker -> record -> finalize` 重建对象级可编辑 `.pptx`。
