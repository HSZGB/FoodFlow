from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "ppt" / "notebooklm"
PACK_DIR = NOTEBOOK_DIR / "upload_pack"

PROMPT_FILES = [
    NOTEBOOK_DIR / "source_pack.md",
    NOTEBOOK_DIR / "notebooklm_master_prompt.md",
    NOTEBOOK_DIR / "slide_by_slide_prompt.md",
    NOTEBOOK_DIR / "assets_manifest.md",
    NOTEBOOK_DIR / "quality_checklist.md",
]

SOURCE_FILES = [
    (ROOT / "report" / "实验报告.md", "实验报告.md"),
    (ROOT / "docs" / "DATA_SOURCE.md", "DATA_SOURCE.md"),
    (ROOT / "docs" / "DATA_AUDIT.md", "DATA_AUDIT.md"),
    (ROOT / "docs" / "IMPROVEMENT_RESEARCH.md", "IMPROVEMENT_RESEARCH.md"),
    (ROOT / "docs" / "WEIGHT_SENSITIVITY.md", "WEIGHT_SENSITIVITY.md"),
    (ROOT / "ppt" / "FoodFlow" / "outline.md", "ppt_outline.md"),
    (ROOT / "作业要求.md", "作业要求.md"),
    (ROOT / "硬性要求.md", "硬性要求.md"),
]

OPTIONAL_SOURCE_FILES = [
    (ROOT / "ppt" / "FoodFlow" / "drafts" / "slide_07_sample_prompt.md", "slide_07_local_draft_prompt.md"),
]

RESULT_FILES = [
    ROOT / "outputs" / "results" / "offline_metrics.csv",
    ROOT / "outputs" / "results" / "simulation_metrics.csv",
    ROOT / "outputs" / "results" / "tripartite_frontier.csv",
    ROOT / "outputs" / "results" / "data_audit.json",
]

FIGURE_FILES = [
    ROOT / "outputs" / "figures" / "offline_recall20.png",
    ROOT / "outputs" / "figures" / "offline_ndcg20.png",
    ROOT / "outputs" / "figures" / "offline_coverage20.png",
    ROOT / "outputs" / "figures" / "offline_exposure_gini.png",
    ROOT / "outputs" / "figures" / "tradeoff_ndcg_gini.png",
    ROOT / "outputs" / "figures" / "tradeoff_recall_coverage.png",
    ROOT / "outputs" / "figures" / "simulation_avg_eta.png",
    ROOT / "outputs" / "figures" / "simulation_timeout_rate.png",
    ROOT / "outputs" / "figures" / "simulation_rider_load_std.png",
    ROOT / "outputs" / "figures" / "simulation_platform_utility.png",
    ROOT / "outputs" / "figures" / "simulation_exposure_gini.png",
    ROOT / "outputs" / "figures" / "tradeoff_eta_utility.png",
    ROOT / "outputs" / "figures" / "pareto_recall_utility.png",
    ROOT / "outputs" / "figures" / "tripartite_scorecard.png",
]


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required file is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if PACK_DIR.exists():
        shutil.rmtree(PACK_DIR)
    (PACK_DIR / "prompts").mkdir(parents=True)
    (PACK_DIR / "sources").mkdir(parents=True)
    (PACK_DIR / "results").mkdir(parents=True)
    (PACK_DIR / "figures").mkdir(parents=True)

    copied: list[str] = []

    for src in PROMPT_FILES:
        dst = PACK_DIR / "prompts" / src.name
        copy_required(src, dst)
        copied.append(str(dst.relative_to(PACK_DIR)))

    for src, name in SOURCE_FILES:
        dst = PACK_DIR / "sources" / name
        copy_required(src, dst)
        copied.append(str(dst.relative_to(PACK_DIR)))

    for src, name in OPTIONAL_SOURCE_FILES:
        if src.exists():
            dst = PACK_DIR / "sources" / name
            copy_required(src, dst)
            copied.append(str(dst.relative_to(PACK_DIR)))

    for src in RESULT_FILES:
        dst = PACK_DIR / "results" / src.name
        copy_required(src, dst)
        copied.append(str(dst.relative_to(PACK_DIR)))

    for src in FIGURE_FILES:
        dst = PACK_DIR / "figures" / src.name
        copy_required(src, dst)
        copied.append(str(dst.relative_to(PACK_DIR)))

    index = [
        "# FoodFlow NotebookLM Upload Index",
        "",
        "Upload every file in this directory to NotebookLM.",
        "Then paste `prompts/notebooklm_master_prompt.md` as the generation prompt.",
        "",
        "## Files",
        "",
    ]
    index.extend(f"- `{path}`" for path in sorted(copied))
    index.extend(
        [
            "",
            "## Suggested first prompt",
            "",
            "请阅读所有上传资料，并严格按照 `notebooklm_master_prompt.md` 生成 11 页中文答辩 PPT。",
            "",
        ]
    )
    (PACK_DIR / "UPLOAD_INDEX.md").write_text("\n".join(index), encoding="utf-8")
    print(f"Prepared NotebookLM upload pack: {PACK_DIR}")
    print(f"Files copied: {len(copied)}")


if __name__ == "__main__":
    main()
