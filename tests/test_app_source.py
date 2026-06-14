from pathlib import Path


def test_demo_does_not_use_raw_html_cards():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "unsafe_allow_html" not in source
    assert "<div" not in source
    assert "<span" not in source
    assert "<br" not in source
    assert "<extra" not in source
    assert "ff-control-card" not in source


def test_demo_copy_is_presentation_facing():
    source = Path("app.py").read_text(encoding="utf-8")
    forbidden_phrases = [
        "课堂上可以",
        "课堂演示",
        "结果怎么读",
        "项目里到底",
        "怎么讲",
        "讲清楚为什么",
        "从论文思想",
        "Pareto 答辩",
        "指标故事线",
        "方法看板",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in source
