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
        "为什么把这家店排上来",
        "这单派给谁",
        "这单会优先派给谁",
        "这单的路径",
        "午餐高峰怎么变化",
        "推荐有没有命中",
        "这个筛选没命中",
        "从论文思想",
        "Pareto 答辩",
        "指标故事线",
        "方法看板",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in source


def test_demo_map_supports_real_tiles_with_offline_fallback():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "当前订单附近的用户、商家和骑手" in source
    # 真实底图（Scattermap/carto-positron）默认开启，离线时可切回抽象画布。
    assert "go.Scattermap(" in source
    assert "carto-positron" in source
    assert "use_real_map" in source
    assert "离线请关闭" in source


def test_demo_has_tripartite_perspective_panels():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "top_dishes_for_user" in source
    assert "merchant_supply_pressure" in source
    assert "enroute_opportunities" in source
    assert "配送地图回放" in source


def test_recommendation_result_labels_use_plain_chinese():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "TRUTH" not in source
    assert "测试期真实复购" not in source
    assert 'st.success("推荐命中")' in source
