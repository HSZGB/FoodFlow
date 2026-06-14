from pathlib import Path


def test_demo_does_not_use_raw_html_cards():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "unsafe_allow_html" not in source
    assert "<div" not in source
    assert "<span" not in source
    assert "ff-control-card" not in source
