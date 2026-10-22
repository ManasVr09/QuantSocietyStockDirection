from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_loads_with_exploratory_audit_posture():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert not app.warning
    assert any("Stock direction predictor" in heading.value for heading in app.markdown)
    assert any("seven-stock universe was selected" in text.value for text in app.markdown)
    app.get("button_group")[0].set_value("WMT").run()
    assert not app.exception
    assert any("WMT" in caption.value for caption in app.caption)
    for page in ("Predictions", "Performance", "Features", "Backtest", "Sandbox"):
        app.radio[0].set_value(page).run()
        assert not app.exception


def test_official_threshold_is_present_in_summary_artifact():
    import json

    summary_path = APP.parents[1] / "reports" / "tables" / "dataset_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["validation_declared_champion"]
    assert 0 <= summary["validation_selected_threshold"] <= 1
    assert summary["research_status"] == "exploratory_post_hoc"
