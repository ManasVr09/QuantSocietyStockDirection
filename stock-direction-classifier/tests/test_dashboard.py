from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_loads_with_exploratory_audit_posture():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert not app.warning
    assert any("Stock direction predictor" in heading.value for heading in app.markdown)
    app.session_state["research_protocol"] = True
    app.run()
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


def test_overview_projects_predictions_and_never_reads_unused_ledger(monkeypatch):
    import pandas as pd
    import streamlit as st
    st.cache_data.clear()
    reads = []
    original = pd.read_parquet
    def record(path, *args, **kwargs):
        reads.append((Path(path).name, kwargs.get("columns")))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", record)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert not any(name == "trade_ledger.parquet" for name, columns in reads)
    prediction_reads = [columns for name, columns in reads if name == "test_predictions.parquet"]
    assert prediction_reads == [["model", "ticker", "target_date", "probability_up", "correct"]]
    assert len(app.dataframe) == 0  # Closed optional tables do no rendering work.
    app.session_state["model_comparison"] = True
    app.run()
    assert not app.exception
    assert len(app.dataframe) == 1
    app.session_state["model_comparison"] = True
    app.session_state["model_comparison_tab"] = "Probability metrics"
    app.run()
    assert not app.exception
    assert "roc_auc" in app.dataframe[0].value


def test_changed_artifact_invalidates_cache_immediately(tmp_path):
    import shutil
    import pandas as pd
    root = APP.parents[1]
    for directory in ("data/predictions", "reports/tables"):
        shutil.copytree(root / directory, tmp_path / directory)
    source = APP.read_text().replace("ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(tmp_path)!r})")
    app = AppTest.from_string(source, default_timeout=20).run()
    assert not app.exception
    metrics_path = tmp_path / "reports/tables/metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics.loc[metrics["model"] == "Logistic regression", "accuracy"] = 0.61
    metrics.to_csv(metrics_path, index=False)
    app.run()
    assert not app.exception
    assert app.metric[0].value == "61.0%"


def test_text_outcomes_use_full_wrapping_cards():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.radio[0].set_value("Predictions").run()
    app.selectbox[0].set_value("Always DOWN rule").run()
    assert not app.exception
    assert any(title.value == "CASH / DOWN" for title in app.subheader)
    assert any(text.value == "Prediction" for text in app.markdown)
    assert all(metric.label not in ("Prediction", "Actual open-to-close") for metric in app.metric)
