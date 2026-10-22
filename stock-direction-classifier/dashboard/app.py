from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest import BacktestConfig, simulate  # noqa: E402
from src.features import FEATURE_COLUMNS  # noqa: E402


st.set_page_config(page_title="Stock direction research lab", page_icon=":material/query_stats:", layout="wide")


@st.cache_data(ttl="1h", max_entries=4, show_spinner="Loading immutable research artifacts…")
def load_artifacts() -> dict | None:
    required = {
        "predictions": ROOT / "data" / "predictions" / "test_predictions.parquet",
        "ledger": ROOT / "data" / "predictions" / "trade_ledger.parquet",
        "portfolio": ROOT / "data" / "predictions" / "daily_portfolio.parquet",
        "metrics": ROOT / "reports" / "tables" / "metrics.csv",
        "calibration": ROOT / "reports" / "tables" / "calibration.csv",
        "features": ROOT / "reports" / "tables" / "feature_analysis.csv",
        "reliability": ROOT / "reports" / "tables" / "reliability.csv",
        "economics": ROOT / "reports" / "tables" / "economic_metrics.csv",
        "economic_by_stock": ROOT / "reports" / "tables" / "economic_by_stock.csv",
        "economic_uncertainty": ROOT / "reports" / "tables" / "economic_uncertainty.csv",
        "summary": ROOT / "reports" / "tables" / "dataset_summary.json",
        "manifest": ROOT / "reports" / "tables" / "manifest.json",
    }
    if any(not path.exists() for path in required.values()):
        return None
    result: dict = {}
    for name in ("predictions", "ledger", "portfolio"):
        result[name] = pd.read_parquet(required[name])
    for name in ("metrics", "calibration", "features", "reliability", "economics", "economic_by_stock", "economic_uncertainty"):
        result[name] = pd.read_csv(required[name])
    result["summary"] = json.loads(required["summary"].read_text())
    result["manifest"] = json.loads(required["manifest"].read_text())
    for frame_name in ("predictions", "ledger", "portfolio"):
        for column in ("date", "feature_time", "signal_time", "target_date", "entry_time", "exit_time"):
            if column in result[frame_name]:
                result[frame_name][column] = pd.to_datetime(result[frame_name][column])
    return result


def pct(value: float) -> str:
    return f"{value:.1%}"


def research_banner(summary: dict) -> None:
    st.warning(
        "Exploratory, post-hoc research — not an untouched confirmation and not suitable for portfolio action. "
        "The seven-stock universe was selected after earlier results were inspected.",
        icon=":material/warning:",
    )
    contract = summary["execution_contract"]
    st.caption(
        f"Executable clock: information after close(t) → signal after {contract['processing_delay_minutes']} minutes → "
        f"entry at open(t+1) → exit at close(t+1)."
    )


def overview(a: dict) -> None:
    st.title("Stock direction research lab")
    st.caption("Leakage-controlled classification and executable next-open backtesting")
    research_banner(a["summary"])
    champion = a["summary"]["validation_declared_champion"]
    metrics = a["metrics"]
    official = metrics.loc[metrics["model"] == champion].iloc[0]
    baseline = metrics.loc[metrics["model"] == "Majority baseline"].iloc[0]
    base_cost = float(a["summary"]["execution_contract"]["transaction_cost_bps"])
    economics = a["economics"].loc[a["economics"]["transaction_cost_bps"] == base_cost].iloc[0]
    with st.container(horizontal=True):
        st.metric("Validation-declared champion", champion, border=True)
        st.metric("Test accuracy", pct(official["accuracy"]), f"{official['accuracy'] - baseline['accuracy']:+.1%} vs baseline", border=True)
        st.metric("ROC-AUC", f"{official['roc_auc']:.3f}", border=True)
        st.metric("Brier score", f"{official['brier_score']:.3f}", help="Lower is better.", border=True)
        st.metric("Net strategy return", pct(economics["total_return"]), border=True)
    with st.container(border=True):
        st.subheader("Research protocol")
        st.write(
            "The champion and policy threshold are declared from validation results before test metrics are displayed. "
            "Train/validation/test rows are purged by outcome date, so labels cannot cross partition boundaries."
        )
        st.json({
            "status": a["manifest"]["research_status"],
            "champion": champion,
            "locked threshold": a["summary"]["validation_selected_threshold"],
            "run id": a["manifest"]["run_id"],
            "code commit": a["manifest"]["code_commit"],
        }, expanded=False)
    display = metrics.copy()
    for column in ("actual_up_rate", "predicted_up_rate", "accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        display[column] = display[column].map(pct)
    st.subheader("All model results")
    st.dataframe(display, hide_index=True)


def prediction_explorer(a: dict) -> None:
    st.title("Prediction explorer")
    st.caption("Audit one historical signal from information cutoff through executable outcome")
    data = a["predictions"]
    c1, c2, c3 = st.columns(3)
    champion = a["summary"]["validation_declared_champion"]
    models = sorted(data["model"].unique())
    model = c1.selectbox("Model", models, index=models.index(champion))
    ticker = c2.selectbox("Stock", sorted(data["ticker"].unique()))
    subset = data[(data["model"] == model) & (data["ticker"] == ticker)].sort_values("feature_time")
    chosen = c3.select_slider("Signal date", options=subset["date"].dt.date.tolist(), value=subset["date"].dt.date.iloc[-1])
    row = subset.loc[subset["date"].dt.date == chosen].iloc[0]
    with st.container(horizontal=True):
        st.metric("Probability of up", pct(row["probability_up"]), border=True)
        st.metric("Locked threshold", f"{row['decision_threshold']:.2f}", border=True)
        st.metric("Prediction", "UP" if row["prediction"] else "CASH / DOWN", border=True)
        st.metric("Actual open-to-close", "UP" if row["target"] else "DOWN", border=True)
        st.metric("Gross return", pct(row["gross_return"]), border=True)
    timing = pd.DataFrame({
        "event": ["Feature cutoff", "Signal", "Entry", "Exit"],
        "timestamp": [row["feature_time"], row["signal_time"], row["entry_time"], row["exit_time"]],
        "price": [row["close"], np.nan, row["entry_price"], row["exit_price"]],
    })
    st.subheader("Execution timeline")
    st.dataframe(timing, hide_index=True)
    history = subset[subset["feature_time"] <= row["feature_time"]].tail(90)
    st.line_chart(history, x="date", y="close", x_label="Date", y_label="Adjusted close")
    with st.expander("Feature values used for this signal"):
        st.dataframe(pd.DataFrame({"feature": FEATURE_COLUMNS, "value": [row[column] for column in FEATURE_COLUMNS]}), hide_index=True)


def model_performance(a: dict) -> None:
    st.title("Model performance")
    data = a["predictions"]
    champion = a["summary"]["validation_declared_champion"]
    models = sorted(data["model"].unique())
    model = st.selectbox("Model", models, index=models.index(champion), key="performance_model")
    subset = data[data["model"] == model]
    view = st.segmented_control("Analysis", ["Classification", "Probability quality", "Stability", "Reliability"], default="Classification")
    if view == "Classification":
        matrix = confusion_matrix(subset["target"], subset["prediction"])
        matrix_df = pd.DataFrame(matrix, index=["Actual down", "Actual up"], columns=["Predicted down", "Predicted up"])
        st.dataframe(matrix_df)
        fpr, tpr, _ = roc_curve(subset["target"], subset["probability_up"])
        precision, recall, _ = precision_recall_curve(subset["target"], subset["probability_up"])
        c1, c2 = st.columns(2)
        c1.line_chart(pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr}), x="False positive rate", y="True positive rate")
        c2.line_chart(pd.DataFrame({"Recall": recall, "Precision": precision}), x="Recall", y="Precision")
    elif view == "Probability quality":
        row = a["metrics"].loc[a["metrics"]["model"] == model].iloc[0]
        with st.container(horizontal=True):
            st.metric("Log loss", f"{row['log_loss']:.3f}", border=True)
            st.metric("Brier score", f"{row['brier_score']:.3f}", border=True)
            st.metric("ROC-AUC", f"{row['roc_auc']:.3f}", border=True)
        calibration = a["calibration"].loc[a["calibration"]["model"] == model]
        chart = alt.Chart(calibration).mark_line(point=True).encode(
            x=alt.X("mean_predicted:Q", title="Mean predicted probability"),
            y=alt.Y("observed_up_rate:Q", title="Observed up rate"),
            tooltip=["mean_predicted", "observed_up_rate", "count"],
        ) + alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[5, 5], color="gray").encode(x="x", y="y")
        st.altair_chart(chart)
        st.dataframe(calibration, hide_index=True)
    elif view == "Stability":
        stock = subset.groupby("ticker", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
        month = subset.assign(month=subset["target_date"].dt.to_period("M").astype(str)).groupby("month", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
        c1, c2 = st.columns(2)
        c1.bar_chart(stock, x="ticker", y="accuracy", y_label="Accuracy")
        c2.bar_chart(month, x="month", y="accuracy", y_label="Accuracy")
    else:
        reliability = a["reliability"].copy()
        for column in ("accuracy_delta_vs_baseline", "accuracy_delta_ci_low", "accuracy_delta_ci_high"):
            reliability[column] = reliability[column].map(pct)
        st.dataframe(reliability, hide_index=True)
        st.caption("Uncertainty uses a moving-block bootstrap, preserving cross-sectional and short-run serial dependence.")


def feature_analysis(a: dict) -> None:
    st.title("Feature analysis")
    data = a["features"]
    available = data["model"].drop_duplicates().tolist()
    model = st.segmented_control("Model", available, default=available[0])
    subset = data[data["model"] == model].sort_values("value")
    st.bar_chart(subset, x="feature", y="value", horizontal=True)
    st.info("Feature importance and coefficients describe fitted associations, not causal effects.", icon=":material/info:")


def official_strategy(a: dict) -> None:
    st.title("Official executable backtest")
    st.caption("Locked validation policy; controls on this page cannot alter the official result")
    research_banner(a["summary"])
    champion = a["summary"]["validation_declared_champion"]
    threshold = a["summary"]["validation_selected_threshold"]
    costs = a["economics"].sort_values("transaction_cost_bps")
    base_cost = float(a["summary"]["execution_contract"]["transaction_cost_bps"])
    official = costs[costs["transaction_cost_bps"] == base_cost].iloc[0]
    uncertainty = a["economic_uncertainty"].iloc[0]
    with st.container(horizontal=True):
        st.metric("Champion", champion, border=True)
        st.metric("Locked threshold", f"{threshold:.2f}", border=True)
        st.metric("Total net return", pct(official["total_return"]), border=True)
        st.metric("Sharpe", f"{official['sharpe']:.2f}", border=True)
        st.metric("Maximum drawdown", pct(official["maximum_drawdown"]), border=True)
        st.metric("Trades", f"{int(official['trade_count']):,}", border=True)
    st.subheader("Cost sensitivity")
    st.dataframe(costs, hide_index=True)
    st.info(
        f"Moving-block 95% interval: total return {uncertainty['total_return_ci_low']:.1%} to "
        f"{uncertainty['total_return_ci_high']:.1%}; Sharpe {uncertainty['sharpe_ci_low']:.2f} to "
        f"{uncertainty['sharpe_ci_high']:.2f}.",
        icon=":material/analytics:",
    )
    portfolio = a["portfolio"].rename(columns={"entry_time": "Date", "wealth": "Net wealth"})
    st.line_chart(portfolio, x="Date", y="Net wealth")
    st.subheader("Per-stock results")
    st.dataframe(a["economic_by_stock"], hide_index=True)


def exploratory_sandbox(a: dict) -> None:
    st.title("Exploratory sandbox")
    st.error("Sandbox results are test-period exploration. They are not official metrics and must not be used to claim improvement.", icon=":material/science:")
    data = a["predictions"]
    models = sorted(data["model"].unique())
    with st.form("sandbox_controls"):
        c1, c2, c3, c4 = st.columns(4)
        model = c1.selectbox("Model", models)
        ticker = c2.selectbox("Stock", sorted(data["ticker"].unique()))
        threshold = c3.slider("Exploratory probability threshold", 0.50, 0.80, 0.60, 0.01)
        cost = c4.number_input("One-way transaction cost (bps)", 0.0, 100.0, 10.0, 1.0)
        submitted = st.form_submit_button("Run exploratory scenario", icon=":material/play_arrow:")
    if submitted:
        selected = data[(data["model"] == model) & (data["ticker"] == ticker)]
        report = simulate(selected, BacktestConfig(
            probability_threshold=threshold,
            transaction_cost_bps=cost,
            slippage_bps=a["summary"]["execution_contract"]["slippage_bps"],
        ))
        with st.container(horizontal=True):
            st.metric("Exploratory return", pct(report.metrics["total_return"]), border=True)
            st.metric("Maximum drawdown", pct(report.metrics["maximum_drawdown"]), border=True)
            st.metric("Trades", report.metrics["trade_count"], border=True)
            st.metric("Exposure", pct(report.metrics["exposure"]), border=True)
        st.line_chart(report.daily_portfolio, x="entry_time", y="wealth")


a = load_artifacts()
if a is None:
    st.title("Stock direction research lab")
    st.error("Required immutable experiment artifacts are missing. Run the pipeline before starting the dashboard.")
    st.code("python -m src.pipeline\nstreamlit run dashboard/app.py", language="bash")
    st.stop()

pages = {
    "Audit overview": overview,
    "Prediction explorer": prediction_explorer,
    "Model performance": model_performance,
    "Feature analysis": feature_analysis,
    "Official backtest": official_strategy,
    "Exploratory sandbox": exploratory_sandbox,
}
with st.sidebar:
    st.header("Research views")
    page = st.radio("Navigate", list(pages), label_visibility="collapsed")
    st.caption(f"Run {a['manifest']['run_id']} • educational use only")
pages[page](a)
