from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Stock direction predictor", page_icon=":material/query_stats:", layout="wide", initial_sidebar_state="expanded")


ARTIFACTS = {
    "predictions": ROOT / "data/predictions/test_predictions.parquet",
    "ledger": ROOT / "data/predictions/trade_ledger.parquet",
    "portfolio": ROOT / "data/predictions/daily_portfolio.parquet",
    **{key: ROOT / "reports/tables" / filename for key, filename in {
        "metrics": "metrics.csv", "calibration": "calibration.csv", "features": "feature_analysis.csv",
        "reliability": "reliability.csv", "economics": "economic_metrics.csv",
        "economic_by_stock": "economic_by_stock.csv", "economic_uncertainty": "economic_uncertainty.csv",
        "summary": "dataset_summary.json", "manifest": "manifest.json",
    }.items()},
}
PAGE_ARTIFACTS = {
    "Overview": ("metrics", "economics", "predictions", "portfolio"),
    "Predictions": ("predictions",),
    "Performance": ("predictions", "metrics", "calibration", "reliability"),
    "Features": ("features",),
    "Backtest": ("economics", "economic_uncertainty", "portfolio", "economic_by_stock"),
    "Sandbox": ("predictions",),
}


def result_card(label, value, *, text=False, help=None, width=250):
    """Wrap full labels and text outcomes; reserve large metric type for short numbers."""
    with st.container(border=True, width=width):
        st.markdown(label, help=help)
        if text or len(str(value)) > 8:
            st.subheader(str(value), anchor=False)
        else:
            st.metric(label, value, label_visibility="collapsed")


@st.cache_data(max_entries=24, show_spinner=False)
def read_artifact(path: str, signature: tuple[int, int], columns: tuple[str, ...] | None = None):
    """Cache trusted saved artifacts; file changes invalidate instead of waiting for a TTL."""
    source = Path(path)
    if source.suffix == ".json":
        return json.loads(source.read_text())
    frame = pd.read_parquet(source, columns=list(columns) if columns else None) if source.suffix == ".parquet" else pd.read_csv(source)
    for column in ("date", "feature_time", "signal_time", "target_date", "entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def load_artifacts(page: str = "Overview") -> dict | None:
    # Preserve the existing complete-experiment check, but only read what this view uses.
    if any(not path.exists() for path in ARTIFACTS.values()):
        return None
    result = {}
    for name in ("summary", "manifest", *PAGE_ARTIFACTS[page]):
        path = ARTIFACTS[name]
        stat = path.stat()
        columns = ("model", "ticker", "target_date", "probability_up", "correct") if page == "Overview" and name == "predictions" else None
        result[name] = read_artifact(str(path), (stat.st_mtime_ns, stat.st_size), columns)
    return result


def pct(value: float) -> str:
    return f"{value:.1%}"


def research_table(frame: pd.DataFrame, *, hide_index: bool = True) -> None:
    """Use readable labels, numeric formats, and roomy rows across research views."""
    labels = {
        "model": "Model", "ticker": "Stock", "observations": "Observations",
        "actual_up_rate": "Actual up", "predicted_up_rate": "Predicted up",
        "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
        "roc_auc": "ROC-AUC", "f1": "F1", "brier_score": "Brier score",
        "log_loss": "Log loss", "transaction_cost_bps": "Cost (bps / side)",
        "cagr": "CAGR", "total_return": "Net return", "maximum_drawdown": "Max drawdown",
        "annualized_volatility": "Annual volatility", "expected_shortfall_5pct": "Expected shortfall (5%)",
        "trade_count": "Trades", "average_turnover": "Average turnover",
        "accuracy_delta_vs_baseline": "Accuracy vs baseline",
        "accuracy_delta_ci_low": "95% interval · lower", "accuracy_delta_ci_high": "95% interval · upper",
        "mean_predicted": "Mean prediction", "observed_up_rate": "Observed up",
        "bootstrap_method": "Bootstrap method", "statistically_clear": "Interval excludes zero",
    }
    percentages = {
        "actual_up_rate", "predicted_up_rate", "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "total_return", "cagr", "maximum_drawdown", "annualized_volatility", "expected_shortfall_5pct",
        "hit_rate", "exposure", "average_turnover", "accuracy_delta_vs_baseline", "accuracy_delta_ci_low",
        "accuracy_delta_ci_high", "mean_predicted", "observed_up_rate",
    }
    config = {}
    for name in frame.columns:
        label = labels.get(name, str(name).replace("_", " ").capitalize())
        if name in percentages:
            config[name] = st.column_config.NumberColumn(label, format="percent", width="medium")
        elif pd.api.types.is_float_dtype(frame[name]):
            config[name] = st.column_config.NumberColumn(label, format="%.3f", width="medium")
        elif pd.api.types.is_datetime64_any_dtype(frame[name]):
            config[name] = st.column_config.DatetimeColumn(label, format="YYYY-MM-DD HH:mm", width="medium")
        elif name in ("model", "ticker", "feature", "event", "bootstrap_method"):
            config[name] = st.column_config.TextColumn(label, width="large" if name in ("model", "feature", "bootstrap_method") else "medium", pinned=name in ("model", "ticker"))
        else:
            config[name] = st.column_config.Column(label, width="medium")
    st.dataframe(frame, hide_index=hide_index, width="stretch", height="content", row_height=44, column_config=config)


@st.fragment
def probability_chart(data: pd.DataFrame, summary: dict) -> None:
    champion = summary["validation_declared_champion"]
    ticker = st.pills("Explore a stock", summary["tickers"], default=summary["tickers"][0], key="overview_stock")
    ticker = ticker or summary["tickers"][0]
    history = data[(data["model"] == champion) & (data["ticker"] == ticker)].sort_values("target_date")
    with st.container(border=True):
        st.subheader("The model’s conviction over time")
        st.caption(f"{ticker} · probability of an up move at the next session’s open-to-close horizon")
        line = alt.Chart(history[["target_date", "probability_up"]]).mark_line(strokeWidth=2).encode(
            x=alt.X("target_date:T", title=None),
            y=alt.Y("probability_up:Q", title="Probability of up", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("target_date:T", title="Trade date"),
                     alt.Tooltip("probability_up:Q", title="Probability of up", format=".1%")],
        )
        rule = alt.Chart(pd.DataFrame({"threshold": [summary["validation_selected_threshold"]]})).mark_rule(
            color="#9A7A45", strokeDash=[5, 5]
        ).encode(y="threshold:Q")
        st.altair_chart((line + rule).properties(height=300).configure_axis(labelFontSize=14, titleFontSize=15, labelColor="currentColor", titleColor="currentColor").configure_legend(labelFontSize=14, titleFontSize=15, labelColor="currentColor", titleColor="currentColor"), width="stretch")
        st.caption(f"Dashed line: locked trading threshold ({summary['validation_selected_threshold']:.0%}). "
                   "The four headline metrics cover all seven stocks; this chart follows the selected stock.")



def overview(a: dict) -> None:
    summary = a["summary"]
    champion = summary["validation_declared_champion"]
    metrics = a["metrics"]
    official = metrics.loc[metrics["model"] == champion].iloc[0]
    baseline = metrics.loc[metrics["model"] == "Majority baseline"].iloc[0]
    base_cost = float(summary["execution_contract"]["transaction_cost_bps"])
    economics = a["economics"].loc[a["economics"]["transaction_cost_bps"] == base_cost].iloc[0]
    with st.container(horizontal=True):
        result_card("Historical accuracy", pct(official["accuracy"]), width=210)
        result_card("Majority baseline", pct(baseline["accuracy"]), width=210)
        result_card("ROC-AUC", f"{official['roc_auc']:.3f}", width=210,
                    help="Ranking ability: 0.5 is chance; 1.0 is perfect.")
        result_card("Net return", pct(economics["total_return"]), width=210,
                    help="Locked validation policy, including transaction costs and slippage.")

    data = a["predictions"]
    probability_chart(data, summary)

    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Accuracy by stock")
        st.caption("How often the stored class prediction matched the outcome.")
        stock = data[data["model"] == champion].groupby("ticker", as_index=False)["correct"].mean()
        chart = alt.Chart(stock).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("correct:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            y=alt.Y("ticker:N", title=None, sort="-x"),
            tooltip=["ticker", alt.Tooltip("correct:Q", title="Accuracy", format=".1%")],
        ).properties(height=240)
        st.altair_chart(chart.configure_axis(labelFontSize=14, titleFontSize=15, labelColor="currentColor", titleColor="currentColor").configure_legend(labelFontSize=14, titleFontSize=15, labelColor="currentColor", titleColor="currentColor"), width="stretch")
    with right.container(border=True):
        st.subheader("The executable result")
        st.caption("Growth of one unit of capital under the locked policy.")
        chart = alt.Chart(a["portfolio"][["entry_time", "wealth"]]).mark_line(strokeWidth=2).encode(
            x=alt.X("entry_time:T", title=None),
            y=alt.Y("wealth:Q", title="Net wealth", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("entry_time:T", title="Date"), alt.Tooltip("wealth:Q", format=".4f")],
        ).properties(height=240)
        st.altair_chart(chart.configure_axis(labelFontSize=14, titleFontSize=15, labelColor="currentColor", titleColor="currentColor").configure_legend(labelFontSize=14, titleFontSize=15, labelColor="currentColor", titleColor="currentColor"), width="stretch")

    with st.container(border=True):
        st.subheader("What do these results mean?")
        difference = 100 * (official["accuracy"] - baseline["accuracy"])
        st.write(
            f"The validation-selected model scored {official['accuracy']:.1%} accuracy, "
            f"{abs(difference):.1f} percentage points {'above' if difference >= 0 else 'below'} "
            "the majority baseline. Accuracy and trading returns answer different questions: "
            "the backtest also depends on the probability threshold, trade timing, and costs."
        )
        st.caption("These descriptive results do not establish a reliable trading edge or a future return.")
    details = st.expander('Compare all models', on_change="rerun", key="model_comparison")
    if details.open:
        with details:
            classification, probability = st.tabs(["Classification metrics", "Probability metrics"], on_change="rerun", key="model_comparison_tab")
            if classification.open:
                with classification:
                    research_table(metrics[["model", "observations", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]])
            if probability.open:
                with probability:
                    research_table(metrics[["model", "actual_up_rate", "predicted_up_rate", "roc_auc", "log_loss", "brier_score"]])
    details = st.expander('Methodology & research protocol', on_change="rerun", key="research_protocol")
    if details.open:
        with details:
            contract = summary["execution_contract"]
            st.caption(
                f"Executable clock: information after close(t) → signal after {contract['processing_delay_minutes']} minutes → "
                "entry at open(t+1) → exit at close(t+1)."
            )
            st.write(
                "Train: 2016–2022. Validation: 2023–2024. Exploratory evaluation: 2025. "
                "The champion and policy threshold are declared from validation results before test metrics are displayed. "
                "Train/validation/test rows are purged by outcome date, so labels cannot cross partition boundaries. "
                "The seven-stock universe was selected after reviewing earlier 2025 results."
            )
            st.caption(f"Brier score: {official['brier_score']:.3f} · lower is better.")
            st.json({
                "status": a["manifest"]["research_status"],
                "champion": champion,
                "locked threshold": summary["validation_selected_threshold"],
                "run id": a["manifest"]["run_id"],
                "code commit": a["manifest"]["code_commit"],
            }, expanded=False)


def prediction_explorer(a: dict) -> None:
    from src.features import FEATURE_COLUMNS

    st.title("Follow a historical prediction")
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
        result_card('Probability of up', pct(row['probability_up']), width=250)
        result_card('Locked threshold', f"{row['decision_threshold']:.2f}", width=250)
        result_card('Prediction', 'UP' if row['prediction'] else 'CASH / DOWN', width=250, text=True)
        result_card('Actual open-to-close', 'UP' if row['target'] else 'DOWN', width=250, text=True)
        result_card('Gross return', pct(row['gross_return']), width=250)
    timing = pd.DataFrame({
        "event": ["Feature cutoff", "Signal", "Entry", "Exit"],
        "timestamp": [row["feature_time"], row["signal_time"], row["entry_time"], row["exit_time"]],
        "price": [row["close"], np.nan, row["entry_price"], row["exit_price"]],
    })
    st.subheader("Execution timeline")
    research_table(timing)
    history = subset[subset["feature_time"] <= row["feature_time"]].tail(90)
    st.line_chart(history, x="date", y="close", x_label="Date", y_label="Adjusted close")
    details = st.expander('Feature values used for this signal', on_change="rerun", key="signal_features")
    if details.open:
        with details:
            research_table(pd.DataFrame({"feature": FEATURE_COLUMNS, "value": [row[column] for column in FEATURE_COLUMNS]}))


def model_performance(a: dict) -> None:
    st.title("Look beyond headline accuracy")
    data = a["predictions"]
    champion = a["summary"]["validation_declared_champion"]
    models = sorted(data["model"].unique())
    model = st.selectbox("Model", models, index=models.index(champion), key="performance_model")
    subset = data[data["model"] == model]
    view = st.segmented_control("Analysis", ["Classification", "Probability quality", "Stability", "Reliability"], default="Classification")
    if view == "Classification":
        from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

        matrix = confusion_matrix(subset["target"], subset["prediction"])
        matrix_df = pd.DataFrame(matrix, index=["Actual down", "Actual up"], columns=["Predicted down", "Predicted up"])
        research_table(matrix_df, hide_index=False)
        fpr, tpr, _ = roc_curve(subset["target"], subset["probability_up"])
        precision, recall, _ = precision_recall_curve(subset["target"], subset["probability_up"])
        c1, c2 = st.columns(2)
        c1.line_chart(pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr}), x="False positive rate", y="True positive rate")
        c2.line_chart(pd.DataFrame({"Recall": recall, "Precision": precision}), x="Recall", y="Precision")
    elif view == "Probability quality":
        row = a["metrics"].loc[a["metrics"]["model"] == model].iloc[0]
        with st.container(horizontal=True):
            result_card('Log loss', f"{row['log_loss']:.3f}", width=250)
            result_card('Brier score', f"{row['brier_score']:.3f}", width=250)
            result_card('ROC-AUC', f"{row['roc_auc']:.3f}", width=250)
        calibration = a["calibration"].loc[a["calibration"]["model"] == model]
        chart = alt.Chart(calibration).mark_line(point=True).encode(
            x=alt.X("mean_predicted:Q", title="Mean predicted probability"),
            y=alt.Y("observed_up_rate:Q", title="Observed up rate"),
            tooltip=["mean_predicted", "observed_up_rate", "count"],
        ) + alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[5, 5], color="gray").encode(x="x", y="y")
        st.altair_chart(chart.configure_axis(labelFontSize=14, titleFontSize=15, labelColor="currentColor", titleColor="currentColor").configure_legend(labelFontSize=14, titleFontSize=15, labelColor="currentColor", titleColor="currentColor"), width="stretch")
        research_table(calibration)
    elif view == "Stability":
        stock = subset.groupby("ticker", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
        month = subset.assign(month=subset["target_date"].dt.to_period("M").astype(str)).groupby("month", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
        c1, c2 = st.columns(2)
        c1.bar_chart(stock, x="ticker", y="accuracy", y_label="Accuracy")
        c2.bar_chart(month, x="month", y="accuracy", y_label="Accuracy")
    else:
        reliability = a["reliability"]
        research_table(reliability[["model", "accuracy_delta_vs_baseline", "accuracy_delta_ci_low", "accuracy_delta_ci_high", "statistically_clear"]])
        details = st.expander('Stock, month, and bootstrap details', on_change="rerun", key="reliability_details")
        if details.open:
            with details:
                research_table(reliability[["model", "stocks_beating_baseline", "stocks_total", "months_beating_baseline", "months_total", "bootstrap_method"]])
        st.caption("Uncertainty uses a moving-block bootstrap, preserving cross-sectional and short-run serial dependence.")


def feature_analysis(a: dict) -> None:
    st.title("What does the model pay attention to?")
    data = a["features"]
    available = data["model"].drop_duplicates().tolist()
    model = st.segmented_control("Model", available, default=available[0])
    subset = data[data["model"] == model].sort_values("value")
    st.bar_chart(subset, x="feature", y="value", horizontal=True)
    st.info("Feature importance and coefficients describe fitted associations, not causal effects.", icon=":material/info:")


def official_strategy(a: dict) -> None:
    st.title("From a prediction to an executable trade")
    st.caption("Locked validation policy; controls on this page cannot alter the official result")
    champion = a["summary"]["validation_declared_champion"]
    threshold = a["summary"]["validation_selected_threshold"]
    costs = a["economics"].sort_values("transaction_cost_bps")
    base_cost = float(a["summary"]["execution_contract"]["transaction_cost_bps"])
    official = costs[costs["transaction_cost_bps"] == base_cost].iloc[0]
    uncertainty = a["economic_uncertainty"].iloc[0]
    st.write(f"**Validation-selected model:** {champion}")
    with st.container(horizontal=True):
        result_card('Locked threshold', f'{threshold:.2f}', width=250)
        result_card('Total net return', pct(official['total_return']), width=250)
        result_card('Sharpe', f"{official['sharpe']:.2f}", width=250)
        result_card('Maximum drawdown', pct(official['maximum_drawdown']), width=250)
        result_card('Trades', f"{int(official['trade_count']):,}", width=250)
    st.subheader("Cost sensitivity")
    returns, risk = st.tabs(["Returns & trading", "Risk & exposure"], on_change="rerun", key="cost_tab")
    if returns.open:
        with returns:
            research_table(costs[["transaction_cost_bps", "total_return", "cagr", "sharpe", "sortino", "trade_count", "hit_rate"]])
    if risk.open:
        with risk:
            research_table(costs[["transaction_cost_bps", "maximum_drawdown", "annualized_volatility", "expected_shortfall_5pct", "exposure", "average_turnover"]])
    st.info(
        f"Moving-block 95% interval: total return {uncertainty['total_return_ci_low']:.1%} to "
        f"{uncertainty['total_return_ci_high']:.1%}; Sharpe {uncertainty['sharpe_ci_low']:.2f} to "
        f"{uncertainty['sharpe_ci_high']:.2f}.",
        icon=":material/analytics:",
    )
    portfolio = a["portfolio"].rename(columns={"entry_time": "Date", "wealth": "Net wealth"})
    st.line_chart(portfolio, x="Date", y="Net wealth")
    st.subheader("Per-stock results")
    research_table(a["economic_by_stock"])


def exploratory_sandbox(a: dict) -> None:
    st.title("Explore a different scenario")
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
        from src.backtest import BacktestConfig, simulate

        selected = data[(data["model"] == model) & (data["ticker"] == ticker)]
        report = simulate(selected, BacktestConfig(
            probability_threshold=threshold,
            transaction_cost_bps=cost,
            slippage_bps=a["summary"]["execution_contract"]["slippage_bps"],
        ))
        with st.container(horizontal=True):
            result_card('Exploratory return', pct(report.metrics['total_return']), width=250)
            result_card('Maximum drawdown', pct(report.metrics['maximum_drawdown']), width=250)
            result_card('Trades', report.metrics['trade_count'], width=250)
            result_card('Exposure', pct(report.metrics['exposure']), width=250)
        st.line_chart(report.daily_portfolio, x="entry_time", y="wealth")


pages = {
    "Overview": overview,
    "Predictions": prediction_explorer,
    "Performance": model_performance,
    "Features": feature_analysis,
    "Backtest": official_strategy,
    "Sandbox": exploratory_sandbox,
}
with st.sidebar:
    st.header("Research views")
    page = st.radio("Research views", list(pages), key="research_view", label_visibility="collapsed")

st.title("Stock direction predictor")
a = load_artifacts(page)
if a is None:
    st.error("Required immutable experiment artifacts are missing. Run the pipeline before starting the dashboard.")
    st.code("python -m src.pipeline\nstreamlit run dashboard/app.py", language="bash")
    st.stop()

pages[page](a)
st.divider()
st.caption("QUANT SOCIETY · Stock direction research lab · Educational research, not a live trading signal.")
st.caption(f"Data: Yahoo Finance · historical evaluation: {a['summary']['test_start']} to {a['summary']['test_end']} · "
           f"Run {a['manifest']['run_id']}")
