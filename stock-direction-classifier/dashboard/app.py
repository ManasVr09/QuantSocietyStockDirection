from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest import simulate_strategy  # noqa: E402
from src.features import FEATURE_COLUMNS  # noqa: E402


st.set_page_config(page_title="Stock Direction Lab", page_icon="↗", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1450px;}
      /* These cards use fixed light backgrounds, so their text must also have a
         fixed dark color when Streamlit is running in dark theme. */
      [data-testid="stMetric"] {
        background:#f6f8fb;
        border:1px solid #cbd5e1;
        padding:14px;
        border-radius:12px;
      }
      [data-testid="stMetric"] label,
      [data-testid="stMetric"] [data-testid="stMetricLabel"],
      [data-testid="stMetric"] [data-testid="stMetricValue"],
      [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color:#0f172a !important;
      }
      [data-testid="stMetric"] svg {fill:#0f172a !important;}
      .disclaimer {
        padding:12px 16px;
        border:1px solid #facc15;
        border-left:5px solid #ca8a04;
        background:#fef9c3;
        color:#422006 !important;
        border-radius:6px;
      }
      .disclaimer b {color:#422006 !important;}
      /* Streamlit's info/warning boxes can also retain dark-theme text colors
         on light yellow/blue backgrounds. Force accessible contrast inside them. */
      [data-testid="stAlert"] {color:#172033 !important;}
      [data-testid="stAlert"] p,
      [data-testid="stAlert"] div,
      [data-testid="stAlert"] span {color:#172033 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_artifacts():
    prediction_path = ROOT / "data" / "predictions" / "test_predictions.parquet"
    metrics_path = ROOT / "reports" / "tables" / "metrics.csv"
    if not prediction_path.exists() or not metrics_path.exists():
        return None
    predictions = pd.read_parquet(prediction_path)
    predictions["date"] = pd.to_datetime(predictions["date"])
    return {
        "predictions": predictions,
        "metrics": pd.read_csv(metrics_path),
        "calibration": pd.read_csv(ROOT / "reports" / "tables" / "calibration.csv"),
        "features": pd.read_csv(ROOT / "reports" / "tables" / "feature_analysis.csv"),
        "reliability": pd.read_csv(ROOT / "reports" / "tables" / "reliability.csv"),
        "summary": json.loads((ROOT / "reports" / "tables" / "dataset_summary.json").read_text()),
    }


def pct(value: float) -> str:
    return f"{value:.1%}"


def overview(artifacts):
    st.title("Stock Direction Lab")
    st.caption("A reproducible next-day UP/DOWN classification experiment")
    metrics, summary = artifacts["metrics"], artifacts["summary"]
    learned = metrics.loc[metrics["model"] != "Majority baseline"].sort_values("accuracy", ascending=False).iloc[0]
    baseline = metrics.loc[metrics["model"] == "Majority baseline"].iloc[0]
    columns = st.columns(5)
    columns[0].metric("Test observations", f"{summary['test_rows']:,}")
    columns[1].metric("Stocks", len(summary["tickers"]))
    columns[2].metric(f"{learned['model']} accuracy", pct(learned["accuracy"]), f"{learned['accuracy'] - baseline['accuracy']:+.1%} vs baseline")
    columns[3].metric("Balanced accuracy", pct(learned["balanced_accuracy"]))
    columns[4].metric("ROC-AUC", f"{learned['roc_auc']:.3f}")
    st.subheader("Experiment design")
    evaluation_label = "post-hoc subset evaluation" if summary.get("universe_note") else "untouched evaluation"
    st.write(
        f"Models learn from earlier data, use a separate validation period for model and threshold selection, and are refit on all pre-{summary['test_start'][:4]} data before the {summary['test_start'][:4]} {evaluation_label}. "
        "Every input is known by the close of the prediction date; the outcome is the following trading day's adjusted-close direction."
    )
    if summary.get("universe_note"):
        st.warning(
            "Universe-selection warning: this seven-stock subset was chosen after reviewing the original ten-stock 2025 results. "
            "Its aggregate performance is descriptive and must not be treated as a fresh out-of-sample result."
        )
        groups = summary.get("universe_groups", {})
        c1, c2 = st.columns(2)
        c1.markdown("**Original supporting cases**\n\n" + ", ".join(groups.get("supporting", [])))
        c2.markdown("**Original counterexamples**\n\n" + ", ".join(groups.get("counterexamples", [])))
    display = metrics.copy()
    for column in ["actual_up_rate", "predicted_up_rate", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]:
        display[column] = display[column].map(pct)
    st.dataframe(display, hide_index=True, use_container_width=True)
    reliability = artifacts["reliability"]
    champion_reliability = reliability.loc[reliability["model"] == learned["model"]].iloc[0]
    if bool(champion_reliability["statistically_clear"]):
        st.success(f"{learned['model']} beat the baseline with a positive 95% date-clustered confidence interval.")
    else:
        st.warning(
            f"The {learned['model']} improvement is not statistically clear: its 95% accuracy-difference interval is "
            f"{champion_reliability['accuracy_delta_ci_low']:.1%} to {champion_reliability['accuracy_delta_ci_high']:.1%}."
        )
    st.markdown('<div class="disclaimer"><b>Research disclaimer:</b> Historical classification results are not a live signal or investment advice. The seven-stock universe was selected post hoc, performance may not persist, and trading costs can remove small edges.</div>', unsafe_allow_html=True)


def prediction_explorer(artifacts):
    st.title("Prediction Explorer")
    data = artifacts["predictions"]
    c1, c2, c3 = st.columns(3)
    models = sorted(data["model"].unique())
    default = next((name for name in ("Extra trees", "Random forest", "Logistic regression") if name in models), models[0])
    model = c1.selectbox("Model", models, index=models.index(default))
    ticker = c2.selectbox("Stock", sorted(data["ticker"].unique()))
    subset = data[(data["model"] == model) & (data["ticker"] == ticker)].sort_values("date")
    chosen_date = c3.select_slider("Historical prediction date", options=subset["date"].dt.date.tolist(), value=subset["date"].dt.date.iloc[-1])
    row = subset.loc[subset["date"].dt.date == chosen_date].iloc[0]
    groups = artifacts["summary"].get("universe_groups", {})
    original_group = "Supporting case" if ticker in groups.get("supporting", []) else "Counterexample"
    st.caption(f"Original ten-stock classification: {original_group}")
    cols = st.columns(4)
    cols[0].metric("Probability of UP", pct(row["probability_up"]))
    cols[1].metric("Prediction", "UP" if row["prediction"] else "DOWN")
    cols[2].metric("Actual next day", "UP" if row["target"] else "DOWN")
    cols[3].metric("Outcome", "Correct" if row["correct"] else "Incorrect")
    history = subset[subset["date"] <= pd.Timestamp(chosen_date)].tail(90)
    fig = px.line(history, x="date", y="close", title=f"{ticker} adjusted close — prior 90 observations")
    fig.add_vline(x=pd.Timestamp(chosen_date).timestamp() * 1000, line_dash="dash", line_color="#f59e0b")
    st.plotly_chart(fig, use_container_width=True)
    feature_values = pd.DataFrame({"feature": FEATURE_COLUMNS, "value": [row[column] for column in FEATURE_COLUMNS]})
    st.subheader("Inputs used at this date")
    st.dataframe(feature_values, hide_index=True, use_container_width=True)


def performance(artifacts):
    st.title("Model Performance")
    data = artifacts["predictions"]
    models = sorted(data["model"].unique())
    default = next((name for name in ("Extra trees", "Random forest", "Logistic regression") if name in models), models[0])
    model = st.selectbox("Model", models, index=models.index(default), key="performance_model")
    subset = data[data["model"] == model]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Classification", "Curves", "Stability", "Calibration", "Reliability"])
    with tab1:
        matrix = confusion_matrix(subset["target"], subset["prediction"])
        fig = px.imshow(matrix, text_auto=True, x=["Predicted DOWN", "Predicted UP"], y=["Actual DOWN", "Actual UP"], color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fpr, tpr, _ = roc_curve(subset["target"], subset["probability_up"])
        precision, recall, _ = precision_recall_curve(subset["target"], subset["probability_up"])
        c1, c2 = st.columns(2)
        roc = go.Figure([go.Scatter(x=fpr, y=tpr, name=model), go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Chance")])
        roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        c1.plotly_chart(roc, use_container_width=True)
        pr = px.line(x=recall, y=precision, labels={"x": "Recall", "y": "Precision"}, title="Precision–recall curve")
        c2.plotly_chart(pr, use_container_width=True)
    with tab3:
        stock = subset.groupby("ticker", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
        supporting = artifacts["summary"].get("universe_groups", {}).get("supporting", [])
        stock["original_group"] = np.where(stock["ticker"].isin(supporting), "Supporting case", "Counterexample")
        month = subset.assign(month=subset["date"].dt.strftime("%b")).groupby("month", as_index=False, sort=False)["correct"].mean().rename(columns={"correct": "accuracy"})
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(stock.sort_values("accuracy"), x="accuracy", y="ticker", color="original_group", orientation="h", title="Accuracy by stock", range_x=[0, 1]), use_container_width=True)
        c2.plotly_chart(px.bar(month, x="month", y="accuracy", title="Accuracy by month", range_y=[0, 1]), use_container_width=True)
    with tab4:
        calibration = artifacts["calibration"]
        fig = px.line(calibration, x="mean_predicted", y="observed_up_rate", color="model", markers=True, title="Probability calibration")
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect calibration"))
        st.plotly_chart(fig, use_container_width=True)
    with tab5:
        reliability = artifacts["reliability"].copy()
        for column in ["accuracy_delta_vs_baseline", "accuracy_delta_ci_low", "accuracy_delta_ci_high"]:
            reliability[column] = reliability[column].map(pct)
        st.dataframe(reliability, hide_index=True, use_container_width=True)
        st.caption("Confidence intervals resample whole trading dates, preserving correlation among stocks observed on the same day.")


def feature_analysis_page(artifacts):
    st.title("Feature Analysis")
    data = artifacts["features"]
    available = data["model"].drop_duplicates().tolist()
    model = st.radio("Model", available, horizontal=True)
    subset = data[data["model"] == model].sort_values("value")
    fig = px.bar(subset, x="value", y="feature", orientation="h", color="value", color_continuous_scale="RdBu", title=subset["measure"].iloc[0].title())
    st.plotly_chart(fig, use_container_width=True)
    if model in ("Random forest", "Extra trees"):
        st.info("Impurity importance describes association with model decisions. It is not evidence that a feature causes returns.")
    else:
        st.info("Coefficients are based on standardized inputs. Their signs describe the fitted association, not causality.")


def strategy_page(artifacts):
    st.title("Strategy Simulation")
    test_year = artifacts["summary"]["test_start"][:4]
    st.caption(f"Illustrative close-to-close long/cash simulation on the {test_year} test set")
    data = artifacts["predictions"]
    c1, c2, c3, c4 = st.columns(4)
    models = sorted(data["model"].unique())
    default = next((name for name in ("Extra trees", "Random forest", "Logistic regression") if name in models), models[0])
    model = c1.selectbox("Model", models, index=models.index(default), key="strategy_model")
    ticker = c2.selectbox("Stock", sorted(data["ticker"].unique()), key="strategy_ticker")
    threshold = c3.slider("UP probability threshold", 0.50, 0.80, 0.55, 0.01)
    cost = c4.number_input("Transaction cost (bps)", 0.0, 100.0, 5.0, 1.0)
    selected = data[(data["model"] == model) & (data["ticker"] == ticker)]
    result = simulate_strategy(selected, threshold, cost)
    cols = st.columns(4)
    cols[0].metric("Strategy return", pct(result["strategy_curve"].iloc[-1] - 1))
    cols[1].metric("Buy & hold", pct(result["buy_hold_curve"].iloc[-1] - 1))
    cols[2].metric("Days invested", pct(result["position"].mean()))
    cols[3].metric("Maximum drawdown", pct(result["strategy_drawdown"].min()))
    curves = result.melt(id_vars="date", value_vars=["strategy_curve", "buy_hold_curve"], var_name="series", value_name="growth")
    st.plotly_chart(px.line(curves, x="date", y="growth", color="series", title="$1 growth during the test period"), use_container_width=True)
    st.warning("This simplified simulation is not proof of a tradable strategy. It excludes slippage, taxes, market impact, and execution delay.")


artifacts = load_artifacts()
if artifacts is None:
    st.title("Stock Direction Lab")
    st.error("No experiment artifacts were found.")
    st.code("python -m src.pipeline\nstreamlit run dashboard/app.py", language="bash")
    st.stop()

page = st.sidebar.radio("Navigate", ["Overview", "Prediction Explorer", "Model Performance", "Feature Analysis", "Strategy Simulation"])
st.sidebar.caption(f"Historical {artifacts['summary']['test_start'][:4]} replay • educational use only")
{
    "Overview": overview,
    "Prediction Explorer": prediction_explorer,
    "Model Performance": performance,
    "Feature Analysis": feature_analysis_page,
    "Strategy Simulation": strategy_page,
}[page](artifacts)
