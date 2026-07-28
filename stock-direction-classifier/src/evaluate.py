from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features import FEATURE_COLUMNS


def predict_all(models: dict[str, object], test: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    base_columns = ["date", "ticker", "close", "target", "next_return", *FEATURE_COLUMNS]
    outputs = []
    for name, model in models.items():
        result = test[base_columns].copy()
        result["model"] = name
        result["probability_up"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        result["decision_threshold"] = (thresholds or {}).get(name, 0.5)
        result["prediction"] = (result["probability_up"] >= result["decision_threshold"]).astype(int)
        result["correct"] = result["prediction"] == result["target"]
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model"):
        actual, predicted, probability = group["target"], group["prediction"], group["probability_up"]
        rows.append(
            {
                "model": model,
                "observations": len(group),
                "actual_up_rate": actual.mean(),
                "predicted_up_rate": predicted.mean(),
                "accuracy": accuracy_score(actual, predicted),
                "balanced_accuracy": balanced_accuracy_score(actual, predicted),
                "precision": precision_score(actual, predicted, zero_division=0),
                "recall": recall_score(actual, predicted, zero_division=0),
                "f1": f1_score(actual, predicted, zero_division=0),
                "roc_auc": roc_auc_score(actual, probability),
            }
        )
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model"):
        observed, predicted = calibration_curve(group["target"], group["probability_up"], n_bins=bins, strategy="quantile")
        rows.extend({"model": model, "mean_predicted": x, "observed_up_rate": y} for x, y in zip(predicted, observed))
    return pd.DataFrame(rows)


def reliability_table(predictions: pd.DataFrame, bootstrap_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Measure consistency and date-clustered uncertainty versus the majority baseline."""
    baseline = predictions[predictions["model"] == "Majority baseline"].set_index(["date", "ticker"])
    dates = np.array(sorted(pd.to_datetime(predictions["date"]).unique()))
    rng = np.random.default_rng(seed)
    rows = []
    for model, group in predictions.groupby("model"):
        indexed = group.set_index(["date", "ticker"])
        stock_accuracy = group.groupby("ticker")["correct"].mean()
        stock_baseline = baseline.groupby("ticker")["correct"].mean()
        monthly_accuracy = group.assign(month=pd.to_datetime(group["date"]).dt.to_period("M")).groupby("month")["correct"].mean()
        monthly_baseline = baseline.reset_index().assign(month=lambda x: pd.to_datetime(x["date"]).dt.to_period("M")).groupby("month")["correct"].mean()
        daily_delta = (
            indexed["correct"].groupby(level="date").mean()
            - baseline["correct"].groupby(level="date").mean()
        ).reindex(pd.to_datetime(dates)).to_numpy()
        sampled_positions = rng.integers(0, len(daily_delta), size=(bootstrap_samples, len(daily_delta)))
        deltas = daily_delta[sampled_positions].mean(axis=1)
        rows.append({
            "model": model,
            "accuracy_delta_vs_baseline": group["correct"].mean() - baseline["correct"].mean(),
            "accuracy_delta_ci_low": np.quantile(deltas, 0.025),
            "accuracy_delta_ci_high": np.quantile(deltas, 0.975),
            "stocks_beating_baseline": int((stock_accuracy > stock_baseline + 1e-12).sum()),
            "stocks_total": len(stock_accuracy),
            "months_beating_baseline": int((monthly_accuracy > monthly_baseline + 1e-12).sum()),
            "months_total": len(monthly_accuracy),
            "statistically_clear": bool(np.quantile(deltas, 0.025) > 0),
        })
    return pd.DataFrame(rows).sort_values("accuracy_delta_vs_baseline", ascending=False)


def feature_analysis(models: dict[str, object]) -> pd.DataFrame:
    rows = []
    tree_name = next(name for name in ("Extra trees", "Random forest") if name in models)
    forest = models[tree_name]
    rows.extend(
        {"model": tree_name, "feature": feature, "value": value, "measure": "impurity importance"}
        for feature, value in zip(FEATURE_COLUMNS, forest.feature_importances_)
    )
    logistic = models["Logistic regression"].named_steps["classifier"]
    rows.extend(
        {"model": "Logistic regression", "feature": feature, "value": value, "measure": "standardized coefficient"}
        for feature, value in zip(FEATURE_COLUMNS, logistic.coef_[0])
    )
    return pd.DataFrame(rows)
