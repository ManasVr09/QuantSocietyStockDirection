from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features import FEATURE_COLUMNS


def predict_all(models: dict[str, object], test: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    base_columns = [
        "date", "ticker", "close", "target", "next_return", "gross_return",
        "feature_time", "signal_time", "target_date", "entry_time", "exit_time",
        "entry_price", "exit_price", *FEATURE_COLUMNS,
    ]
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


def rule_baseline_predictions(test: pd.DataFrame) -> pd.DataFrame:
    """Transparent rules using the exact same information timestamp as the models."""
    base_columns = [
        "date", "ticker", "close", "target", "next_return", "gross_return",
        "feature_time", "signal_time", "target_date", "entry_time", "exit_time",
        "entry_price", "exit_price", *FEATURE_COLUMNS,
    ]
    rules = {
        "Always UP rule": np.ones(len(test)),
        "Always DOWN rule": np.zeros(len(test)),
        "Previous-direction rule": (test["return_1d"].to_numpy() > 0).astype(float),
        "Market-direction rule": (test["market_return_1d"].to_numpy() > 0).astype(float),
        "5-day momentum rule": (test["return_5d"].to_numpy() > 0).astype(float),
        "1-day mean-reversion rule": (test["return_1d"].to_numpy() < 0).astype(float),
    }
    outputs = []
    for name, probability in rules.items():
        result = test[base_columns].copy()
        result["model"] = name
        result["probability_up"] = probability
        result["decision_threshold"] = 0.5
        result["prediction"] = probability >= 0.5
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
                "log_loss": log_loss(actual, probability),
                "brier_score": brier_score_loss(actual, probability),
            }
        )
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model"):
        observed, predicted = calibration_curve(group["target"], group["probability_up"], n_bins=bins, strategy="quantile")
        bucket = pd.qcut(group["probability_up"], q=min(bins, group["probability_up"].nunique()), duplicates="drop")
        counts = group.groupby(bucket, observed=True).size().to_numpy()
        rows.extend({"model": model, "mean_predicted": x, "observed_up_rate": y, "count": int(count)} for x, y, count in zip(predicted, observed, counts))
    return pd.DataFrame(rows)


def _moving_block_means(values: np.ndarray, samples: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    blocks_needed = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(samples, blocks_needed))
    offsets = np.arange(block_length)
    positions = (starts[..., None] + offsets) % n
    sampled = values[positions].reshape(samples, -1)[:, :n]
    return sampled.mean(axis=1)


def reliability_table(predictions: pd.DataFrame, bootstrap_samples: int = 2000, seed: int = 42, block_length: int = 10) -> pd.DataFrame:
    """Measure consistency with a moving-block bootstrap over trading dates."""
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
        deltas = _moving_block_means(daily_delta, bootstrap_samples, block_length, rng)
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
            "bootstrap_method": f"moving_block_{block_length}_dates",
        })
    return pd.DataFrame(rows).sort_values("accuracy_delta_vs_baseline", ascending=False)


def feature_analysis(models: dict[str, object]) -> pd.DataFrame:
    rows = []
    tree_name = next(name for name in ("Extra trees", "Random forest") if name in models)
    forest = models[tree_name].estimator.estimator
    rows.extend(
        {"model": tree_name, "feature": feature, "value": value, "measure": "impurity importance"}
        for feature, value in zip(FEATURE_COLUMNS, forest.feature_importances_)
    )
    logistic = models["Logistic regression"].estimator.estimator.named_steps["classifier"]
    rows.extend(
        {"model": "Logistic regression", "feature": feature, "value": value, "measure": "standardized coefficient"}
        for feature, value in zip(FEATURE_COLUMNS, logistic.coef_[0])
    )
    return pd.DataFrame(rows)
