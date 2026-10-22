from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ROOT
from .features import FEATURE_COLUMNS


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(data: pd.DataFrame, config: dict) -> SplitData:
    feature_times = pd.to_datetime(data["feature_time"])
    target_dates = pd.to_datetime(data["target_date"])
    split = config["splits"]
    train_end = pd.Timestamp(split["train_end"]) + pd.Timedelta("1D") - pd.Timedelta("1us")
    validation_start = pd.Timestamp(split["validation_start"])
    validation_end = pd.Timestamp(split["validation_end"]) + pd.Timedelta("1D") - pd.Timedelta("1us")
    test_start = pd.Timestamp(split["test_start"])
    test_end = pd.Timestamp(split["test_end"]) + pd.Timedelta("1D") - pd.Timedelta("1us")
    train = data.loc[(feature_times <= train_end) & (target_dates <= train_end)].copy()
    validation = data.loc[(feature_times >= validation_start) & (target_dates <= validation_end)].copy()
    test = data.loc[(feature_times >= test_start) & (target_dates <= test_end)].copy()
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("One or more chronological splits are empty.")
    if not (train["target_date"].max() < validation["feature_time"].min() and validation["target_date"].max() < test["feature_time"].min()):
        raise ValueError("Chronological splits overlap or are out of order.")
    return SplitData(train, validation, test)


def model_candidates(config: dict) -> dict[str, object]:
    seed = config["models"]["random_state"]
    return {
        "Majority baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=config["models"]["logistic_c"], max_iter=2000, random_state=seed)),
            ]
        ),
    }


def _validation_score(model, validation: pd.DataFrame) -> tuple[float, float, float]:
    """Score each validation year separately so one regime cannot dominate selection."""
    yearly = []
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    for _, indices in validation.groupby(pd.to_datetime(validation["date"]).dt.year).groups.items():
        actual = validation.loc[indices, "target"]
        probability = probabilities[validation.index.get_indexer(indices)]
        yearly.append((balanced_accuracy_score(actual, probability >= 0.5), roc_auc_score(actual, probability)))
    mean_balanced = float(np.mean([item[0] for item in yearly]))
    mean_auc = float(np.mean([item[1] for item in yearly]))
    return mean_balanced + mean_auc - 1.0, mean_balanced, mean_auc


def select_tree_model(train: pd.DataFrame, validation: pd.DataFrame, config: dict) -> tuple[str, dict, pd.DataFrame]:
    records = []
    best_score = -np.inf
    best_params: dict = {}
    best_family = "Random forest"
    families = [
        ("Random forest", RandomForestClassifier, config["models"]["random_forest_candidates"]),
        ("Extra trees", ExtraTreesClassifier, config["models"]["extra_trees_candidates"]),
    ]
    for family, estimator, candidates in families:
        for params in candidates:
            model = estimator(**params, class_weight="balanced", n_jobs=-1, random_state=config["models"]["random_state"])
            model.fit(train[FEATURE_COLUMNS], train["target"])
            score, balanced, auc = _validation_score(model, validation)
            records.append({"family": family, **params, "mean_yearly_balanced_accuracy": balanced, "mean_yearly_roc_auc": auc, "selection_score": score})
            if score > best_score:
                best_score, best_params, best_family = score, dict(params), family
    return best_family, best_params, pd.DataFrame(records).sort_values("selection_score", ascending=False)


def select_threshold(model, validation: pd.DataFrame, config: dict) -> tuple[float, float]:
    """Select an abstaining long/cash policy on validation net returns only."""
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    candidates = np.asarray(config["validation"]["threshold_grid"], dtype=float)
    round_trip_cost = 2 * (config["execution"]["transaction_cost_bps"] + config["execution"]["slippage_bps"]) / 10_000
    scores = []
    for threshold in candidates:
        position = probabilities >= threshold
        net = np.where(position, validation["gross_return"].to_numpy() - round_trip_cost, 0.0)
        daily = pd.Series(net, index=pd.to_datetime(validation["target_date"])).groupby(level=0).mean()
        instability_penalty = daily.std(ddof=0) / np.sqrt(max(len(daily), 1))
        scores.append(float(daily.mean() - instability_penalty))
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def fit_final_models(splits: SplitData, config: dict) -> tuple[dict[str, object], dict[str, float], pd.DataFrame]:
    validation_year = pd.to_datetime(splits.validation["feature_time"]).dt.year
    years = sorted(validation_year.unique())
    if len(years) < 2:
        raise ValueError("At least two validation years are required for fold-safe calibration and policy selection.")
    calibration = splits.validation.loc[validation_year == years[0]].copy()
    policy_validation = splits.validation.loc[validation_year > years[0]].copy()
    best_family, best_params, selection = select_tree_model(splits.train, calibration, config)
    models = model_candidates(config)
    tree_class = RandomForestClassifier if best_family == "Random forest" else ExtraTreesClassifier
    models[best_family] = tree_class(
        **best_params,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config["models"]["random_state"],
    )
    thresholds: dict[str, float] = {}
    calibrated_development: dict[str, object] = {}
    for name, model in models.items():
        development_base = clone(model).fit(splits.train[FEATURE_COLUMNS], splits.train["target"])
        development_model = CalibratedClassifierCV(FrozenEstimator(development_base), method="sigmoid")
        development_model.fit(calibration[FEATURE_COLUMNS], calibration["target"])
        calibrated_development[name] = development_model
        thresholds[name], threshold_score = select_threshold(development_model, policy_validation, config)
        probabilities = development_model.predict_proba(policy_validation[FEATURE_COLUMNS])[:, 1]
        selection = pd.concat([selection, pd.DataFrame([{
            "family": name,
            "selected_threshold": thresholds[name],
            "threshold_validation_net_objective": threshold_score,
            "calibration_year": years[0],
            "policy_validation_years": ",".join(map(str, years[1:])),
            "validation_log_loss": log_loss(policy_validation["target"], probabilities),
        }])], ignore_index=True)
    policy_rows = selection.dropna(subset=["selected_threshold"]).sort_values("threshold_validation_net_objective", ascending=False)
    champion = str(policy_rows.iloc[0]["family"])
    selection["validation_declared_champion"] = selection["family"].eq(champion)
    base_fit = pd.concat([splits.train, calibration]).sort_values(["date", "ticker"])
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fitted: dict[str, object] = {}
    for name, model in models.items():
        fitted_base = clone(model).fit(base_fit[FEATURE_COLUMNS], base_fit["target"])
        fitted_model = CalibratedClassifierCV(FrozenEstimator(fitted_base), method="sigmoid")
        fitted_model.fit(policy_validation[FEATURE_COLUMNS], policy_validation["target"])
        fitted[name] = fitted_model
        joblib.dump({"model": fitted_model, "features": FEATURE_COLUMNS, "decision_threshold": thresholds[name], "calibration_method": "sigmoid", "research_status": config["research_status"]}, model_dir / f"{name.lower().replace(' ', '_')}.joblib")
    return fitted, thresholds, selection, champion
