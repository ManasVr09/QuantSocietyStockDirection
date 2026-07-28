from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
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
    dates = pd.to_datetime(data["date"])
    split = config["splits"]
    train = data.loc[dates <= split["train_end"]].copy()
    validation = data.loc[(dates >= split["validation_start"]) & (dates <= split["validation_end"])].copy()
    test = data.loc[(dates >= split["test_start"]) & (dates <= split["test_end"])].copy()
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("One or more chronological splits are empty.")
    if not (train["date"].max() < validation["date"].min() <= validation["date"].max() < test["date"].min()):
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


def select_threshold(model, validation: pd.DataFrame) -> tuple[float, float]:
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    candidates = np.arange(0.42, 0.581, 0.01)
    scores = [balanced_accuracy_score(validation["target"], probabilities >= threshold) for threshold in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def fit_final_models(splits: SplitData, config: dict) -> tuple[dict[str, object], dict[str, float], pd.DataFrame]:
    best_family, best_params, selection = select_tree_model(splits.train, splits.validation, config)
    models = model_candidates(config)
    tree_class = RandomForestClassifier if best_family == "Random forest" else ExtraTreesClassifier
    models[best_family] = tree_class(
        **best_params,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config["models"]["random_state"],
    )
    thresholds: dict[str, float] = {}
    for name, model in models.items():
        development_model = clone(model).fit(splits.train[FEATURE_COLUMNS], splits.train["target"])
        thresholds[name], threshold_score = select_threshold(development_model, splits.validation)
        selection = pd.concat([selection, pd.DataFrame([{"family": name, "selected_threshold": thresholds[name], "threshold_validation_balanced_accuracy": threshold_score}])], ignore_index=True)
    pre_test = pd.concat([splits.train, splits.validation]).sort_values(["date", "ticker"])
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fitted: dict[str, object] = {}
    for name, model in models.items():
        fitted_model = clone(model).fit(pre_test[FEATURE_COLUMNS], pre_test["target"])
        fitted[name] = fitted_model
        joblib.dump({"model": fitted_model, "features": FEATURE_COLUMNS, "decision_threshold": thresholds[name]}, model_dir / f"{name.lower().replace(' ', '_')}.joblib")
    return fitted, thresholds, selection
