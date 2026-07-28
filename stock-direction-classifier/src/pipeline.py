from __future__ import annotations

import argparse
import json
import shutil

from .config import ROOT, load_config
from .data import download_prices
from .evaluate import calibration_table, feature_analysis, metric_table, predict_all, reliability_table
from .features import build_features
from .train import chronological_split, fit_final_models


def run(config_path: str | None = None) -> None:
    config = load_config(config_path)
    processed_dir = ROOT / "data" / "processed"
    predictions_dir = ROOT / "data" / "predictions"
    tables_dir = ROOT / "reports" / "tables"
    for directory in (processed_dir, predictions_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["_config_path"], tables_dir / "experiment_config.yaml")

    print("Downloading adjusted daily prices...")
    prices = download_prices(config)
    print("Building leak-safe features and next-day targets...")
    featured = build_features(prices, config["tickers"], config["market_ticker"])
    featured.to_parquet(processed_dir / "features.parquet", index=False)
    splits = chronological_split(featured, config)
    print(f"Rows — train: {len(splits.train):,}, validation: {len(splits.validation):,}, test: {len(splits.test):,}")
    print("Selecting and fitting models...")
    models, thresholds, selection = fit_final_models(splits, config)
    predictions = predict_all(models, splits.test, thresholds)
    metrics = metric_table(predictions)
    predictions.to_parquet(predictions_dir / "test_predictions.parquet", index=False)
    metrics.to_csv(tables_dir / "metrics.csv", index=False)
    selection.to_csv(tables_dir / "model_validation.csv", index=False)
    calibration_table(predictions).to_csv(tables_dir / "calibration.csv", index=False)
    reliability_table(predictions).to_csv(tables_dir / "reliability.csv", index=False)
    feature_analysis(models).to_csv(tables_dir / "feature_analysis.csv", index=False)
    summary = {
        "train_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "test_start": str(splits.test["date"].min().date()),
        "test_end": str(splits.test["date"].max().date()),
        "tickers": config["tickers"],
        "universe_note": config.get("universe_note", ""),
        "universe_groups": config.get("universe_groups", {}),
    }
    (tables_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    label = "Post-hoc subset" if config.get("universe_note") else "Untouched"
    print(f"\n{label} {config['splits']['test_start'][:4]} test results:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nArtifacts saved. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the stock direction experiment.")
    parser.add_argument("--config", default=None, help="Path to a YAML project configuration.")
    run(parser.parse_args().config)
