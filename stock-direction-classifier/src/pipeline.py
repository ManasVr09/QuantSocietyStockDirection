from __future__ import annotations

import argparse
import json
import shutil
import pandas as pd

from .config import ROOT, load_config
from .data import download_prices
from .evaluate import calibration_table, feature_analysis, metric_table, predict_all, reliability_table, rule_baseline_predictions
from .backtest import BacktestConfig, block_bootstrap_economics, simulate
from .experiment import build_manifest, fold_assignments
from .features import build_features
from .train import chronological_split, fit_final_models


def run(config_path: str | None = None, refresh_data: bool = False) -> None:
    config = load_config(config_path)
    processed_dir = ROOT / "data" / "processed"
    predictions_dir = ROOT / "data" / "predictions"
    tables_dir = ROOT / "reports" / "tables"
    for directory in (processed_dir, predictions_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["_config_path"], tables_dir / "experiment_config.yaml")

    snapshot_path = ROOT / "data" / "raw" / "prices.parquet"
    if refresh_data or not snapshot_path.exists():
        print("Downloading and freezing adjusted daily prices...")
        prices = download_prices(config)
    else:
        print(f"Using immutable local snapshot: {snapshot_path}")
        prices = pd.read_parquet(snapshot_path)
    print("Building leak-safe features and next-day targets...")
    featured = build_features(prices, config["tickers"], config["market_ticker"])
    featured.to_parquet(processed_dir / "features.parquet", index=False)
    splits = chronological_split(featured, config)
    print(f"Rows — train: {len(splits.train):,}, validation: {len(splits.validation):,}, test: {len(splits.test):,}")
    print("Selecting and fitting models...")
    models, thresholds, selection, champion = fit_final_models(splits, config)
    predictions = pd.concat([predict_all(models, splits.test, thresholds), rule_baseline_predictions(splits.test)], ignore_index=True)
    metrics = metric_table(predictions)
    predictions.to_parquet(predictions_dir / "test_predictions.parquet", index=False)
    metrics.to_csv(tables_dir / "metrics.csv", index=False)
    selection.to_csv(tables_dir / "model_validation.csv", index=False)
    calibration_table(predictions).to_csv(tables_dir / "calibration.csv", index=False)
    reliability_table(
        predictions,
        bootstrap_samples=config["validation"]["bootstrap_samples"],
        block_length=config["validation"]["block_length"],
    ).to_csv(tables_dir / "reliability.csv", index=False)
    feature_analysis(models).to_csv(tables_dir / "feature_analysis.csv", index=False)
    fold_assignments(splits).to_parquet(tables_dir / "fold_assignments.parquet", index=False)

    official = predictions[predictions["model"] == champion].copy()
    economic_rows = []
    official_report = None
    for cost_bps in config["execution"]["cost_scenarios_bps"]:
        report = simulate(official, BacktestConfig(
            initial_capital=config["execution"]["initial_capital"],
            max_position=config["execution"]["max_position"],
            transaction_cost_bps=float(cost_bps),
            slippage_bps=config["execution"]["slippage_bps"],
            probability_threshold=thresholds[champion],
        ))
        economic_rows.append({"model": champion, "transaction_cost_bps": cost_bps, **report.metrics})
        if float(cost_bps) == float(config["execution"]["transaction_cost_bps"]):
            official_report = report
    pd.DataFrame(economic_rows).to_csv(tables_dir / "economic_metrics.csv", index=False)
    if official_report is None:
        official_report = report
    official_report.ledger.to_parquet(predictions_dir / "trade_ledger.parquet", index=False)
    official_report.daily_portfolio.to_parquet(predictions_dir / "daily_portfolio.parquet", index=False)
    official_report.per_stock.to_csv(tables_dir / "economic_by_stock.csv", index=False)
    uncertainty = block_bootstrap_economics(
        official_report.daily_portfolio,
        samples=config["validation"]["bootstrap_samples"],
        block_length=config["validation"]["block_length"],
        seed=config["models"]["random_state"],
    )
    pd.DataFrame([{"model": champion, **uncertainty}]).to_csv(tables_dir / "economic_uncertainty.csv", index=False)

    metadata_path = ROOT / "data" / "raw" / "metadata.json"
    data_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    manifest = build_manifest(config, champion, thresholds[champion], data_metadata)
    (tables_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "train_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "test_start": str(splits.test["date"].min().date()),
        "test_end": str(splits.test["date"].max().date()),
        "tickers": config["tickers"],
        "universe_note": config.get("universe_note", ""),
        "universe_groups": config.get("universe_groups", {}),
        "research_status": config["research_status"],
        "validation_declared_champion": champion,
        "validation_selected_threshold": thresholds[champion],
        "execution_contract": config["execution"],
    }
    (tables_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    label = "Post-hoc subset" if config.get("universe_note") else "Untouched"
    print(f"\n{label} {config['splits']['test_start'][:4]} test results:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nArtifacts saved. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the stock direction experiment.")
    parser.add_argument("--config", default=None, help="Path to a YAML project configuration.")
    parser.add_argument("--refresh-data", action="store_true", help="Download a new raw snapshot instead of reusing the existing one.")
    args = parser.parse_args()
    run(args.config, refresh_data=args.refresh_data)
