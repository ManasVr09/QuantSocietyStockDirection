# Stock Direction Classifier

An educational machine-learning project that asks:

> Can information available after today's close predict whether a stock rises from the next session's executable open to that session's close?

The pipeline uses a chronological experiment:

- Train: 2016-2022
- Validation/model and threshold selection: 2023-2024
- Exploratory/post-hoc evaluation: 2025 (already inspected; not confirmatory)
- Models: majority-class baseline, logistic regression, random forest, and Extra Trees candidates
- Dashboard: Streamlit + Plotly

## Seven-stock universe

The focused dashboard contains:

- Supporting cases: AAPL, AMZN, META, MSFT, and NVDA
- Counterexamples: JNJ and WMT

“Supporting” means logistic regression beat that stock's always-UP accuracy in the original ten-stock 2025 evaluation. “Counterexample” means it did not. This seven-stock universe was therefore selected after observing 2025 outcomes. Results for the subset are descriptive and cannot be presented as a new untouched out-of-sample test. A later study needs a new future period or a pre-registered rolling evaluation.

This is a historical research experiment, not investment advice or a live trading signal. **Do not use for portfolio action until remediated further and re-audited on a genuinely new period.**

## Quick start

Python 3.11 is recommended.

```bash
cd stock-direction-classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
streamlit run dashboard/app.py
```

## Dashboard design

The Quant Society interface uses a restrained light research layout with a left-hand navigation rail, a centred 1200 px workspace, readable 16 px base typography, headline metrics, and interactive historical charts. Tables use the available width, taller rows, and focused groups of metrics rather than squeezing every column into one view. All six research views remain available. Stock selection changes the probability chart; overview metrics continue to describe the full seven-stock universe. Methodology and full model comparisons are available in expanders.

Run with Streamlit 1.60 or later. The theme is included both in this project's `.streamlit/config.toml` for local runs and the repository-root `.streamlit/config.toml` for Community Cloud. No custom CSS or external component service is required.

## Deploy on Streamlit Community Cloud

The deployment uses the committed dashboard-ready prediction and report artifacts; it does not download data or retrain models when the app starts.

1. Push the latest project files to GitHub.
2. Sign in at `https://share.streamlit.io` with GitHub and select **Create app**.
3. Choose the `ManasVr09/QuantSocietyStockDirection` repository and the branch containing these files.
4. Set the entrypoint to `stock-direction-classifier/dashboard/app.py`.
5. Under **Advanced settings**, select Python 3.12. No secrets are required for this historical dashboard.
6. Click **Deploy**.

Community Cloud finds the runtime dependencies in `stock-direction-classifier/dashboard/requirements.txt`. The files under `data/predictions/` and the required files under `reports/tables/` must remain committed because the dashboard loads them at startup.

The pipeline downloads daily adjusted OHLCV data from Yahoo Finance, writes raw and processed Parquet files, fits the models, and saves dashboard-ready artifacts. It never retrains from the dashboard.

To use another config:

```bash
python -m src.pipeline --config configs/project.yaml
```

## Outputs

- `data/raw/`: source data and retrieval metadata
- `data/processed/features.parquet`: leak-safe modeling table
- `data/predictions/test_predictions.parquet`: 2025 out-of-sample predictions
- `data/predictions/trade_ledger.parquet`: timestamped official-policy executions
- `data/predictions/daily_portfolio.parquet`: equal-weight daily portfolio path
- `models/`: fitted model bundles
- `reports/tables/manifest.json`: versioned research protocol and provenance
- `reports/tables/`: classification, probability, reliability, economic, fold, and feature reports

## Methodology

The executable prediction contract is fixed in configuration:

```text
information cutoff: after close(t)
signal:             close(t) + five-minute processing delay
entry:              open(t+1)
exit:               close(t+1)
target:             entry-to-exit return
```

Each row persists `feature_time`, `signal_time`, `target_date`, `entry_time`, `exit_time`, `entry_price`, `exit_price`, and `gross_return`. A row is assigned to a split only when both its feature time and outcome date belong to that partition. Labels that cross a boundary are purged.

The 2023-2024 validation split selects the tree family, hyperparameters, champion, and an abstaining long/cash threshold. Threshold selection maximizes a validation-only net-return objective after round-trip transaction costs and slippage with an instability penalty. Scaling for logistic regression remains inside its fitted pipeline. The selected champion and threshold are written to the manifest before 2025 metrics are presented.

The official backtest is independent of Streamlit. It enters only after the stored signal timestamp, charges both entry and terminal exit costs, creates a trade ledger, and aggregates simultaneous stock trades as an equal-weight daily portfolio. It reports gross/net results under 0, 5, 10, and 25 bps one-way transaction-cost scenarios plus declared slippage. The dashboard's official page is locked; user-controlled thresholds live only in an explicitly exploratory sandbox.

Pipeline runs reuse the immutable local raw snapshot by default. Use `python -m src.pipeline --refresh-data` only when intentionally creating a new snapshot.

## Tests

```bash
pytest
```

Tests cover executable target alignment, future-price perturbation, ticker isolation, purged splits, trade timing, round-trip costs, monotonic cost sensitivity, row-order invariance, dashboard loading, and audit labeling.

## Limitations

- Yahoo Finance is convenient for a prototype but is not an institutional market-data source.
- Adjusted OHLC reconstruction is approximate because Yahoo's adjustment factor is applied to OHLC.
- Daily observations are limited and market regimes change.
- Technical features are correlated and feature importance is not causal evidence.
- A small predictive edge may disappear after trading costs.
- The project does not yet implement a full nested walk-forward outer loop or fold-safe probability calibrator; these remain requirements before any confirmatory promotion.
- The seven-stock universe is post-selection contaminated, so 2025 cannot validate it regardless of improved timing controls.
- Taxes, borrow, market impact, capacity, and point-in-time index membership remain outside the current prototype.

## Performance

See the [performance notes](PERFORMANCE.md) for caching, lazy loading, and preservation guarantees.
