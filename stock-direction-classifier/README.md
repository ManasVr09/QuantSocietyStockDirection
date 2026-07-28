# Stock Direction Classifier

An educational machine-learning project that asks:

> Can historical price, volume, and technical indicators predict whether a stock's next trading-day adjusted close will be up or down?

The pipeline uses a chronological experiment:

- Train: 2016-2022
- Validation/model and threshold selection: 2023-2024
- Final untouched test: 2025
- Models: majority-class baseline, logistic regression, random forest, and Extra Trees candidates
- Dashboard: Streamlit + Plotly

## Seven-stock universe

The focused dashboard contains:

- Supporting cases: AAPL, AMZN, META, MSFT, and NVDA
- Counterexamples: JNJ and WMT

“Supporting” means logistic regression beat that stock's always-UP accuracy in the original ten-stock 2025 evaluation. “Counterexample” means it did not. This seven-stock universe was therefore selected after observing 2025 outcomes. Results for the subset are descriptive and cannot be presented as a new untouched out-of-sample test. A later study needs a new future period or a pre-registered rolling evaluation.

This is a historical research experiment, not investment advice or a live trading signal.

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
- `models/`: fitted model bundles
- `reports/tables/`: metrics, feature analysis, and validation selection results

## Methodology

Features at date `t` use only values known by the close of `t`. The label is 1 when adjusted close at `t+1` is greater than adjusted close at `t`. The last observation for every ticker is excluded because its future outcome is unknown.

The 2023-2024 split selects the tree family, hyperparameters, and decision thresholds using both balanced accuracy and ROC-AUC across separate calendar years. The 2025 split is evaluated once using models refit on all pre-2025 observations. Scaling for logistic regression is contained inside its fitted pipeline.

The strategy page is deliberately simple: it holds a stock for one close-to-close period only when the model's UP probability exceeds a threshold. Transaction costs are charged when the position changes. This is illustrative and omits slippage, taxes, market impact, and execution constraints.

## Tests

```bash
pytest
```

Tests cover next-day target alignment, per-ticker rolling isolation, feature timing, and chronological splits.

## Limitations

- Yahoo Finance is convenient for a prototype but is not an institutional market-data source.
- Adjusted OHLC reconstruction is approximate because Yahoo's adjustment factor is applied to OHLC.
- Daily observations are limited and market regimes change.
- Technical features are correlated and feature importance is not causal evidence.
- A small predictive edge may disappear after trading costs.
