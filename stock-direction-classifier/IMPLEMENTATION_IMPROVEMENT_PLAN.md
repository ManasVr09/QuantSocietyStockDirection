# Stock Direction Classifier: Audit and Implementation Improvement Plan

## Executive summary

**Audit verdict: not ready for trading or portfolio-action claims.** The project is a useful educational research pipeline, and its existing tests pass, but two high-impact timing defects invalidate the current strategy interpretation:

1. Signals use the completed close, high, low, and volume for date `t`, then the backtest earns the close(`t`) to close(`t+1`) return. That fill is not executable after those inputs are known.
2. Splits use the feature date rather than the label/outcome date, so next-day labels cross the train/validation/test boundaries.

The selected seven-stock universe was also chosen after reviewing 2025 results. The README discloses this correctly, but the resulting 2025 metrics are descriptive—not a fresh out-of-sample result. The recorded logistic-regression performance (54.29% accuracy versus 53.66% for always-UP, 50.85% balanced accuracy, ROC-AUC 0.520, and an accuracy-difference confidence interval spanning zero) does **not** establish a reliable edge.

“Better results” should mean an executable, leakage-free, cost-aware process that is stable across time and securities—not a higher score obtained through more model complexity or repeated test-set tuning.

## Scope and diagnostics

Reviewed:

- `configs/project.yaml`
- `src/data.py`, `src/features.py`, `src/train.py`, `src/evaluate.py`, `src/backtest.py`, `src/pipeline.py`, and `src/config.py`
- `dashboard/app.py`
- tests, README, saved predictions, and report tables

Diagnostics run on 2026-08-03:

- `.venv/bin/python -m pytest -q`: **4 tests passed**.
- Existing artifacts contain 1,750 test observations across seven stocks.
- Logistic regression: accuracy 0.5429, balanced accuracy 0.5085, ROC-AUC 0.5201.
- Majority baseline: accuracy 0.5366, balanced accuracy 0.5000.
- Date-clustered 95% interval for logistic accuracy improvement: approximately **-0.46% to +1.71%**.

## What is already sound

- Features are calculated within ticker boundaries (`src/features.py:49-70`), and tests cover this.
- Scaling is inside the logistic-regression pipeline (`src/train.py:44-49`), avoiding full-sample scaler leakage.
- Model fitting is chronologically separated at a coarse date level (`src/train.py:27-37`).
- Hyperparameters and thresholds are selected on validation data rather than directly inside the fitting call (`src/train.py:66-116`).
- Predictions store probabilities, thresholds, labels, and forward returns for reproducibility (`src/evaluate.py:18-29`).
- The README and dashboard explicitly disclose post-hoc universe selection (`README.md:15-24`; `dashboard/app.py:97-106`).
- Reliability resampling clusters stocks observed on the same date (`src/evaluate.py:61-89`), which is better than treating all stock-days as independent.

## Priority findings

| Priority | Finding type | Location | Finding | Why it matters | Required fix |
|---|---|---|---|---|---|
| P0 | Formula/control defect | `src/features.py:53-69`, `src/backtest.py:8-10` | Same-day completed OHLCV features generate a position that earns the same close-to-next-close return. | The assumed entry price is unavailable once all inputs are known; strategy returns contain execution look-ahead. | Define a decision timestamp and align features, target, fill, holding period, and costs to an executable clock. |
| P0 | Formula/control defect | `src/features.py:68-69`, `src/train.py:27-36` | Labels use the next observation, but splits use only feature date. | Outcomes cross split boundaries and contaminate fitting/selection. | Store `target_date`; purge rows whose label date is outside the partition and add an embargo where required. |
| P0 | Unsupported assumption | `configs/project.yaml:3-14` | Seven stocks were selected after reviewing the 2025 results. | Aggregate 2025 performance is post-selection contaminated. Cross-sectional ranks also change when the selected universe changes. | Freeze a predeclared broad universe and evaluate on a new future period, or label all 2025 subset work exploratory. |
| P1 | Unsupported assumption | `src/train.py:86-91` | Threshold maximizes pooled validation balanced accuracy. | Classification accuracy is not the economic objective and ignores costs, turnover, calibration, and abstention. | Select threshold/policy using nested walk-forward validation and a cost-aware objective; include a no-trade zone. |
| P1 | Missing decision output | `dashboard/app.py:221-231` | User can tune the probability threshold while viewing test-period returns. | This encourages visual test-set overfitting. | Lock the validation-selected threshold for confirmatory results; label slider output exploratory and segregate it from untouched evaluation. |
| P1 | Unsupported assumption | `src/evaluate.py:61-89` | Bootstrap resamples single dates independently. | It preserves cross-sectional dependence but not serial or regime dependence. | Use moving/stationary block bootstrap and report uncertainty by stock, date, and regime. |
| P1 | Formula/control defect | `src/backtest.py:6-15` | Final liquidation cost is not explicit; execution delay, slippage, spread, and capacity are absent. | Small apparent edges can disappear after implementable costs. | Build a timestamped trade ledger with entry/exit prices, one-way turnover, terminal liquidation, and cost scenarios. |
| P2 | Output presentation | `dashboard/app.py:88-95` | “Best learned model” is chosen by test accuracy for the overview. | Choosing the headline after seeing test results adds another post-test selection layer. | Declare the champion from validation before test scoring and show all test results without reranking the research conclusion. |
| P2 | Model architecture | `src/train.py`, `src/evaluate.py` | Calibration, feature selection, policy selection, and evaluation are not represented as a single fitted research object. | It is easy to apply mismatched thresholds or transformations. | Persist a versioned experiment bundle with model, calibrator, feature schema, timestamps, universe, policy, and provenance. |
| P2 | Streamlit/runtime | `dashboard/app.py:21-58,114-204` | Custom CSS, deprecated `use_container_width`, horizontal radio, Plotly-heavy charts, and eager tabs remain. | Creates maintenance and rerun costs and diverges from current Streamlit APIs. | Prefer theme/native elements, `width`, segmented controls, cached artifacts, and conditionally rendered heavy views. |

## Phase 0: repair time semantics and split integrity

### 0.1 Choose one executable prediction contract

Recommended daily contract:

```text
information cutoff: after official close on date t
signal timestamp:   close(t) + processing delay
earliest fill:       open(t+1)
exit:                close(t+1) or open(t+2), declared in advance
target:              return between those executable prices
```

If the research goal must remain close(`t`) to close(`t+1`), lag every feature one full observation so the position for that interval is determined no later than close(`t-1`). Do not assume a market-on-close order can use final close/high/low/volume values.

Add explicit columns:

```python
feature_time
signal_time
entry_time
exit_time
target_date
entry_price
exit_price
gross_return
```

### 0.2 Purge by outcome time

For each partition, require both feature and outcome to belong to the allowed information set:

```python
train = rows[(feature_time <= train_cutoff) & (target_date <= train_cutoff)]
validation = rows[(feature_time >= val_start) & (target_date <= val_end)]
test = rows[(feature_time >= test_start) & (target_date <= test_end)]
```

For horizon `h`, apply a purge/embargo of at least `h` trading observations between fitting and scoring folds. Persist fold membership and verify that no `target_date` from a fitting fold reaches a validation/test fold.

### 0.3 Acceptance criteria

- Every prediction has an explicit information cutoff and executable entry after that cutoff.
- `max(train.target_date) < min(validation.feature_time)` and `max(validation.target_date) < min(test.feature_time)` for the confirmatory split.
- No 2025 outcome is used in training or validation for the 2025 confirmatory test.
- A hand-built timing fixture proves that changing future OHLCV cannot change an earlier signal.
- Strategy returns reconcile exactly to the stored entry/exit prices.

## Phase 1: establish an honest experimental design

### 1.1 Freeze the universe and research protocol

- Replace the outcome-selected subset with a predeclared liquid universe or a point-in-time constituent dataset.
- Preserve delisted names where relevant to avoid survivorship bias.
- Fit cross-sectional ranks against the frozen point-in-time universe, not a subset chosen after results.
- Treat 2025 as exploratory because it has already been inspected. Use a genuinely unseen future window or nested walk-forward pseudo-out-of-sample evaluation for confirmation.
- Version a research manifest containing data retrieval time, symbols, corporate-action policy, feature schema, fold dates, models, hyperparameters, thresholds, cost assumptions, and code commit.

### 1.2 Use nested walk-forward evaluation

Recommended structure:

```text
outer fold: produces untouched pseudo-OOS predictions
inner folds: choose hyperparameters, calibrator, threshold/policy
purge/embargo: based on label horizon
refit cadence: monthly or quarterly, declared before evaluation
```

Aggregate only predictions produced when that observation was outside all fitting and selection steps. Retain a final untouched holdout only for one confirmatory read.

### 1.3 Baselines

Keep the majority classifier, and add:

- Always-UP and always-DOWN by stock and pooled.
- Previous-direction and market-direction rules.
- A simple momentum/mean-reversion rule using the same information timestamp.
- Logistic regression with intercept only and with a small preregistered feature set.
- Buy-and-hold and cash for economic comparisons.

The champion must beat relevant baselines after costs and uncertainty, not merely rank first among weak candidates.

## Phase 2: improve probabilities and decision rules

1. Measure log loss and Brier score in addition to ROC-AUC and balanced accuracy.
2. Fit calibration (Platt or isotonic) only inside inner training/validation folds; never calibrate on the outer test fold.
3. Report calibration slope/intercept and expected calibration error with sample counts per bin.
4. Replace a single directional cutoff with an abstaining policy:

```text
long if expected net return > cost + safety margin
cash otherwise
```

For a symmetric direction classifier, a probability threshold is justified only if it maps to expected return/payoff. Estimate expected conditional return, not just class probability, or add a regression/ranking target alongside classification.

5. Select policies using an inner-fold economic score penalized for turnover and instability. Do not optimize Sharpe on the final test set.
6. Use simple models first: regularized logistic regression, monotonic/generalized additive models, or shallow boosted trees. Add complexity only if repeated outer folds show stable incremental value.
7. Evaluate performance by stock, sector, volatility regime, year, bull/bear market, and probability bucket; require the sign of the edge to be reasonably stable.

## Phase 3: implement an executable backtest

Create a pure backtest API independent of Streamlit:

```python
BacktestConfig(
    entry_rule,
    exit_rule,
    initial_capital,
    max_position,
    transaction_cost_bps,
    slippage_bps,
    borrow_bps,
    execution_delay,
)

simulate(predictions, prices, config) -> BacktestReport
```

The report should include a trade ledger and daily portfolio series. Charge costs on absolute one-way turnover and explicitly close residual positions at the end. Add conservative cost scenarios rather than one precise assumption.

Required metrics:

- CAGR/total return, annualized volatility, Sharpe and Sortino (with clear conventions).
- Maximum drawdown, time under water, downside tail/expected shortfall.
- Turnover, trade count, hit rate, average win/loss, exposure, and capacity proxy.
- Gross and net results at 0, 5, 10, and 25 bps one-way costs.
- Equal-weight pooled portfolio plus per-stock results; do not compound interleaved stock rows as one path.
- Block-bootstrap confidence intervals for return, Sharpe, and baseline difference.

Acceptance criteria:

- No trade occurs before its signal timestamp.
- A zero-signal policy returns cash minus no costs.
- A buy-and-hold fixture matches a hand calculation including entry and liquidation costs.
- Increasing transaction costs cannot increase net wealth.
- Reordering input rows does not change results.
- Duplicate timestamps, missing prices, and overlapping positions fail explicitly.

## Phase 4: reusable research architecture

Suggested layout:

```text
src/
  schemas.py       # typed observations, folds, experiment bundle
  data.py          # provider adapter, point-in-time validation, metadata
  features.py      # fit/transform feature builders with timing metadata
  splits.py        # purged walk-forward folds
  models.py        # candidate factories
  calibration.py   # fold-safe calibration
  policy.py        # probability-to-position rules
  backtest.py      # execution and ledger
  metrics.py       # statistical and economic metrics
  experiment.py    # orchestration and artifact manifest
```

Separate data retrieval from experiment execution. Allow the pipeline to run from an immutable local data snapshot so repeated runs do not silently change history. Validate config with a typed schema and reject overlapping dates, incomplete horizons, unknown features, and invalid costs.

Persist one versioned bundle per run:

```text
run_id/
  manifest.json
  config.yaml
  data_metadata.json
  fold_assignments.parquet
  out_of_fold_predictions.parquet
  final_test_predictions.parquet
  model_bundle.joblib
  metrics.parquet
  trade_ledger.parquet
```

## Phase 5: dashboard and reporting changes

- Make the first page lead with audit posture: exploratory/post-hoc versus untouched confirmatory.
- Show the validation-declared champion and policy; do not choose the headline from test accuracy.
- Separate “confirmatory results” from an “exploratory sandbox.” Threshold changes in the sandbox must never overwrite or visually masquerade as official metrics.
- Show log loss, Brier score, calibration, net economic results, turnover, drawdown, and uncertainty beside accuracy.
- Replace deprecated `use_container_width` with current `width` behavior.
- Prefer a theme file and native bordered metrics/containers over injected CSS.
- Prefer `st.segmented_control` over horizontal radio navigation and use `st.navigation`/`st.Page` if the app is split into pages.
- Cache artifact loading with an explicit bound/TTL where appropriate. Render stable page chrome before slow calculations.
- Avoid eagerly computing all heavy tab contents; guard them or use separate views/fragments.
- Keep business logic out of `dashboard/app.py`; the UI should consume immutable report objects.

## Test plan

### Timing and leakage

- Assert each feature uses source timestamps `<= feature_time`.
- Assert `entry_time > signal_time` and returns use exactly entry/exit prices.
- Assert target dates cannot cross fold boundaries.
- Perturb all future prices and confirm earlier features/predictions are unchanged.
- Verify cross-sectional ranks use the frozen point-in-time universe.

### Model selection and calibration

- Confirm scalers, selectors, calibrators, thresholds, and models are fit only on inner-fold data.
- Confirm every outer-fold prediction is generated by an object that never saw that label.
- Confirm the declared champion is selected without outer-test metrics.
- Check probability bounds, Brier/log-loss calculations, and calibration-bin counts.

### Backtest

- Use deterministic price paths and signals with hand-calculated trades, turnover, costs, and wealth.
- Test initial entry and terminal liquidation costs.
- Test no-trade, always-long, alternating, missing-session, split, and dividend-adjustment cases.
- Test portfolio aggregation separately from per-stock compounding.

### Statistical robustness

- Use stationary/moving block bootstrap with deterministic seeds.
- Run placebo labels and time-shuffled tests; performance should collapse toward baseline.
- Report sensitivity to universe, period, refit cadence, feature set, and cost assumptions.
- Add multiple-testing controls or a deflated/probabilistic Sharpe analysis when many variants are tried.

### Dashboard

- Streamlit AppTest for missing artifacts, normal load, navigation, confirmatory/exploratory labeling, and threshold sandbox behavior.
- Assert changing a chart-only control does not recompute models or mutate official results.
- Assert no deprecated `use_container_width` remains.

## Promotion gates

Do not call the algorithm improved or tradable until all of the following hold:

1. Timing invariants and purged splits pass.
2. Universe, folds, features, model candidates, policy, and costs are frozen before confirmatory evaluation.
3. The model improves a relevant baseline across multiple outer folds and stocks, not only in aggregate.
4. Probability quality improves out of sample (Brier/log loss and calibration), not just thresholded accuracy.
5. Net performance remains positive under conservative costs and has confidence intervals that exclude an economically trivial edge.
6. No single ticker, month, or regime explains the majority of performance.
7. A new untouched period confirms the result without retuning.

Until then, preserve the current educational disclaimer and state: **do not use for portfolio action until remediated and re-audited.**
