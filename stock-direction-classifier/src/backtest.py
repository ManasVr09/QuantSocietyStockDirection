from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1.0
    max_position: float = 1.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    probability_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.initial_capital <= 0 or not 0 <= self.max_position <= 1:
            raise ValueError("Initial capital must be positive and max_position must be in [0, 1].")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Costs cannot be negative.")
        if not 0 <= self.probability_threshold <= 1:
            raise ValueError("Probability threshold must be in [0, 1].")


@dataclass
class BacktestReport:
    ledger: pd.DataFrame
    daily_portfolio: pd.DataFrame
    per_stock: pd.DataFrame
    metrics: dict[str, float]


REQUIRED_COLUMNS = {
    "ticker", "feature_time", "signal_time", "entry_time", "exit_time",
    "entry_price", "exit_price", "gross_return", "probability_up",
}


def _max_drawdown(wealth: pd.Series) -> float:
    return float((wealth / wealth.cummax() - 1).min()) if len(wealth) else 0.0


def _economic_metrics(daily: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, float]:
    returns = daily["net_return"] if len(daily) else pd.Series(dtype=float)
    wealth = daily["wealth"] if len(daily) else pd.Series(dtype=float)
    periods = len(returns)
    annualized_vol = float(returns.std(ddof=0) * np.sqrt(252)) if periods else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252)) if periods and returns.std(ddof=0) > 0 else 0.0
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = float(returns.mean() / downside_std * np.sqrt(252)) if len(downside) and downside_std > 0 else 0.0
    total = float(wealth.iloc[-1] - 1) if len(wealth) else 0.0
    cagr = float(wealth.iloc[-1] ** (252 / periods) - 1) if periods and wealth.iloc[-1] > 0 else -1.0
    losses = returns[returns <= returns.quantile(0.05)] if periods else returns
    return {
        "total_return": total,
        "cagr": cagr,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "maximum_drawdown": _max_drawdown(wealth),
        "expected_shortfall_5pct": float(losses.mean()) if len(losses) else 0.0,
        "trade_count": int(ledger["trade"].sum()) if len(ledger) else 0,
        "hit_rate": float(ledger.loc[ledger["trade"], "net_return"].gt(0).mean()) if ledger["trade"].any() else 0.0,
        "exposure": float(ledger["position"].mean()) if len(ledger) else 0.0,
        "average_turnover": float(ledger["one_way_turnover"].mean()) if len(ledger) else 0.0,
    }


def simulate(predictions: pd.DataFrame, config: BacktestConfig) -> BacktestReport:
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Backtest is missing columns: {sorted(missing)}")
    data = predictions.copy()
    for column in ("feature_time", "signal_time", "entry_time", "exit_time"):
        data[column] = pd.to_datetime(data[column])
    if data.duplicated(["ticker", "entry_time"]).any():
        raise ValueError("Duplicate ticker/entry timestamps are not allowed.")
    if not (data["feature_time"] <= data["signal_time"]).all() or not (data["signal_time"] < data["entry_time"]).all() or not (data["entry_time"] < data["exit_time"]).all():
        raise ValueError("A trade timestamp occurs before its information or signal timestamp.")
    if (data[["entry_price", "exit_price"]] <= 0).any().any():
        raise ValueError("Entry and exit prices must be positive.")

    data = data.sort_values(["entry_time", "ticker"]).reset_index(drop=True)
    data["trade"] = data["probability_up"] >= config.probability_threshold
    data["position"] = data["trade"].astype(float) * config.max_position
    # Each signal is a declared next-open to same-day-close round trip. Entry and
    # terminal liquidation each incur one-way turnover and costs.
    data["one_way_turnover"] = 2.0 * data["position"]
    one_way_cost = (config.transaction_cost_bps + config.slippage_bps) / 10_000
    data["gross_strategy_return"] = data["position"] * data["gross_return"]
    data["cost"] = data["one_way_turnover"] * one_way_cost
    data["net_return"] = data["gross_strategy_return"] - data["cost"]
    data["entry_fill"] = data["entry_price"] * (1 + one_way_cost)
    data["exit_fill"] = data["exit_price"] * (1 - one_way_cost)

    daily = data.groupby("entry_time", as_index=False).agg(
        gross_return=("gross_strategy_return", "mean"),
        cost=("cost", "mean"),
        net_return=("net_return", "mean"),
        exposure=("position", "mean"),
    )
    daily["wealth"] = config.initial_capital * (1 + daily["net_return"]).cumprod()
    per_stock_rows = []
    for ticker, group in data.groupby("ticker"):
        stock_returns = group["net_return"]
        stock_wealth = (1 + stock_returns).cumprod()
        per_stock_rows.append({
            "ticker": ticker,
            "total_return": float(stock_wealth.iloc[-1] - 1),
            "maximum_drawdown": _max_drawdown(stock_wealth),
            "trades": int(group["trade"].sum()),
            "exposure": float(group["position"].mean()),
        })
    per_stock = pd.DataFrame(per_stock_rows)
    return BacktestReport(data, daily, per_stock, _economic_metrics(daily, data))


def simulate_strategy(predictions: pd.DataFrame, threshold: float = 0.55, transaction_cost_bps: float = 5.0) -> pd.DataFrame:
    """Compatibility wrapper returning the ledger for legacy callers."""
    report = simulate(predictions, BacktestConfig(probability_threshold=threshold, transaction_cost_bps=transaction_cost_bps))
    return report.ledger


def block_bootstrap_economics(daily: pd.DataFrame, samples: int = 2000, block_length: int = 10, seed: int = 42) -> dict[str, float]:
    """Moving-block uncertainty for total return and annualized Sharpe."""
    values = daily["net_return"].to_numpy(dtype=float)
    if not len(values):
        return {"total_return_ci_low": 0.0, "total_return_ci_high": 0.0, "sharpe_ci_low": 0.0, "sharpe_ci_high": 0.0}
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(values) / block_length))
    starts = rng.integers(0, len(values), size=(samples, blocks))
    positions = (starts[..., None] + np.arange(block_length)) % len(values)
    sampled = values[positions].reshape(samples, -1)[:, : len(values)]
    total = np.prod(1 + sampled, axis=1) - 1
    std = sampled.std(axis=1)
    sharpe = np.divide(sampled.mean(axis=1), std, out=np.zeros_like(std), where=std > 0) * np.sqrt(252)
    return {
        "total_return_ci_low": float(np.quantile(total, 0.025)),
        "total_return_ci_high": float(np.quantile(total, 0.975)),
        "sharpe_ci_low": float(np.quantile(sharpe, 0.025)),
        "sharpe_ci_high": float(np.quantile(sharpe, 0.975)),
        "bootstrap_method": f"moving_block_{block_length}_dates",
    }
