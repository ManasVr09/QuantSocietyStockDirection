import pandas as pd
import pytest

from src.backtest import BacktestConfig, simulate


def predictions(probabilities=(0.9, 0.1)):
    feature = pd.to_datetime(["2025-01-02 16:00", "2025-01-03 16:00"])
    entry = pd.to_datetime(["2025-01-03 09:30", "2025-01-06 09:30"])
    return pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "feature_time": feature,
        "signal_time": feature + pd.Timedelta("5min"),
        "entry_time": entry,
        "exit_time": entry.normalize() + pd.Timedelta("16h"),
        "entry_price": [100.0, 100.0],
        "exit_price": [110.0, 90.0],
        "gross_return": [0.10, -0.10],
        "probability_up": probabilities,
    })


def test_no_signal_stays_in_cash_without_costs():
    report = simulate(predictions((0.1, 0.1)), BacktestConfig(probability_threshold=0.8))
    assert report.metrics["total_return"] == 0
    assert report.ledger["cost"].sum() == 0


def test_round_trip_cost_and_terminal_liquidation_are_charged():
    report = simulate(predictions(), BacktestConfig(probability_threshold=0.8, transaction_cost_bps=10, slippage_bps=0))
    first = report.ledger.iloc[0]
    assert first["one_way_turnover"] == 2
    assert first["cost"] == pytest.approx(0.002)
    assert first["net_return"] == pytest.approx(0.098)


def test_higher_cost_cannot_increase_wealth_and_order_is_irrelevant():
    low = simulate(predictions(), BacktestConfig(probability_threshold=0.5, transaction_cost_bps=0, slippage_bps=0))
    high = simulate(predictions().iloc[::-1], BacktestConfig(probability_threshold=0.5, transaction_cost_bps=25, slippage_bps=0))
    assert high.daily_portfolio["wealth"].iloc[-1] <= low.daily_portfolio["wealth"].iloc[-1]


def test_trade_before_signal_fails():
    bad = predictions()
    bad["entry_time"] = bad["feature_time"]
    with pytest.raises(ValueError, match="timestamp"):
        simulate(bad, BacktestConfig())
