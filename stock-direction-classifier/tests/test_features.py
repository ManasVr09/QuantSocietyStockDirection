import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_features


def make_prices(ticker: str, start: str, periods: int, first: float, step: float) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    close = first + np.arange(periods) * step
    return pd.DataFrame({"date": dates, "ticker": ticker, "open": close, "high": close, "low": close, "close": close, "volume": 1_000 + np.arange(periods)})


def test_target_uses_next_trading_observation():
    prices = pd.concat([make_prices("AAA", "2020-01-01", 260, 100, 1), make_prices("SPY", "2020-01-01", 260, 200, 1)])
    result = build_features(prices, ["AAA"], "SPY")
    row = result.iloc[0]
    source = prices[(prices["ticker"] == "AAA")].sort_values("date").reset_index(drop=True)
    idx = source.index[source["date"] == row["date"]][0]
    expected = source.loc[idx + 1, "close"] / source.loc[idx, "close"] - 1
    assert np.isclose(row["next_return"], expected)
    assert row["target"] == int(expected > 0)


def test_last_ticker_observation_has_no_label_and_is_removed():
    prices = pd.concat([make_prices("AAA", "2020-01-01", 260, 100, 1), make_prices("SPY", "2020-01-01", 260, 200, 1)])
    result = build_features(prices, ["AAA"], "SPY")
    assert result["date"].max() < prices.loc[prices["ticker"] == "AAA", "date"].max()


def test_rolling_features_do_not_cross_tickers():
    prices = pd.concat([
        make_prices("AAA", "2020-01-01", 260, 100, 1),
        make_prices("BBB", "2020-01-01", 260, 10_000, -10),
        make_prices("SPY", "2020-01-01", 260, 200, 1),
    ])
    result = build_features(prices, ["AAA", "BBB"], "SPY")
    first_b = result[result["ticker"] == "BBB"].iloc[0]
    assert first_b["return_1d"] < 0
    assert result[FEATURE_COLUMNS].notna().all().all()
