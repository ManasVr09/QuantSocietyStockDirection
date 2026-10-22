from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "overnight_return",
    "intraday_return",
    "daily_range",
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "price_to_ma_5",
    "price_to_ma_20",
    "ma_5_to_20",
    "rsi_14",
    "volatility_5",
    "volatility_20",
    "volatility_ratio",
    "volume_to_avg_20",
    "volume_change_1d",
    "market_return_1d",
    "market_return_5d",
    "market_volatility_20",
    "market_price_to_ma_200",
    "excess_return_1d",
    "excess_return_5d",
    "return_rank_1d",
    "return_rank_5d",
    "rsi_rank",
    "volatility_rank",
    "day_of_week_sin",
    "day_of_week_cos",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff()
    avg_gain = change.clip(lower=0).rolling(window, min_periods=window).mean()
    avg_loss = -change.clip(upper=0).rolling(window, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    return result.where(avg_loss.ne(0), 100.0)


def _ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    close = group["close"]
    daily_return = close.pct_change()
    group["overnight_return"] = group["open"] / close.shift(1) - 1
    group["intraday_return"] = close / group["open"] - 1
    group["daily_range"] = (group["high"] - group["low"]) / close
    group["return_1d"] = daily_return
    for period in (2, 5, 10, 20):
        group[f"return_{period}d"] = close.pct_change(period)
    group["price_to_ma_5"] = close / close.rolling(5, min_periods=5).mean() - 1
    group["price_to_ma_20"] = close / close.rolling(20, min_periods=20).mean() - 1
    group["ma_5_to_20"] = close.rolling(5, min_periods=5).mean() / close.rolling(20, min_periods=20).mean() - 1
    group["rsi_14"] = _rsi(close)
    group["volatility_5"] = daily_return.rolling(5, min_periods=5).std()
    group["volatility_20"] = daily_return.rolling(20, min_periods=20).std()
    group["volatility_ratio"] = group["volatility_5"] / group["volatility_20"]
    group["volume_to_avg_20"] = group["volume"] / group["volume"].rolling(20, min_periods=20).mean()
    group["volume_change_1d"] = group["volume"].pct_change()
    # Executable contract: all features are known after close(t), the signal is
    # processed after that close, and the trade enters at open(t+1) and exits at
    # close(t+1).  Never earn close(t)->close(t+1) from completed date-t inputs.
    group["feature_time"] = pd.to_datetime(group["date"]).dt.normalize() + pd.Timedelta("16h")
    group["signal_time"] = group["feature_time"] + pd.Timedelta("5min")
    group["target_date"] = pd.to_datetime(group["date"]).shift(-1)
    group["entry_time"] = group["target_date"] + pd.Timedelta("9h30min")
    group["exit_time"] = group["target_date"] + pd.Timedelta("16h")
    group["entry_price"] = group["open"].shift(-1)
    group["exit_price"] = close.shift(-1)
    group["gross_return"] = group["exit_price"] / group["entry_price"] - 1
    # Kept as a compatibility alias for older report consumers.
    group["next_return"] = group["gross_return"]
    group["target"] = np.where(group["gross_return"].notna(), (group["gross_return"] > 0).astype(int), np.nan)
    return group


def build_features(prices: pd.DataFrame, stock_tickers: list[str], market_ticker: str) -> pd.DataFrame:
    # Explicit iteration keeps ticker boundaries obvious and works across pandas 2.x.
    featured = pd.concat(
        [_ticker_features(group.copy()) for _, group in prices.groupby("ticker", sort=False)],
        ignore_index=True,
    )
    market = featured.loc[
        featured["ticker"] == market_ticker,
        ["date", "return_1d", "return_5d", "volatility_20", "close"],
    ].copy()
    market["market_price_to_ma_200"] = market["close"] / market["close"].rolling(200, min_periods=200).mean() - 1
    market = market.rename(columns={
        "return_1d": "market_return_1d",
        "return_5d": "market_return_5d",
        "volatility_20": "market_volatility_20",
    }).drop(columns="close")
    stocks = featured.loc[featured["ticker"].isin(stock_tickers)].merge(
        market, on="date", how="left", validate="many_to_one"
    )
    stocks["excess_return_1d"] = stocks["return_1d"] - stocks["market_return_1d"]
    stocks["excess_return_5d"] = stocks["return_5d"] - stocks["market_return_5d"]
    stocks["return_rank_1d"] = stocks.groupby("date")["return_1d"].rank(pct=True)
    stocks["return_rank_5d"] = stocks.groupby("date")["return_5d"].rank(pct=True)
    stocks["rsi_rank"] = stocks.groupby("date")["rsi_14"].rank(pct=True)
    stocks["volatility_rank"] = stocks.groupby("date")["volatility_20"].rank(pct=True)
    weekday = pd.to_datetime(stocks["date"]).dt.dayofweek
    stocks["day_of_week_sin"] = np.sin(2 * np.pi * weekday / 5)
    stocks["day_of_week_cos"] = np.cos(2 * np.pi * weekday / 5)
    timing_columns = ["feature_time", "signal_time", "target_date", "entry_time", "exit_time", "entry_price", "exit_price", "gross_return"]
    result = stocks.dropna(subset=FEATURE_COLUMNS + ["target", *timing_columns]).copy()
    if not (result["entry_time"] > result["signal_time"]).all():
        raise ValueError("Execution contract violated: entry must occur after signal time.")
    result["target"] = result["target"].astype(int)
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)
