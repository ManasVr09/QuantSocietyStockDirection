from __future__ import annotations

import pandas as pd


def simulate_strategy(predictions: pd.DataFrame, threshold: float = 0.55, transaction_cost_bps: float = 5.0) -> pd.DataFrame:
    data = predictions.sort_values(["ticker", "date"]).copy()
    data["position"] = (data["probability_up"] >= threshold).astype(float)
    data["turnover"] = data.groupby("ticker")["position"].diff().abs().fillna(data["position"])
    data["strategy_return"] = data["position"] * data["next_return"] - data["turnover"] * transaction_cost_bps / 10_000
    data["buy_hold_curve"] = data.groupby("ticker")["next_return"].transform(lambda x: (1 + x).cumprod())
    data["strategy_curve"] = data.groupby("ticker")["strategy_return"].transform(lambda x: (1 + x).cumprod())
    data["strategy_peak"] = data.groupby("ticker")["strategy_curve"].cummax()
    data["strategy_drawdown"] = data["strategy_curve"] / data["strategy_peak"] - 1
    return data

