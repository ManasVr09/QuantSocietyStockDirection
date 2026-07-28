from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import ROOT


REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _normalize_download(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Convert yfinance's varying column layouts into a tidy adjusted OHLCV table."""
    if frame.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    pieces: list[pd.DataFrame] = []
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = set(frame.columns.get_level_values(0))
        price_first = "Close" in level0
        for ticker in tickers:
            part = frame.xs(ticker, axis=1, level=1 if price_first else 0).copy()
            part.columns = [str(c).lower().replace(" ", "_") for c in part.columns]
            part["ticker"] = ticker
            pieces.append(part)
    else:
        part = frame.copy()
        part.columns = [str(c).lower().replace(" ", "_") for c in part.columns]
        part["ticker"] = tickers[0]
        pieces.append(part)

    tidy = pd.concat(pieces).reset_index()
    tidy = tidy.rename(columns={tidy.columns[0]: "date", "adj_close": "close"})
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.tz_localize(None)
    tidy["ticker"] = tidy["ticker"].astype(str)
    return tidy[REQUIRED_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_prices(prices: pd.DataFrame, expected_tickers: list[str]) -> None:
    missing_columns = set(REQUIRED_COLUMNS) - set(prices.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date rows found.")
    missing_tickers = set(expected_tickers) - set(prices["ticker"].unique())
    if missing_tickers:
        raise ValueError(f"No observations returned for: {sorted(missing_tickers)}")
    if prices[["close", "volume"]].isna().any().any():
        raise ValueError("Missing close or volume values found.")
    if (prices["close"] <= 0).any() or (prices["volume"] < 0).any():
        raise ValueError("Invalid non-positive price or negative volume found.")


def download_prices(config: dict, output_dir: Path | None = None) -> pd.DataFrame:
    tickers = list(dict.fromkeys(config["tickers"] + [config["market_ticker"]]))
    raw_dir = output_dir or ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = yf.download(
        tickers=tickers,
        start=config["data"]["start"],
        end=config["data"]["end"],
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    prices = _normalize_download(downloaded, tickers)
    validate_prices(prices, tickers)
    prices.to_parquet(raw_dir / "prices.parquet", index=False)
    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": config["data"]["source"],
        "start": config["data"]["start"],
        "end_exclusive": config["data"]["end"],
        "tickers": tickers,
        "rows": len(prices),
    }
    (raw_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return prices

