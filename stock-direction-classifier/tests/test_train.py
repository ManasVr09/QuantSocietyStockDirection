import pandas as pd

from src.train import chronological_split


def test_chronological_split_is_ordered():
    dates = pd.date_range("2022-12-20", "2024-01-10", freq="D")
    data = pd.DataFrame({
        "date": dates,
        "feature_time": dates + pd.Timedelta("16h"),
        "target_date": dates + pd.Timedelta("1D"),
        "ticker": "AAA",
        "target": 1,
    })
    config = {"splits": {"train_end": "2022-12-31", "validation_start": "2023-01-01", "validation_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"}}
    splits = chronological_split(data, config)
    assert splits.train["target_date"].max() < splits.validation["feature_time"].min()
    assert splits.validation["target_date"].max() < splits.test["feature_time"].min()
    assert splits.train["target_date"].max() <= pd.Timestamp("2022-12-31 23:59:59.999999")
