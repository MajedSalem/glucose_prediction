import numpy as np
import pandas as pd

from glucose_prediction.evaluation.regimes import (
    hypoglycemia_mask, overnight_mask, postprandial_mask,
)


def _grid(start, n, freq="5min"):
    return pd.date_range(start, periods=n, freq=freq)


def test_postprandial_mask_boundary_is_inclusive_at_exactly_lookback_minutes():
    idx = _grid("2024-01-01 00:00", 9)
    carbs = pd.Series(0.0, index=idx)
    carbs.iloc[4] = 30.0  # meal at 00:20

    target_time = np.array([
        idx[3],  # 00:15 -- 5 min BEFORE the meal -> not postprandial
        idx[4],  # 00:20 -- exactly at the meal -> postprandial
        idx[5],  # 00:25 -- 5 min after -> postprandial
        idx[7],  # 00:35 -- 15 min after (== lookback boundary) -> postprandial (inclusive)
        idx[8],  # 00:40 -- 20 min after (just past lookback) -> not postprandial
    ], dtype="datetime64[ns]")

    mask = postprandial_mask(target_time, carbs, lookback_minutes=15)
    assert mask.tolist() == [False, True, True, True, False]


def test_postprandial_mask_no_meals_is_all_false():
    idx = _grid("2024-01-01", 10)
    carbs = pd.Series(0.0, index=idx)
    mask = postprandial_mask(idx.to_numpy(), carbs, lookback_minutes=60)
    assert not mask.any()


def test_postprandial_mask_handles_target_with_no_earlier_history():
    idx = _grid("2024-01-01", 5)
    carbs = pd.Series([50.0, 0.0, 0.0, 0.0, 0.0], index=idx)  # meal AT the first point
    mask = postprandial_mask(idx.to_numpy(), carbs, lookback_minutes=15)
    assert mask.tolist() == [True, True, True, True, False]


def test_overnight_mask_default_hours_are_half_open():
    target_time = pd.to_datetime([
        "2024-01-01 23:59", "2024-01-02 00:00", "2024-01-02 03:00",
        "2024-01-02 05:59", "2024-01-02 06:00", "2024-01-02 12:00",
    ]).to_numpy()
    mask = overnight_mask(target_time)
    assert mask.tolist() == [False, True, True, True, False, False]


def test_hypoglycemia_mask_boundary_matches_classify_range_convention():
    y_true = np.array([69.9, 70.0, 70.1, 40.0])
    assert hypoglycemia_mask(y_true).tolist() == [True, False, False, True]
