"""Boolean regime masks for Day 9 error analysis, anchored at the TARGET
time (t + horizon_minutes) of each windowing.make_windows example, not the
input window's last-input time -- a meal or a night hour is defined by when
the *prediction* lands, not when the model was standing.

Pure boolean-array functions, composing with harness.evaluate the same way
baselines/models already do: no metric computation happens here, only
slicing. postprandial_mask needs the patient's FULL carbs_g column (not
just what's inside a 24-lag input window X) because both the lookback and
the target time fall after the window's last input sample.
"""
import numpy as np
import pandas as pd

from glucose_prediction.config import HYPOGLYCEMIA_THRESHOLD, SAMPLING_INTERVAL_MINUTES

# See progress.md Day 9 for the real inter-meal-gap-distribution justification.
POSTPRANDIAL_LOOKBACK_MINUTES = 120
OVERNIGHT_HOURS = (0, 6)  # [start, end) local hour-of-day, half-open


def postprandial_mask(target_time, carbs: pd.Series,
                       lookback_minutes: int = POSTPRANDIAL_LOOKBACK_MINUTES) -> np.ndarray:
    """True where a carbs_g>0 grid point exists in the closed interval
    [target_time - lookback_minutes, target_time] on `carbs`'s own grid.

    `carbs` must be that patient's full test-split carbs_g Series, indexed
    by the same strict, gap-free 5-minute DatetimeIndex windowing.py
    assumes; `target_time` values must already lie on that grid (guaranteed
    by construction: t from make_windows plus an integer-multiple-of-5min
    horizon). Uses a backward rolling window (pandas' own min_periods=1
    handles targets with less than lookback_minutes of history -- no
    special-casing needed).
    """
    steps = lookback_minutes // SAMPLING_INTERVAL_MINUTES
    had_carb = carbs > 0
    recent = had_carb.rolling(window=steps + 1, min_periods=1).sum() > 0
    return recent.reindex(pd.DatetimeIndex(target_time)).to_numpy(dtype=bool)


def overnight_mask(target_time, start_hour: int = OVERNIGHT_HOURS[0],
                    end_hour: int = OVERNIGHT_HOURS[1]) -> np.ndarray:
    """True where target_time's local hour-of-day falls in [start_hour, end_hour)."""
    hours = pd.DatetimeIndex(target_time).hour
    return (hours >= start_hour) & (hours < end_hour)


def hypoglycemia_mask(y_true, threshold: float = HYPOGLYCEMIA_THRESHOLD) -> np.ndarray:
    """True where true glucose at the target time is below the hypoglycemia threshold."""
    return np.asarray(y_true, dtype=float) < threshold
