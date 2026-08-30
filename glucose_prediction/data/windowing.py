"""Turn a per-patient CGM DataFrame into (input window -> horizon target) examples.

Windows are built purely positionally over the 5-minute grid (every processed
CSV/DataFrame in this project is a strict, gap-free DatetimeIndex -- real time
gaps show up as glucose_missing=True rows, never as missing grid rows), so a
plain fixed-size sliding window over row position is exactly time-aligned. A
window (or a horizon target) touching any glucose_missing=True row is dropped
outright, never imputed -- consistent with evaluation/metrics.py's own
"caller filters NaN" convention.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from glucose_prediction.config import (
    HORIZONS_MINUTES, INPUT_WINDOW_MINUTES, SAMPLING_INTERVAL_MINUTES,
)

WINDOW_SIZE = INPUT_WINDOW_MINUTES // SAMPLING_INTERVAL_MINUTES        # 24 samples
HORIZON_STEPS = {h: h // SAMPLING_INTERVAL_MINUTES for h in HORIZONS_MINUTES}  # {30: 6, 60: 12}


def make_windows(df, window_size=WINDOW_SIZE, horizon_steps=HORIZON_STEPS):
    """Slide a fixed-size window of `glucose` over `df`.

    Returns (X, y, t):
      X -- float64 ndarray (n_valid, window_size); glucose lags, oldest->newest
           (X[:, -1] is the most recent observed value).
      y -- dict[int, ndarray]; horizon_minutes -> target glucose (n_valid,).
      t -- ndarray (df.index.dtype); timestamp of each window's last input
           sample, kept for future per-window error analysis (Day 9) -- not
           used by the baselines themselves.

    One shared validity mask is applied across every horizon (a window is
    dropped from X and *every* y[h] if it or *any* horizon's target touches
    glucose_missing=True), so X and every y[h] stay row-aligned with each
    other -- there is no case where y[30] and y[60] have different lengths.
    """
    glucose = df["glucose"].to_numpy(dtype=float)
    missing = df["glucose_missing"].to_numpy(dtype=bool)
    timestamps = df.index.to_numpy()

    max_step = max(horizon_steps.values())
    n_windows = len(df) - window_size - max_step + 1
    if n_windows <= 0:
        empty_t = np.empty(0, dtype=timestamps.dtype)
        return np.empty((0, window_size)), {h: np.empty(0) for h in horizon_steps}, empty_t

    X_all = sliding_window_view(glucose, window_size)[:n_windows]
    missing_all = sliding_window_view(missing, window_size)[:n_windows]
    last_idx = np.arange(n_windows) + window_size - 1

    valid = ~missing_all.any(axis=1)
    target_glucose = {}
    for h, step in horizon_steps.items():
        idx = last_idx + step
        target_glucose[h] = glucose[idx]
        valid &= ~missing[idx]

    X = X_all[valid]
    y = {h: vals[valid] for h, vals in target_glucose.items()}
    t = timestamps[last_idx[valid]]
    return X, y, t
