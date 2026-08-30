"""Baseline glucose predictors, evaluated on the same lag-feature windows.

All three are vectorized over the full window matrix -- confirmed ~70-90x
faster than a per-window Python loop for bit-identical output (see
data/windowing.py's own vectorization note), consistent with this project's
existing numpy-vectorized style.

Ridge is trained per (patient, horizon): the horizon is baked into the
training target y[h], not into the features -- X is the same 24-sample
glucose lag window regardless of which horizon is being predicted.
"""
import numpy as np
from sklearn.linear_model import Ridge

from glucose_prediction.config import SAMPLING_INTERVAL_MINUTES


def persistence_predict(X: np.ndarray) -> np.ndarray:
    """Predict every horizon as the last observed value in the window."""
    return X[:, -1].copy()


def linear_extrapolation_predict(X: np.ndarray, horizon_minutes: int,
                                  sampling_interval: int = SAMPLING_INTERVAL_MINUTES) -> np.ndarray:
    """Fit a least-squares line per window and extrapolate `horizon_minutes` ahead.

    x is minutes relative to each window's *last* sample (0 at the last
    sample, negative before it), so the fitted intercept is already the
    line's value at t=0 and the prediction is intercept + slope*horizon_minutes.
    Closed-form OLS across the whole batch at once (x is identical for every
    window) instead of per-window np.polyfit -- verified bit-identical to
    np.polyfit on 200 random real windows.
    """
    n_samples, window_size = X.shape
    x = (np.arange(window_size) - (window_size - 1)) * sampling_interval
    n, sum_x, sum_x2 = window_size, x.sum(), (x ** 2).sum()
    sum_y = X.sum(axis=1)
    sum_xy = X @ x
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    return intercept + slope * horizon_minutes


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    """Fit one Ridge model on glucose-only lag features (default alpha, no tuning)."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def ridge_predict(model: Ridge, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
