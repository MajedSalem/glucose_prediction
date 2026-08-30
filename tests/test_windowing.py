import numpy as np
import pandas as pd
import pytest

from glucose_prediction.config import DATA_PROCESSED_DIR
from glucose_prediction.data.windowing import make_windows

CSV_PATH = DATA_PROCESSED_DIR / "ohio_t1dm" / "559_train.csv"


def test_make_windows_excludes_input_and_target_missing():
    glucose = [100, 101, 102, 103, 104, 105, 106, 107, 108, 110]
    missing = [False, False, False, False, True, False, False, False, False, False]
    df = pd.DataFrame({"glucose": glucose, "glucose_missing": missing})

    X, y, t = make_windows(df, window_size=3, horizon_steps={5: 1})

    assert X.tolist() == [[100, 101, 102], [105, 106, 107], [106, 107, 108]]
    assert y[5].tolist() == [103, 108, 110]
    assert t.tolist() == [2, 7, 8]


def test_make_windows_all_valid_when_no_missing():
    n = 20
    df = pd.DataFrame({"glucose": np.arange(n, dtype=float), "glucose_missing": [False] * n})
    X, y, t = make_windows(df, window_size=3, horizon_steps={5: 1, 10: 2})
    expected_n = n - 3 - 2 + 1
    assert X.shape == (expected_n, 3)
    assert y[5].shape == y[10].shape == (expected_n,)


def test_make_windows_too_short_returns_empty():
    df = pd.DataFrame({"glucose": [1.0, 2.0, 3.0], "glucose_missing": [False, False, False]})
    X, y, t = make_windows(df, window_size=3, horizon_steps={5: 1})
    assert X.shape == (0, 3)
    assert y[5].shape == (0,)
    assert t.shape == (0,)


def test_make_windows_shares_one_validity_mask_across_horizons():
    # Only the 2-step horizon's target is missing on the last window; that
    # window must be dropped from *every* horizon, not just the 2-step one.
    glucose = list(range(100, 110))
    missing = [False] * 10
    missing[9] = True
    df = pd.DataFrame({"glucose": glucose, "glucose_missing": missing})
    X, y, t = make_windows(df, window_size=3, horizon_steps={5: 1, 10: 2})
    assert X.shape == y[5].shape + (3,) == (5, 3)
    assert y[10].shape == (5,)
    assert 107 not in X[:, -1]  # the dropped window's last input value


@pytest.mark.skipif(not CSV_PATH.exists(), reason="requires local OhioT1DM processed data (gitignored, DUA-restricted)")
def test_make_windows_real_patient_valid_count():
    df = pd.read_csv(CSV_PATH, index_col="timestamp", parse_dates=True)
    X, y, t = make_windows(df)
    assert X.shape == (9342, 24)
    assert y[30].shape == y[60].shape == (9342,)
    assert not np.isnan(X).any() and not np.isnan(y[30]).any()
