import numpy as np
import pandas as pd
import pytest

from glucose_prediction.evaluation.metrics import rmse, mae, classify_range, time_in_range, range_agreement


def test_rmse_mae_known_values():
    y_true, y_pred = [0, 0], [3, 4]
    assert rmse(y_true, y_pred) == pytest.approx(3.5355339059)
    assert mae(y_true, y_pred) == pytest.approx(3.5)


def test_rmse_zero_for_identical_input():
    assert rmse([100, 200, 150], [100, 200, 150]) == 0.0


@pytest.mark.parametrize("wrapper", [np.asarray, pd.Series])
def test_metrics_accept_ndarray_and_series(wrapper):
    y_true, y_pred = wrapper([100.0, 200.0]), wrapper([110.0, 190.0])
    assert rmse(y_true, y_pred) == pytest.approx(10.0)
    assert mae(y_true, y_pred) == pytest.approx(10.0)


def test_classify_range_boundaries_are_inclusive_target():
    result = classify_range([69, 70, 150, 180, 181])
    assert list(result) == ["hypo", "target", "target", "target", "hyper"]


def test_time_in_range_percentages():
    result = time_in_range([50, 50, 100, 100, 200])
    assert result["hypo"] == pytest.approx(40.0)
    assert result["target"] == pytest.approx(40.0)
    assert result["hyper"] == pytest.approx(20.0)


def test_range_agreement_perfect():
    rate, confusion = range_agreement([50, 100, 200], [50, 100, 200])
    assert rate == 1.0
    assert confusion.loc["hypo", "hypo"] == 1
    assert confusion.loc["target", "target"] == 1
    assert confusion.loc["hyper", "hyper"] == 1


def test_range_agreement_confusion_matrix_shape_is_always_3x3():
    rate, confusion = range_agreement([50, 100], [100, 50])
    assert confusion.shape == (3, 3)
    assert confusion.loc["hyper"].sum() == 0
    assert rate == 0.0
