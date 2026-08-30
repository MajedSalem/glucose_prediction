import numpy as np
import pytest

from glucose_prediction.models.baselines import (
    fit_ridge, linear_extrapolation_predict, persistence_predict, ridge_predict,
)


def test_persistence_predict_returns_last_column():
    X = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    assert persistence_predict(X).tolist() == [30.0, 3.0]


def test_linear_extrapolation_on_perfect_line():
    # slope 0.4 mg/dL/min over a 4-sample, 5-min-spaced window
    X = np.array([[100.0, 102.0, 104.0, 106.0]])
    assert linear_extrapolation_predict(X, 30) == pytest.approx([118.0])
    assert linear_extrapolation_predict(X, 60) == pytest.approx([130.0])


def test_linear_extrapolation_matches_hand_computed_ols():
    X = np.array([[1.0, 2.0, 4.0]])
    assert linear_extrapolation_predict(X, 10) == pytest.approx([6.8333333], rel=1e-6)


def test_ridge_beats_constant_baseline_on_synthetic_data():
    rng = np.random.default_rng(0)
    X_train = np.cumsum(rng.normal(0, 3, size=(200, 24)), axis=1) + 120
    y_train = X_train[:, -1] + rng.normal(0, 5, size=200)
    X_test = np.cumsum(rng.normal(0, 3, size=(50, 24)), axis=1) + 120
    y_test = X_test[:, -1] + rng.normal(0, 5, size=50)

    model = fit_ridge(X_train, y_train)
    pred = ridge_predict(model, X_test)
    const_pred = np.full(50, y_train.mean())

    ridge_rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    const_rmse = np.sqrt(np.mean((y_test - const_pred) ** 2))
    assert ridge_rmse < const_rmse
