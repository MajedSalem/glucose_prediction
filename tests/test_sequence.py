import numpy as np
import pytest
import torch

from glucose_prediction.config import DATA_PROCESSED_DIR
from glucose_prediction.models.sequence import (
    LSTMForecaster, chronological_split, compute_norm_stats,
    denormalize, normalize, train_lstm,
)

CSV_PATH = DATA_PROCESSED_DIR / "ohio_t1dm" / "559_train.csv"


def test_forward_pass_shape():
    model = LSTMForecaster(hidden_size=8, num_layers=1)
    x = torch.randn(5, 24, 1)
    assert model(x).shape == (5, 2)


def test_normalize_denormalize_roundtrip():
    x = np.array([[80.0, 120.0, 200.0]])
    mu, sigma = 150.0, 40.0
    assert np.allclose(denormalize(normalize(x, mu, sigma), mu, sigma), x)


def test_chronological_split_preserves_order_and_sizes():
    X = np.arange(20).reshape(20, 1).astype(float)
    y = {30: np.arange(20, dtype=float)}
    X_tr, y_tr, X_val, y_val = chronological_split(X, y, val_frac=0.2)
    assert len(X_tr) == 16 and len(X_val) == 4
    assert X_tr[:, 0].max() < X_val[:, 0].min()


def test_train_lstm_reduces_loss_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 200
    X = np.cumsum(rng.normal(0, 2, size=(n, 24)), axis=1) + 120
    y = {30: X[:, -1] + rng.normal(0, 2, n), 60: X[:, -1] + rng.normal(0, 3, n)}
    X_tr, y_tr, X_val, y_val = chronological_split(X, y, val_frac=0.2)
    mu, sigma = compute_norm_stats(X_tr)
    model, history = train_lstm(X_tr, y_tr, X_val, y_val, mu, sigma,
                                 hidden_size=8, max_epochs=10, patience=3, seed=0)
    assert history[-1] < history[0]


@pytest.mark.skipif(not CSV_PATH.exists(), reason="requires local OhioT1DM processed data (gitignored, DUA-restricted)")
def test_run_sequence_model_real_data_smoke():
    from scripts.run_sequence_model import run_sequence_model

    per_patient, model, history = run_sequence_model(
        patient_ids=["559", "563"], seed=0, max_epochs=5, patience=2)
    assert not per_patient[["rmse", "mae"]].isna().any().any()
    assert per_patient["rmse"].between(5, 100).all()
