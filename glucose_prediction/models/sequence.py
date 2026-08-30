"""Pooled LSTM sequence model predicting +30/+60 min glucose from a 2-hour
CGM history window, trained across all patients and evaluated per-patient.

Glucose-lag-only input, matching the baselines exactly (carbs/insulin
deferred). Architecture size was benchmarked (hidden_size 16/32/64,
num_layers 1) before picking hidden_size=32: differences were within noise
(~0.1 RMSE) across a 13x parameter range -- the ceiling here is set by the
glucose-only-lag-feature task itself, not by model capacity.
"""
import numpy as np
import torch
import torch.nn as nn

from glucose_prediction.config import HORIZONS_MINUTES

HORIZONS = list(HORIZONS_MINUTES)


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, len(HORIZONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size=24, 1) glucose lags, oldest->newest. Returns (batch, len(HORIZONS))."""
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def compute_norm_stats(X_train: np.ndarray):
    return float(X_train.mean()), float(X_train.std())


def normalize(values, mu, sigma):
    return (values - mu) / sigma


def denormalize(values, mu, sigma):
    return values * sigma + mu


def chronological_split(X: np.ndarray, y: dict, val_frac: float = 0.15):
    """Hold out the last val_frac (by time order -- windows are already
    chronological) of each patient's windows for validation, before pooling.
    """
    n = len(X)
    n_train = n - int(round(n * val_frac))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train = {h: v[:n_train] for h, v in y.items()}
    y_val = {h: v[n_train:] for h, v in y.items()}
    return X_train, y_train, X_val, y_val


def _stack_targets(y: dict) -> np.ndarray:
    return np.stack([y[h] for h in HORIZONS], axis=1)


def train_lstm(X_train, y_train, X_val, y_val, mu, sigma, *,
               hidden_size=32, num_layers=1, max_epochs=50, patience=5,
               batch_size=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)

    X_train_t = torch.tensor(normalize(X_train, mu, sigma), dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(normalize(_stack_targets(y_train), mu, sigma), dtype=torch.float32)
    X_val_t = torch.tensor(normalize(X_val, mu, sigma), dtype=torch.float32).unsqueeze(-1)
    y_val_true = _stack_targets(y_val)  # mg/dL, for early-stopping RMSE

    model = LSTMForecaster(hidden_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    n_train = len(X_train_t)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(X_train_t[idx])
            loss = loss_fn(pred, y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_norm = model(X_val_t).numpy()
        val_pred = denormalize(val_pred_norm, mu, sigma)
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val_true) ** 2)))
        history.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def predict(model, X, mu, sigma) -> dict:
    model.eval()
    X_t = torch.tensor(normalize(X, mu, sigma), dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        pred_norm = model(X_t).numpy()
    pred = denormalize(pred_norm, mu, sigma)
    return {h: pred[:, i] for i, h in enumerate(HORIZONS)}
