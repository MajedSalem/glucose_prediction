"""Train a pooled LSTM across all patients and evaluate it per-patient on
the harness, following the same shape as scripts/run_baselines.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from glucose_prediction.config import DATA_PROCESSED_DIR, HORIZONS_MINUTES, RESULTS_DIR
from glucose_prediction.data.ohio_t1dm import PATIENT_IDS
from glucose_prediction.data.windowing import make_windows
from glucose_prediction.evaluation.harness import evaluate
from glucose_prediction.models.sequence import (
    chronological_split, compute_norm_stats, predict, train_lstm,
)


def _load(patient_id: str, split: str, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / f"{patient_id}_{split}.csv"
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)


def build_pooled_data(patient_ids, processed_dir, val_frac=0.15):
    X_train_parts, X_val_parts = [], []
    y_train_parts = {h: [] for h in HORIZONS_MINUTES}
    y_val_parts = {h: [] for h in HORIZONS_MINUTES}
    test_windows = {}

    for patient_id in patient_ids:
        X, y, _ = make_windows(_load(patient_id, "train", processed_dir))
        X_tr, y_tr, X_val, y_val = chronological_split(X, y, val_frac)
        X_train_parts.append(X_tr)
        X_val_parts.append(X_val)
        for h in HORIZONS_MINUTES:
            y_train_parts[h].append(y_tr[h])
            y_val_parts[h].append(y_val[h])
        test_windows[patient_id] = make_windows(_load(patient_id, "test", processed_dir))

    X_train = np.concatenate(X_train_parts)
    X_val = np.concatenate(X_val_parts)
    y_train = {h: np.concatenate(parts) for h, parts in y_train_parts.items()}
    y_val = {h: np.concatenate(parts) for h, parts in y_val_parts.items()}
    return X_train, y_train, X_val, y_val, test_windows


def run_sequence_model(patient_ids=PATIENT_IDS, processed_dir=DATA_PROCESSED_DIR / "ohio_t1dm",
                        seed=0, **train_kwargs):
    X_train, y_train, X_val, y_val, test_windows = build_pooled_data(patient_ids, processed_dir)
    mu, sigma = compute_norm_stats(X_train)
    model, history = train_lstm(X_train, y_train, X_val, y_val, mu, sigma, seed=seed, **train_kwargs)

    rows = []
    for patient_id, (X_test, y_test, _) in test_windows.items():
        predictions = predict(model, X_test, mu, sigma)
        for h in HORIZONS_MINUTES:
            result = evaluate(y_test[h], predictions[h], horizon_minutes=h, model_name="lstm_pooled")
            result["patient_id"] = patient_id
            rows.append(result)
    return pd.DataFrame(rows), model, history


def summarize(per_patient: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_patient.columns if c not in ("model", "horizon_minutes", "patient_id")]
    return per_patient.groupby(["model", "horizon_minutes"])[metric_cols].agg(["mean", "std"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="+", default=list(PATIENT_IDS))
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR / "ohio_t1dm")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_patient, model, history = run_sequence_model(args.patients, args.processed_dir, seed=args.seed)
    per_patient.to_csv(args.output_dir / "sequence_model_per_patient.csv", index=False)

    summary = summarize(per_patient)
    summary.to_csv(args.output_dir / "sequence_model_summary.csv")
    print(summary[["rmse", "mae"]])
    print(f"Training stopped after {len(history)} epochs, best val RMSE {min(history):.3f}")


if __name__ == "__main__":
    main()
