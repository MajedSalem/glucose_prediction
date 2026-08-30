"""Fit and evaluate the three Day 4-5 baselines (persistence, linear
extrapolation, per-patient ridge) on every OhioT1DM patient, on the harness.

Ridge is trained per (patient, horizon) on that patient's train-split windows
and evaluated on that patient's test-split windows -- 12 patients x 2 horizons
= 24 fitted models -- so every model only ever sees its own patient's history,
unlike the pooled deep model planned for Days 6-8.
"""
import argparse
from pathlib import Path

import pandas as pd

from glucose_prediction.config import DATA_PROCESSED_DIR, HORIZONS_MINUTES, RESULTS_DIR
from glucose_prediction.data.ohio_t1dm import PATIENT_IDS
from glucose_prediction.data.windowing import make_windows
from glucose_prediction.evaluation.harness import evaluate
from glucose_prediction.models.baselines import (
    fit_ridge, linear_extrapolation_predict, persistence_predict, ridge_predict,
)


def _load(patient_id: str, split: str, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / f"{patient_id}_{split}.csv"
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)


def run_baselines(patient_ids=PATIENT_IDS, processed_dir=DATA_PROCESSED_DIR / "ohio_t1dm") -> pd.DataFrame:
    rows = []
    for patient_id in patient_ids:
        X_train, y_train, _ = make_windows(_load(patient_id, "train", processed_dir))
        X_test, y_test, _ = make_windows(_load(patient_id, "test", processed_dir))
        for h in HORIZONS_MINUTES:
            ridge_model = fit_ridge(X_train, y_train[h])
            predictions = {
                "persistence": persistence_predict(X_test),
                "linear_extrapolation": linear_extrapolation_predict(X_test, h),
                "ridge": ridge_predict(ridge_model, X_test),
            }
            for model_name, y_pred in predictions.items():
                result = evaluate(y_test[h], y_pred, horizon_minutes=h, model_name=model_name)
                result["patient_id"] = patient_id
                rows.append(result)
    return pd.DataFrame(rows)


def summarize(per_patient: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_patient.columns if c not in ("model", "horizon_minutes", "patient_id")]
    return per_patient.groupby(["model", "horizon_minutes"])[metric_cols].agg(["mean", "std"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="+", default=list(PATIENT_IDS))
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR / "ohio_t1dm")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_patient = run_baselines(args.patients, args.processed_dir)
    per_patient.to_csv(args.output_dir / "baselines_per_patient.csv", index=False)

    summary = summarize(per_patient)
    summary.to_csv(args.output_dir / "baselines_summary.csv")
    print(summary[["rmse", "mae"]])


if __name__ == "__main__":
    main()
