"""Day 9: slice the 4 already-built models' test-set predictions by clinical
regime (postprandial, overnight, hypoglycemia) and report where each
wins/loses -- especially "does it miss lows?" (plan.md Day 9). Pure
aggregation over harness.evaluate(); no new metrics, no new models.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from glucose_prediction.config import DATA_PROCESSED_DIR, HORIZONS_MINUTES, RESULTS_DIR
from glucose_prediction.data.ohio_t1dm import PATIENT_IDS
from glucose_prediction.data.windowing import make_windows
from glucose_prediction.evaluation.harness import evaluate
from glucose_prediction.evaluation.regimes import (
    hypoglycemia_mask, overnight_mask, postprandial_mask,
)
from glucose_prediction.models.baselines import (
    fit_ridge, linear_extrapolation_predict, persistence_predict, ridge_predict,
)
from glucose_prediction.models.sequence import compute_norm_stats, predict, train_lstm
from scripts.run_sequence_model import build_pooled_data

MODEL_NAMES = ("persistence", "linear_extrapolation", "ridge", "lstm_pooled")
REGIMES = ("overall", "postprandial", "overnight", "hypoglycemia")


def _load(patient_id, split, processed_dir):
    path = processed_dir / f"{patient_id}_{split}.csv"
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)


def collect_predictions(patient_ids, processed_dir, seed=0):
    """Train ridge (per patient) and the pooled LSTM once; predict all 4
    models on every patient's test windows; tag each window with its regime
    flags at the target time.

    Returns dict[(patient_id, horizon_minutes)] -> {
        "y_true": ndarray, "masks": {"postprandial":.., "overnight":.., "hypoglycemia":..},
        "preds": {model_name: ndarray for model_name in MODEL_NAMES},
    }
    """
    X_train_pooled, y_train_pooled, X_val_pooled, y_val_pooled, test_windows = \
        build_pooled_data(patient_ids, processed_dir)
    mu, sigma = compute_norm_stats(X_train_pooled)
    lstm_model, _ = train_lstm(X_train_pooled, y_train_pooled, X_val_pooled, y_val_pooled,
                                mu, sigma, seed=seed)

    out = {}
    for patient_id in patient_ids:
        X_test, y_test, t_test = test_windows[patient_id]
        X_train, y_train, _ = make_windows(_load(patient_id, "train", processed_dir))
        test_carbs = _load(patient_id, "test", processed_dir)["carbs_g"]
        lstm_preds = predict(lstm_model, X_test, mu, sigma)

        for h in HORIZONS_MINUTES:
            ridge_model = fit_ridge(X_train, y_train[h])
            target_time = t_test + np.timedelta64(h, "m")
            preds = {
                "persistence": persistence_predict(X_test),
                "linear_extrapolation": linear_extrapolation_predict(X_test, h),
                "ridge": ridge_predict(ridge_model, X_test),
                "lstm_pooled": lstm_preds[h],
            }
            out[(patient_id, h)] = {
                "y_true": y_test[h],
                "masks": {
                    "postprandial": postprandial_mask(target_time, test_carbs),
                    "overnight": overnight_mask(target_time),
                    "hypoglycemia": hypoglycemia_mask(y_test[h]),
                },
                "preds": preds,
            }
    return out


def pooled_regime_table(predictions, patient_ids) -> pd.DataFrame:
    """Pool (y_true, y_pred) across all patients for each (model, horizon,
    regime) combo and call the existing harness.evaluate -- no new metrics.
    """
    rows = []
    for h in HORIZONS_MINUTES:
        for regime in REGIMES:
            for model in MODEL_NAMES:
                yt_parts, yp_parts = [], []
                for patient_id in patient_ids:
                    d = predictions[(patient_id, h)]
                    mask = (np.ones_like(d["y_true"], dtype=bool) if regime == "overall"
                            else d["masks"][regime])
                    yt_parts.append(d["y_true"][mask])
                    yp_parts.append(d["preds"][model][mask])
                y_true, y_pred = np.concatenate(yt_parts), np.concatenate(yp_parts)
                result = evaluate(y_true, y_pred, horizon_minutes=h, model_name=model)
                result["regime"], result["n"] = regime, len(y_true)
                rows.append(result)
    return pd.DataFrame(rows)


def hypo_detection_table(by_regime: pd.DataFrame) -> pd.DataFrame:
    """Relabel the regime='hypoglycemia' rows for an explicit "does it miss
    lows" readout -- zone_A/D/E _pct on that subset ARE the hit/miss-as-
    normal/miss-as-high rates (B and C are unreachable when y_true<70).
    """
    hypo = by_regime[by_regime["regime"] == "hypoglycemia"].rename(columns={
        "n": "n_hypo_windows", "zone_A_pct": "hit_rate_pct",
        "zone_D_pct": "missed_as_normal_pct", "zone_E_pct": "missed_as_high_pct",
        "rmse": "rmse_mgdl",
    })
    return hypo[["model", "horizon_minutes", "n_hypo_windows", "hit_rate_pct",
                 "missed_as_normal_pct", "missed_as_high_pct", "rmse_mgdl"]]


def lstm_vs_ridge_per_patient(predictions, patient_ids) -> pd.DataFrame:
    """Per-patient (not pooled) ridge-vs-lstm_pooled RMSE, overall (no regime
    filter) -- pinpoints exactly which patients the LSTM loses to ridge on.
    """
    rows = []
    for patient_id in patient_ids:
        for h in HORIZONS_MINUTES:
            d = predictions[(patient_id, h)]
            rmse_ridge = evaluate(d["y_true"], d["preds"]["ridge"], h, "ridge")["rmse"]
            rmse_lstm = evaluate(d["y_true"], d["preds"]["lstm_pooled"], h, "lstm_pooled")["rmse"]
            rows.append({
                "patient_id": patient_id, "horizon_minutes": h,
                "rmse_ridge": rmse_ridge, "rmse_lstm_pooled": rmse_lstm,
                "rmse_diff_ridge_minus_lstm": rmse_ridge - rmse_lstm,
                "lstm_wins": rmse_lstm < rmse_ridge,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="+", default=list(PATIENT_IDS))
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR / "ohio_t1dm")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    predictions = collect_predictions(args.patients, args.processed_dir, seed=args.seed)
    by_regime = pooled_regime_table(predictions, args.patients)
    hypo = hypo_detection_table(by_regime)
    lstm_vs_ridge = lstm_vs_ridge_per_patient(predictions, args.patients)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_regime.to_csv(args.output_dir / "error_analysis_by_regime.csv", index=False)
    hypo.to_csv(args.output_dir / "error_analysis_hypo_detection.csv", index=False)
    lstm_vs_ridge.to_csv(args.output_dir / "error_analysis_lstm_vs_ridge_per_patient.csv", index=False)

    print(by_regime.to_string())
    print("\n--- does it miss lows? ---")
    print(hypo.to_string())
    print("\n--- LSTM loses to ridge at ---")
    print(lstm_vs_ridge[~lstm_vs_ridge["lstm_wins"]].to_string())


if __name__ == "__main__":
    main()
