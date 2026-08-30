import numpy as np
import pandas as pd
import pytest

from glucose_prediction.config import DATA_PROCESSED_DIR
from glucose_prediction.evaluation.harness import evaluate

CSV_PATH = DATA_PROCESSED_DIR / "ohio_t1dm" / "559_train.csv"


@pytest.mark.skipif(not CSV_PATH.exists(), reason="requires local OhioT1DM processed data (gitignored, DUA-restricted)")
def test_harness_smoke_on_real_glucose_distribution():
    df = pd.read_csv(CSV_PATH)
    df = df[~df["glucose_missing"]]
    y_true = df["glucose"].to_numpy()

    # Additive Gaussian noise, NOT a lag/shift -- a shift would effectively be
    # the persistence baseline, which is Day 4-5's job, not this smoke test's.
    rng = np.random.default_rng(0)
    y_pred = y_true + rng.normal(0, 15, size=len(y_true))

    result = evaluate(y_true, y_pred, horizon_minutes=30, model_name="smoke_test_noisy")

    assert not any(pd.isna(v) for v in result.values() if isinstance(v, float))
    assert 10 < result["rmse"] < 20
    assert 0 <= result["tir_true"] <= 100
    assert 0 <= result["tir_pred"] <= 100
    assert 0 <= result["tir_agreement_rate"] <= 1
    zone_pcts = [result[f"zone_{z}_pct"] for z in "ABCDE"]
    assert sum(zone_pcts) == pytest.approx(100.0, abs=0.01)
