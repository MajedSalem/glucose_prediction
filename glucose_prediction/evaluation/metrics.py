"""Point-wise error, time-in-range, and range-classification-agreement metrics.

Functions take plain numpy arrays or pandas Series so baseline and model
code (none exists yet) can call them without any project-specific object
model. Arrays are compared positionally, not index-aligned. Callers are
responsible for filtering missing/NaN glucose before calling -- these
functions do not clean data, they measure already-aligned (y_true, y_pred)
pairs, and will loudly return NaN if fed NaN, by design.

Convention: "_pct" fields are 0-100 (clinical TIR reporting convention),
"_rate" fields are 0-1.
"""
import numpy as np
import pandas as pd

from glucose_prediction.config import HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD

RANGE_CATEGORIES = ("hypo", "target", "hyper")


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def classify_range(glucose, hypo_threshold=HYPOGLYCEMIA_THRESHOLD,
                    hyper_threshold=HYPERGLYCEMIA_THRESHOLD) -> np.ndarray:
    """Classify readings as 'hypo'/'target'/'hyper'; the 70-180 target band is inclusive."""
    glucose = np.asarray(glucose, dtype=float)
    return np.where(glucose < hypo_threshold, "hypo",
                     np.where(glucose > hyper_threshold, "hyper", "target"))


def time_in_range(glucose, hypo_threshold=HYPOGLYCEMIA_THRESHOLD,
                   hyper_threshold=HYPERGLYCEMIA_THRESHOLD) -> dict:
    """Percent of readings in each range: {'hypo': .., 'target': .., 'hyper': ..}."""
    categories = classify_range(glucose, hypo_threshold, hyper_threshold)
    return {cat: 100.0 * float(np.mean(categories == cat)) for cat in RANGE_CATEGORIES}


def range_agreement(y_true, y_pred, hypo_threshold=HYPOGLYCEMIA_THRESHOLD,
                     hyper_threshold=HYPERGLYCEMIA_THRESHOLD):
    """Agreement between true/predicted range classification.

    Returns (agreement_rate, confusion): agreement_rate is 0-1; confusion is
    always a 3x3 pd.DataFrame (rows=true, cols=pred) in RANGE_CATEGORIES
    order, zero-filled for categories absent from this sample -- a fixed
    shape matters for comparing confusion matrices across calls later
    (Day 9 error analysis).
    """
    true_cat = classify_range(y_true, hypo_threshold, hyper_threshold)
    pred_cat = classify_range(y_pred, hypo_threshold, hyper_threshold)
    agreement_rate = float(np.mean(true_cat == pred_cat))
    confusion = pd.crosstab(pd.Series(true_cat, name="true"), pd.Series(pred_cat, name="pred"))
    confusion = confusion.reindex(index=RANGE_CATEGORIES, columns=RANGE_CATEGORIES, fill_value=0)
    return agreement_rate, confusion
