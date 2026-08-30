"""Combine point-wise, time-in-range, and Clarke-grid metrics into one flat
row per (model, horizon) evaluation, ready to accumulate into a results
DataFrame with pd.DataFrame([evaluate(...), evaluate(...), ...]).
"""
from glucose_prediction.evaluation.clarke import clarke_zones, zone_percentages
from glucose_prediction.evaluation.metrics import rmse, mae, time_in_range, range_agreement


def evaluate(y_true, y_pred, horizon_minutes: int, model_name: str) -> dict:
    agreement_rate, _ = range_agreement(y_true, y_pred)
    zones = clarke_zones(y_true, y_pred)
    return {
        "model": model_name,
        "horizon_minutes": horizon_minutes,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "tir_true": time_in_range(y_true)["target"],
        "tir_pred": time_in_range(y_pred)["target"],
        "tir_agreement_rate": agreement_rate,
        **zone_percentages(zones),
    }
