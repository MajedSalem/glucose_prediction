"""Parse OhioT1DM XML into a clean 5-minute-grid DataFrame per patient.

Only glucose_level, meal, basal, temp_basal, and bolus are parsed -- the
streams this project's task definition uses (CGM, optionally insulin and
carbs). Gaps in CGM are marked (glucose_missing), never silently filled;
deciding how to handle them for modeling is a windowing-time concern, not
this loader's.

Bolus dose is spread as a constant-rate delivery across each event's
[ts_begin, ts_end) window, proportional to how much of each 5-minute
bucket that window overlaps. Most boluses are instantaneous
(ts_begin == ts_end) and land entirely in one bucket; "normal dual"/
"square dual" boluses have a real duration (30-60 min, confirmed against
all 12 patients) and get spread across it. The true normal/square split
within a "dual" bolus isn't recoverable from this dataset's schema (only
a single total `dose` is recorded), so the whole dose is treated as one
uniform-rate delivery -- a documented approximation, not the true profile.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from glucose_prediction.config import DATA_RAW_DIR, SAMPLING_INTERVAL_MINUTES

PATIENT_YEAR = {
    "559": 2018, "563": 2018, "570": 2018, "575": 2018, "588": 2018, "591": 2018,
    "540": 2020, "544": 2020, "552": 2020, "567": 2020, "584": 2020, "596": 2020,
}
PATIENT_IDS = tuple(PATIENT_YEAR)
TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"
GRID_FREQ = f"{SAMPLING_INTERVAL_MINUTES}min"
DEFAULT_ROOT = DATA_RAW_DIR / "ohio_t1dm"


def _xml_path(root: Path, patient_id: str, split: str) -> Path:
    suffix = "training" if split == "train" else "testing"
    return root / str(PATIENT_YEAR[patient_id]) / split / f"{patient_id}-ws-{suffix}.xml"


def _events(patient_elem, tag: str, value_attrs, span: bool = False) -> pd.DataFrame:
    """Collect one event stream into a DataFrame with parsed datetime column(s).

    Always returns the expected columns, even with zero events -- some real
    patients have no recorded meals/temp_basal events in a given file.
    """
    stream = patient_elem.find(tag)
    rows = []
    if stream is not None:
        for event in stream.findall("event"):
            row = {attr: float(event.get(attr)) for attr in value_attrs}
            if span:
                row["ts_begin"] = pd.to_datetime(event.get("ts_begin"), format=TIMESTAMP_FORMAT)
                row["ts_end"] = pd.to_datetime(event.get("ts_end"), format=TIMESTAMP_FORMAT)
            else:
                row["ts"] = pd.to_datetime(event.get("ts"), format=TIMESTAMP_FORMAT)
            rows.append(row)
    columns = list(value_attrs) + (["ts_begin", "ts_end"] if span else ["ts"])
    return pd.DataFrame(rows, columns=columns)


def _build_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.floor(GRID_FREQ), end.ceil(GRID_FREQ), freq=GRID_FREQ)


def _align_glucose(grid: pd.DatetimeIndex, glucose_df: pd.DataFrame):
    glucose_df = glucose_df.sort_values("ts")
    merged = pd.merge_asof(
        pd.DataFrame({"ts": grid}), glucose_df, on="ts",
        direction="nearest", tolerance=pd.Timedelta(minutes=SAMPLING_INTERVAL_MINUTES / 2),
    )
    return merged["value"].to_numpy(), merged["value"].isna().to_numpy()


def _bucket_sum(grid: pd.DatetimeIndex, timestamps: pd.Series, values: pd.Series) -> pd.Series:
    if timestamps.empty:
        return pd.Series(0.0, index=grid)
    rounded = timestamps.dt.round(GRID_FREQ)
    summed = values.groupby(rounded).sum()
    return summed.reindex(grid, fill_value=0.0)


def _distribute_bolus(grid: pd.DatetimeIndex, bolus_df: pd.DataFrame) -> np.ndarray:
    """Spread each bolus dose across the 5-minute buckets its [ts_begin, ts_end)
    window overlaps, proportional to overlap duration (a constant-rate delivery
    assumption). Instantaneous boluses (ts_begin == ts_end) fall back to
    nearest-bucket assignment, matching the same tolerance _align_glucose uses.
    """
    dose_per_bucket = np.zeros(len(grid))
    if bolus_df.empty:
        return dose_per_bucket

    grid_starts = grid.values
    grid_ends = grid_starts + np.timedelta64(SAMPLING_INTERVAL_MINUTES, "m")
    tolerance = pd.Timedelta(minutes=SAMPLING_INTERVAL_MINUTES / 2)

    for _, event in bolus_df.iterrows():
        ts_begin, ts_end, dose = event["ts_begin"], event["ts_end"], event["dose"]
        duration_minutes = (ts_end - ts_begin) / pd.Timedelta(minutes=1)

        if duration_minutes <= 0:
            nearest = grid.get_indexer([ts_begin], method="nearest", tolerance=tolerance)
            if nearest[0] != -1:
                dose_per_bucket[nearest[0]] += dose
            continue

        overlap_start = np.maximum(grid_starts, np.datetime64(ts_begin))
        overlap_end = np.minimum(grid_ends, np.datetime64(ts_end))
        overlap_minutes = np.clip((overlap_end - overlap_start) / np.timedelta64(1, "m"), 0, None)
        dose_per_bucket += dose * (overlap_minutes / duration_minutes)

    return dose_per_bucket


def _compute_insulin(grid: pd.DatetimeIndex, basal_df: pd.DataFrame,
                      temp_basal_df: pd.DataFrame, bolus_df: pd.DataFrame) -> pd.Series:
    interval_hours = SAMPLING_INTERVAL_MINUTES / 60

    if basal_df.empty:
        rate = pd.Series(0.0, index=grid)
    else:
        basal_df = basal_df.sort_values("ts")
        merged = pd.merge_asof(pd.DataFrame({"ts": grid}), basal_df, on="ts", direction="backward")
        rate = merged["value"].fillna(0.0)
        rate.index = grid

    for _, tb in temp_basal_df.iterrows():
        mask = (grid >= tb["ts_begin"]) & (grid < tb["ts_end"])
        rate[mask] = tb["value"]

    basal_dose = rate * interval_hours
    bolus_dose = _distribute_bolus(grid, bolus_df)
    return basal_dose.to_numpy() + bolus_dose


def load_patient(patient_id: str, split: str = "train", root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Load one patient's XML file into a 5-minute-grid DataFrame.

    Columns: glucose (mg/dL), glucose_missing (bool), carbs_g, insulin_u.
    """
    tree = ET.parse(_xml_path(root, patient_id, split))
    patient_elem = tree.getroot()

    glucose = _events(patient_elem, "glucose_level", ["value"])
    meals = _events(patient_elem, "meal", ["carbs"])
    basal = _events(patient_elem, "basal", ["value"])
    temp_basal = _events(patient_elem, "temp_basal", ["value"], span=True)
    bolus = _events(patient_elem, "bolus", ["dose"], span=True)

    grid = _build_grid(glucose["ts"].min(), glucose["ts"].max())
    glucose_vals, missing = _align_glucose(grid, glucose)
    carbs_vals = _bucket_sum(grid, meals["ts"], meals["carbs"])
    insulin_vals = _compute_insulin(grid, basal, temp_basal, bolus)

    return pd.DataFrame({
        "glucose": glucose_vals,
        "glucose_missing": missing,
        "carbs_g": carbs_vals.to_numpy(),
        "insulin_u": insulin_vals,
    }, index=grid)
