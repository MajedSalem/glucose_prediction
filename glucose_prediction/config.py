"""Shared paths and constants for the glucose-prediction pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

# Resample CGM streams to this grid before windowing.
SAMPLING_INTERVAL_MINUTES = 5

# Prediction horizons, in minutes ahead of the last observed CGM reading.
HORIZONS_MINUTES = (30, 60)

# Length of the input window fed to models, in minutes.
INPUT_WINDOW_MINUTES = 120

# Glucose ranges (mg/dL) used for time-in-range and hypoglycemia analysis.
HYPOGLYCEMIA_THRESHOLD = 70
HYPERGLYCEMIA_THRESHOLD = 180
