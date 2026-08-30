# Glucose Prediction (OhioT1DM)

Blood-glucose forecasting for type 1 diabetes, benchmarking classical time-series
baselines against a deep sequence model, with a clinically framed evaluation.

## Goal

Predict continuous glucose monitor (CGM) readings 30 and 60 minutes ahead from
recent CGM history (optionally insulin and carbohydrate intake), and evaluate
predictions the way a clinician would: not just error magnitude, but whether
errors are safe (e.g. missed hypoglycemia is worse than a missed high).

This project favors a small, finished, honest comparison over a
state-of-the-art model.

## Data

- **[OhioT1DM](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html)** — 12 real
  type 1 diabetes patients with CGM, insulin, carbohydrates, sleep, and heart
  rate. Requires a signed data-use agreement (DUA) with Ohio University.
  Raw data is **not** included in this repository — see [Data & privacy](#data--privacy).
- **[simglucose](https://github.com/jxx123/simglucose)** — an open-source
  Python UVA/Padova virtual-patient simulator, used to generate synthetic data
  while the OhioT1DM DUA is pending, and as an easy way to reproduce the
  pipeline without restricted data.

## Task

From the last ~2 hours of CGM (optionally insulin and carbs), predict glucose
at +30 and +60 minutes. Data is resampled to a 5-minute grid; splits are by
patient and chronological, so there is no leakage.

## Methodology

- **Evaluation harness first**: RMSE and MAE at each horizon, time-in-range
  agreement, and a Clarke (or Parkes) error grid that scores clinical danger
  rather than raw error size.
- **Baselines**: persistence (glucose stays where it is), linear
  extrapolation, and one classical model (ARIMA or ridge regression on lagged
  features, and/or a Kalman filter).
- **Deep model**: an LSTM or GRU (optionally a small Transformer), trained
  pooled and/or per-patient, compared against the baselines on the exact same
  harness.
- **Error analysis**: where the model wins or loses relative to baselines —
  after meals, overnight, and around hypoglycemia — with the safety-critical
  case (missed lows) called out explicitly.

## Repository structure

```
glucose_prediction/
├── glucose_prediction/       # package
│   ├── config.py             # paths, horizons, sampling constants
│   ├── data/                 # OhioT1DM and simglucose loaders
│   ├── models/               # baselines and sequence models
│   └── evaluation/           # metrics and Clarke/Parkes grid
├── scripts/                  # entry points (data generation, training, evaluation)
├── notebooks/                # exploratory analysis
├── tests/
├── data/                     # gitignored; raw/processed data lives here locally
└── results/                  # gitignored; metrics tables and figures
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Scripts are run as modules from the repository root, e.g.:

```bash
python -m scripts.generate_synthetic_data
```

## Data & privacy

OhioT1DM is distributed under a data-use agreement. This repository never
commits raw patient data — only code and aggregate, de-identified results.
`data/raw/` and `data/processed/` are gitignored.

## Status

Work in progress. Results, a Clarke-grid figure, and a limitations section
will be added here as the project develops.

## License

MIT — see [LICENSE](LICENSE).
