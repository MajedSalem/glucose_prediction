# Progress Log

Status as of 2026-08-30. This documents what has actually been built,
verified, and decided so far — as a technical companion to `README.md`
(the public project pitch) and `plan.md` (the day-by-day personal plan,
kept local/gitignored). Where a decision involved a real trade-off or a
bug was caught, it's recorded here so the reasoning isn't lost.

## Status summary

| Plan stage | Status |
|---|---|
| Day 0 — setup | Done (real data arrived immediately; synthetic-data fallback was skipped) |
| Days 1–2 — data loading & task definition | Done |
| Day 3 — evaluation harness | Done |
| Days 4–5 — baselines | Done |
| Days 6–8 — sequence model | Not started |
| Day 9 — error analysis | Not started |
| Days 10–12 — writeup, polish, buffer | Not started |

All 39 tests pass (`.venv/Scripts/python -m pytest tests/ -v`).

## 1. Repository and environment

- Public repo at `github.com/MajedSalem/glucose_prediction`, `main` branch.
- Python 3.13 virtualenv at `.venv/`, dependencies pinned in
  `requirements.txt`: numpy, pandas, scipy, scikit-learn, statsmodels,
  torch, matplotlib, simglucose, pyyaml, pytest, plus `setuptools<81`
  (simglucose depends on the unmaintained `gym` package, which needs
  `pkg_resources` — removed by default in newer setuptools).
- `glucose_prediction/config.py` centralizes shared constants: data/results
  paths, `SAMPLING_INTERVAL_MINUTES = 5`, `HORIZONS_MINUTES = (30, 60)`,
  `INPUT_WINDOW_MINUTES = 120`, `HYPOGLYCEMIA_THRESHOLD = 70`,
  `HYPERGLYCEMIA_THRESHOLD = 180`.
- Package layout: `glucose_prediction/{data,models,evaluation}/`,
  `scripts/`, `tests/`, `notebooks/`, `data/{raw,processed}/` (gitignored),
  `results/` (gitignored). `README.md` is the public project description;
  `LICENSE` is MIT.
- `plan.md` (the original day-by-day plan, including personal/application
  context) is deliberately **not** committed — it was removed from git
  history entirely (history rewritten and force-pushed) before the repo
  went public, so it never appears even in old commits.

## 2. Data: real OhioT1DM dataset (not synthetic)

The plan's Day 0 called for `simglucose`-generated synthetic data as a
placeholder while the OhioT1DM data-use agreement (DUA) was pending. In
practice, the real dataset was already available, so the synthetic-data
step was skipped entirely and work went straight to the real data.

**Source data**: `data/raw/ohio_t1dm/{2018,2020}/{train,test}/` — copied
from the DUA-provided download into the repo's gitignored `data/raw/`
(never committed; verified byte-identical to the source). Structure:

- **12 unique patients** total, not 6 as originally assumed: 6 in the 2018
  cohort (559, 563, 570, 575, 588, 591) and 6 in the 2020 cohort (540, 544,
  552, 567, 584, 596). Each patient has a `train` and `test` XML file.
- XML streams used: `glucose_level` (CGM, ~5-min cadence), `meal` (carb
  grams, single timestamp), `basal` (U/hr rate, changes irregularly from
  minutes to multiple days apart), `temp_basal` (rate override over a
  `[ts_begin, ts_end)` window, confirmed never overlapping), `bolus` (dose
  in U; mostly instantaneous "normal" boluses, but "normal dual"/"square
  dual" boluses genuinely span ~30 minutes). Other streams present in the
  files (`finger_stick`, `sleep`, `work`, `stressors`, `hypo_event`,
  `illness`, `exercise`, `basis_*`, `acceleration`) are out of scope — not
  part of the task definition (CGM, optionally insulin and carbs).
- Timestamps are `dd-mm-yyyy HH:MM:SS`. Per-patient train/test files are
  chronologically contiguous (test starts 5 minutes after train ends, no
  overlap), so the dataset's own train/test split is used as-is; a
  train/val split within train is left for later (harness/training time),
  not the loader's job.

**Loader — `glucose_prediction/data/ohio_t1dm.py`**: parses one patient's
XML into a clean 5-minute-grid `DataFrame` with columns `glucose` (mg/dL),
`glucose_missing` (bool), `carbs_g`, `insulin_u`.

- CGM alignment: raw `glucose_level` timestamps are essentially exact
  5-minute multiples (2018 files: no jitter at all; 2020 files: sub-2-second
  jitter). A `pd.merge_asof(..., tolerance=2.5min)` aligns them onto a
  strict grid; any grid point with no raw reading nearby is marked
  `glucose_missing = True`, never imputed — deciding how to handle gaps is
  left to future windowing code.
- Real gaps are rare (~0.2–0.4% of intervals per file) but can be large
  (up to ~20 hours in one file) — these show up correctly as runs of
  `glucose_missing`.
- Insulin: `basal` rate is forward-filled onto the grid (`merge_asof`,
  `direction="backward"`), overridden by `temp_basal` during its active
  window, converted to a dose per 5-minute interval, then summed with
  bolus doses. **Known simplification**: bolus dose is treated as
  delivered instantaneously at `ts_begin`, even for the ~30-minute
  extended ("dual"/"square") boluses — flagged in the module docstring as
  a candidate future refinement, not hidden.
- Carbs: summed per 5-minute bucket by rounding each meal's timestamp to
  the nearest grid point.

**Bug caught and fixed during verification**: some real patients (567,
584, 596) have zero recorded `meal` events in a given file. The initial
event-parsing helper returned a columnless empty `DataFrame` in that case,
causing a `KeyError` on `meals["ts"]`. Fixed by always returning the
expected columns regardless of event count.

**Verification performed** (not just "it ran"):
- Grid integrity: strict 5-minute `DatetimeIndex`, zero duplicates, across
  patients from both cohorts and both splits.
- Conservation: total `carbs_g` after bucketing matches the sum of raw
  `<meal carbs=..>` values exactly. Bolus dose conserves to ~99.7% — the
  ~0.3% gap is boluses occurring right at the edge of the CGM timestamp
  range (before the first or after the last glucose reading), which is
  the expected, documented edge effect, not a bug.
- Ran the loader on the sparse-`temp_basal` patient (570, only 3 events)
  to confirm the override logic doesn't error on a near-empty case.

**Batch script — `scripts/build_ohio_dataset.py`**: runs the loader over
all 12 patients × {train, test} and writes 24 CSVs to
`data/processed/ohio_t1dm/{patient_id}_{split}.csv` (also gitignored).
Row counts range ~11,000–14,000 for train files and ~2,500–4,000 for test,
consistent with the source file sizes.

## 3. Evaluation harness (Day 3)

Built before any baseline or model code exists, per the plan's own
rationale: a careful, clinically framed evaluation is the project's
differentiator, not model sophistication.

- **`glucose_prediction/evaluation/metrics.py`**: `rmse`, `mae`,
  `classify_range` (hypo/target/hyper, 70–180 mg/dL inclusive target band),
  `time_in_range` (percentages), `range_agreement` (agreement rate +
  always-3×3 confusion matrix, zero-filled for absent categories). Accepts
  plain numpy arrays or pandas Series, compared positionally. Deliberately
  does **not** filter or impute NaNs — callers (e.g. code using the
  `glucose_missing` flag) must clean data first; an unfiltered call loudly
  returns `NaN` rather than silently averaging around gaps.
- **`glucose_prediction/evaluation/clarke.py`**: a vectorized Clarke Error
  Grid (Clarke et al., 1987) zone classifier (A–E). The boundary
  inequalities were cross-verified against three independent sources that
  each derive from the original paper (a CRAN R package and two
  independent Python implementations, which agree exactly once
  algebraically simplified) before writing any code — this metric makes
  real clinical-safety claims, so getting the geometry right mattered more
  than usual. Zones are checked in strict priority order **A, E, C, D,
  else B**, since some raw inequalities overlap (e.g. `x=180, y=70`
  satisfies both the E and C formulas; E must win). Verified against a
  5,000,000-point uniform random sample: zone-area split A 20.84% / B
  27.91% / C 16.23% / D 15.76% / E 19.26%, matching the theoretical
  ≈20.8/27.9/16.2/15.8/19.3 split almost exactly.
- **`glucose_prediction/evaluation/harness.py`**: `evaluate(y_true, y_pred,
  horizon_minutes, model_name)` flattens RMSE, MAE, time-in-range, range
  agreement, and Clarke zone percentages into one dict — designed to
  accumulate into a `pd.DataFrame` once baselines/models exist to compare.
- No plotting yet (deferred to Days 10–11 per the plan — there's nothing
  real to plot until baselines exist).
- **30 tests** across `tests/test_metrics.py`, `tests/test_clarke.py`,
  `tests/test_harness_smoke.py`: known-value checks, inclusive-boundary
  classification, explicit Clarke boundary-strictness tests at every seam
  (x/y = 70, 180, 240, 290), a check-order regression test, a
  vectorized-vs-naive-reference cross-check, the uniform-random
  distribution sanity check above, and a smoke test on real `559_train.csv`
  glucose values (Gaussian-noised, not shifted, to avoid overlapping with
  the not-yet-built persistence baseline) confirming sane RMSE/MAE/TIR/zone
  output with no NaNs.

## 4. Baselines (Days 4–5)

Three baselines, chosen with the user (ridge over ARIMA/Kalman — simpler
and more reliable across 12 patients), evaluated through the Day 3 harness
on the SAME held-out test split for a fair comparison.

**Windowing — `glucose_prediction/data/windowing.py`** (new; nothing
previously turned the cleaned CSVs into ML examples). `make_windows(df)`
slides a fixed 24-sample (120-minute) window of `glucose` over a patient's
DataFrame and returns `(X, y, t)`: `X` is the lag-feature matrix, `y` is a
`{horizon_minutes: target_array}` dict, `t` is the timestamp of each
window's last input sample. Purely positional (every processed CSV has a
strictly regular 5-minute grid — confirmed by checking `diff().unique()`
— so real gaps show up only as `glucose_missing=True` rows, never as
missing grid rows). A window or target touching any missing point is
dropped entirely, never imputed, consistent with the harness's own
"caller filters NaN" convention. One shared validity mask applies across
both horizons, so `y[30]` and `y[60]` always stay row-aligned with `X`.
Implemented vectorized (`numpy.lib.stride_tricks.sliding_window_view`) —
verified bit-identical to a naive per-window loop and ~70–90x faster.
Valid window counts are comfortably large everywhere (e.g. 559_train:
12082 rows → 9342 valid windows; worst case 567_train: 13537 → 8921).

**Baselines — `glucose_prediction/models/baselines.py`** (new):
- `persistence_predict(X)`: last observed value in the window, for every
  horizon.
- `linear_extrapolation_predict(X, horizon_minutes)`: least-squares line
  fit per window, extrapolated forward. Implemented as closed-form batched
  OLS (not per-window `np.polyfit`, for speed) — verified bit-identical to
  `np.polyfit` on 200 real windows, and to two hand-computed cases.
- `fit_ridge` / `ridge_predict`: sklearn `Ridge(alpha=1.0)`, no tuning
  (a baseline, not a tuned model), glucose-lag features only (no
  carbs/insulin — kept for a fair "CGM-history-only" comparison against
  persistence/linear extrapolation; insulin/carbs as inputs is explicitly
  deferred to the deep-model stage). One model per (patient, horizon) —
  24 fitted models total — trained on that patient's train-split windows,
  evaluated on that patient's test-split windows.

**Runner — `scripts/run_baselines.py`**: loops all 12 patients × 2
horizons, writes `results/baselines_per_patient.csv` (72 rows) and
`results/baselines_summary.csv` (6 rows, mean±std across patients — the
headline comparison). Both gitignored, like other `results/` output.
Full run (24 CSVs loaded, windowed, 24 Ridge models fit, all three
baselines scored): well under a second wall-clock.

**Real results** (mean ± std RMSE across all 12 patients):

| horizon | persistence | linear extrapolation | ridge |
|---|---|---|---|
| 30 min | 23.18 ± 2.83 | 38.08 ± 5.65 | **19.85 ± 2.30** |
| 60 min | 37.99 ± 4.62 | 59.23 ± 8.61 | **33.56 ± 3.82** |

Ridge beats persistence at **every one of the 12 patients, both
horizons**. Linear extrapolation is substantially *worse* than persistence
everywhere — per-window OLS slope amplifies CGM sensor noise into an
overshooting trend, especially near local maxima/minima, a real and
well-documented effect for CGM data. Reported honestly rather than tuned
away, per `plan.md`'s own "report the persistence baseline honestly"
guardrail — this asymmetry (a "smarter"-looking method losing to the
naive floor) is itself a good candidate for the Day 10–11 limitations
section.

**9 new tests** in `tests/test_windowing.py` and `tests/test_baselines.py`:
exact-value windowing tests on hand-built series (including the
shared-validity-mask-across-horizons case), a skip-guarded real-data count
check, exact hand-computed linear-extrapolation cases, and a Ridge smoke
test confirming it beats a constant-mean baseline on synthetic data.

## 5. Known simplifications / limitations so far

(Candidates for the eventual README limitations section, Days 10–11.)

- Bolus dose treated as instantaneous at `ts_begin`; ~30-minute extended
  boluses aren't spread over their actual delivery window.
- Only `glucose_level`, `meal`, `basal`, `temp_basal`, `bolus` streams are
  used; sleep/exercise/stress/heart-rate/skin-temperature streams present
  in the data are not (not part of the task definition).
- Clarke Error Grid only — Parkes/Consensus grid (which uses meaningfully
  different, diabetes-type-specific boundaries and has no Zone E) was
  considered and deliberately not implemented, to keep scope to one
  well-precedented grid.
- CGM gaps are marked, not imputed — dropped entirely at window-building
  time rather than interpolated.
- Baselines use glucose history only, no insulin/carbs — a deliberate
  scope boundary to keep the three-way comparison apples-to-apples;
  insulin/carbs as model inputs is deferred to the deep-model stage.
- Ridge uses default `alpha=1.0`, no hyperparameter search — appropriate
  for a baseline, not for the eventual deep model.

## 6. Not yet started

- Sequence model (LSTM/GRU, optionally a small Transformer) — Days 6–8.
- Error analysis by regime (postprandial, overnight, hypoglycemia) — Day 9.
- Results table, Clarke-grid figure, README results/limitations sections,
  final polish — Days 10–12.
