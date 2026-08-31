# Progress Log

Status as of 2026-08-31. This documents what has actually been built,
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
| Days 6–8 — sequence model | Done |
| Day 9 — error analysis | Done |
| Days 10–12 — writeup, polish, buffer | Not started |

All 49 tests pass (`.venv/Scripts/python -m pytest tests/ -v`).

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

## 5. Sequence model (Days 6–8)

Chosen with the user: **LSTM** (plan's first-listed option; a Transformer
stayed explicitly out of scope), trained **pooled** across all 12 patients
(over per-patient or "both") for far more effective training data and
because it's the standard approach in the CGM-forecasting literature,
evaluated **per-patient** at test time so the comparison against
per-patient ridge stays apples-to-apples.

**Architecture — `glucose_prediction/models/sequence.py`** (new):
`LSTMForecaster` is a single shared LSTM encoder (`input_size=1`,
glucose-lag-only, matching the baselines exactly) feeding one `Linear`
head with 2 outputs (30min and 60min predicted simultaneously) — more
parameter-efficient than training two separate models. Hidden size was
**measured, not guessed**: benchmarked 16/32/64 at `num_layers=1` and got
best validation RMSE 28.85/28.94/28.95 — differences within noise across a
13x parameter range, so bigger doesn't help here. Picked `hidden_size=32`
(4,546 params) as a defensible modest middle choice; this measurement
itself is the evidence that the ceiling is the glucose-only-lag-feature
task, not model capacity. No dropout (`num_layers=1` makes PyTorch's LSTM
dropout a no-op anyway; ~4.5k params against ~102k training rows plus
early stopping keeps overfitting risk low).

**Data pipeline**: reuses `windowing.make_windows` as-is. A chronological
15%-tail-per-patient validation split is taken from each patient's TRAIN
file *before* pooling (so no cross-patient or in-time leakage), then
pooled across all 12 patients: **102,442** pooled training windows, 18,079
pooled validation windows, 28,162 pooled test windows (evaluated
per-patient). Normalization: a single `mu=157.686, sigma=60.182` computed
only from pooled training glucose values, applied identically to
validation/test, predictions un-normalized back to mg/dL before hitting
the harness. **This exact un-normalization step was explicitly verified**
during design (a real bug class: normalized-space predictions look like
near-zero-mean noise, not mg/dL) and is now a permanent unit test.

**Training**: `nn.MSELoss()` over both horizon outputs jointly (same
`mu/sigma`, so no horizon-specific weighting needed), Adam `lr=1e-3` (no
tuning needed — convergence was smooth), batch size 256, early stopping
patience=5 on pooled validation RMSE, `max_epochs=50`. Real run: stopped
at epoch 19, best val RMSE 28.941, ~47 seconds total wall-clock on this
CPU-only machine (`torch.cuda.is_available()` is `False`) — confirmed
stable across 3 random seeds during design verification (epochs 19/27/20,
final RMSE within a tight 19.29–19.37 / 32.63–32.74 range).

**Runner — `scripts/run_sequence_model.py`**: mirrors `run_baselines.py`'s
shape (per-patient DataFrame + mean±std summary, saved under
`RESULTS_DIR`). Real results (mean ± std RMSE/MAE across the 12 patients):

| horizon | persistence | linear extrap. | ridge | **lstm_pooled** |
|---|---|---|---|---|
| 30min RMSE | 23.18±2.83 | 38.08±5.65 | 19.85±2.30 | **19.37±2.17** |
| 60min RMSE | 37.99±4.62 | 59.23±8.61 | 33.56±3.82 | **32.74±3.55** |
| 30min MAE | 16.78±1.99 | 27.42±3.83 | 14.24±1.54 | **13.89±1.58** |
| 60min MAE | 28.13±3.36 | 43.26±6.26 | 25.25±3.04 | **24.58±2.82** |

**Honest framing**: the LSTM beats ridge, and this held up across 3 seeds
(not a fluke), but the win is modest (~0.5 RMSE, ~2–3% relative) and *not*
universal per-patient (wins on 10–12 of 12 patients depending on
seed/horizon) — a narrower, less uniform win than ridge's clean 12/12
sweep over persistence. Read: a 24-step lag window carries limited
information regardless of which model reads it, and ridge was already
close to that ceiling. Reported as-is, not rounded up to "the LSTM wins."

**5 new tests** in `tests/test_sequence.py`: forward-pass shape,
normalize/denormalize round-trip (the un-normalization correctness check
above, made permanent), chronological-split ordering/sizing, a synthetic
training-reduces-loss smoke test, and a skip-guarded real-data smoke test
on 2 patients with a reduced training budget.

## 6. Error analysis by clinical regime (Day 9)

Pure analysis on top of already-built infrastructure — no new models, no
new metrics. **Key structural finding that shaped the whole design**: for
the hypoglycemia-true subset (`y_true < 70`), Clarke zones B and C are
mathematically impossible (verified by tracing the zone formulas by hand
and empirically — exactly `0.0%` in every real hypoglycemia row). So "does
it miss lows?" is already fully answered by the existing `zone_A/D/E_pct`
columns on that subset: zone A is the hit rate, zone D is "missed, called
normal," zone E is "missed, called high" — no new metric code needed at
all, only slicing.

**`glucose_prediction/evaluation/regimes.py`** (new): `postprandial_mask`,
`overnight_mask`, `hypoglycemia_mask` — all anchored at the prediction
**target time** (`t + horizon_minutes`), not the input window's
last-observed time, since a meal or a night hour is defined by when the
prediction lands. Regime definitions chosen from real data, not guessed:
- **Postprandial** = a meal within the last 120 minutes (inclusive). Real
  median gap between meals is 300 min; only 13.4% of gaps are under 120
  min, so this rarely conflates two unrelated meals.
- **Overnight** = target-time local hour in `[0, 6)`.
- **Hypoglycemia** = `y_true < 70` at target time (793/796 pooled windows
  at 30/60min — 2.8% of test windows, small but not degenerate). Patient
  567's test split has zero recorded meals, contributing 0 postprandial
  windows for that patient — a footnote, not a bug.

**`scripts/run_error_analysis.py`** (new): reuses `build_pooled_data`,
`train_lstm`, `predict`, `fit_ridge`, `ridge_predict`,
`persistence_predict`, `linear_extrapolation_predict`, `make_windows`, and
`harness.evaluate` as-is — pure slicing/aggregation, zero new metric code.
Writes 3 CSVs to `results/`: `error_analysis_by_regime.csv` (32 rows: 4
models × 2 horizons × 4 regimes incl. `overall`), `error_analysis_hypo_
detection.csv` (8 rows), `error_analysis_lstm_vs_ridge_per_patient.csv`
(24 rows).

**By-regime RMSE** (pooled across all 12 patients):

| horizon | regime (n) | persistence | linear_extrap | ridge | lstm_pooled |
|---|---|---|---|---|---|
| 30min | postprandial (7086) | 28.09 | 42.87 | 24.23 | **23.61** |
| 30min | overnight (7507) | 15.39 | 23.32 | 14.39 | **13.92** |
| 30min | hypoglycemia (793) | 16.69 | 31.92 | 16.42 | 20.58 |
| 60min | postprandial (7016) | 44.51 | 63.24 | 39.51 | **38.81** |
| 60min | overnight (7457) | 25.67 | 37.52 | 25.35 | **24.17** |
| 60min | hypoglycemia (796) | 35.29 | 56.27 | 37.69 | 41.34 |

**"Does it miss lows?"** — hit rate (zone A), missed→normal (zone D),
missed→high/dangerous (zone E), within the hypoglycemia-true subset:

| model | horizon | hit_rate | missed→normal | missed→high |
|---|---|---|---|---|
| persistence | 30 | **71.75%** | 28.25% | 0.00% |
| linear_extrapolation | 30 | 82.09% | 17.65% | 0.25% |
| ridge | 30 | 57.76% | 42.24% | 0.00% |
| **lstm_pooled** | 30 | **25.98%** | **74.02%** | 0.00% |
| persistence | 60 | **47.11%** | 51.88% | 1.01% |
| linear_extrapolation | 60 | 73.49% | 21.98% | 4.52% |
| ridge | 60 | 7.66% | 92.21% | 0.13% |
| **lstm_pooled** | 60 | **0.63%** | **99.25%** | 0.13% |

**LSTM vs ridge, per patient** (overall RMSE): LSTM wins 21/24
(patient, horizon) pairs. Loses at 567 (30min), 570 (60min), 584 (60min).

**The headline finding — this is the one to lead with in the writeup, not
bury**: the LSTM's RMSE edge over ridge is real and holds in *both* the
postprandial and overnight regimes (not a one-subset fluke). But on the
safety-critical question, accuracy and hypo-sensitivity move in **opposite
directions**: the model with the best average RMSE (LSTM) is dramatically
the *worst* at catching a real low, and this gets worse — not better — at
the longer horizon (26% hit rate at 30min, a mere **0.6%** at 60min) versus
persistence's 72%/47%. This is a textbook regression-to-the-mean effect:
hypoglycemia is only ~2.8% of training windows, so a fitted model (more so
the higher-capacity LSTM than linear ridge) learns to systematically
under-predict the depth of downward excursions, minimizing overall RMSE at
the direct cost of ever crossing the 70 mg/dL line. Reassuringly, the most
dangerous failure mode (Zone E — predicting high when truly low) stays
rare throughout (0–4.5%) — the dominant risk is Zone D (silently calling a
real low "normal"), not an active reversal. Linear extrapolation's
apparently-best hit rate is a "right for the wrong reason" artifact (its
already-documented per-window OLS overshoot happens to frequently cross
below 70 during a real downward trend) and is reported with that caveat,
not touted as "safest." **Net: the naive persistence baseline is the most
clinically sensitive detector of real hypoglycemia among all four
predictors** — this is exactly the failure-mode honesty Day 9 exists to
surface.

**5 new tests** in `tests/test_regimes.py`: exact boundary-inclusive
postprandial-lookback cases, the no-meals-ever case, a target with no
earlier history at all, half-open overnight-hour boundaries, and the
hypoglycemia threshold matching `metrics.classify_range`'s own convention.

## 7. Known simplifications / limitations so far

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
- Baselines and the LSTM both use glucose history only, no insulin/carbs —
  a deliberate scope boundary to keep the whole comparison apples-to-apples;
  insulin/carbs as model inputs is deferred to future work.
- Ridge uses default `alpha=1.0`; the LSTM uses one hidden size chosen from
  a 3-point sweep (16/32/64) — no broader hyperparameter search on either,
  appropriate for a "keep it modest" baseline/first-pass model.
- LSTM early stopping monitors one **pooled** validation RMSE across all
  12 patients combined, not per-patient — a single shared stopping point.
- A single global normalization (`mu`/`sigma`) is used for the pooled
  LSTM rather than per-patient normalization — consistent with the
  "one shared pooled model" design, but means patients with unusually
  high/low glucose ranges are normalized identically to everyone else.
- The LSTM's win over ridge is real but narrow and not universal
  per-patient (see above) — worth surfacing honestly, not smoothing over.
- **The LSTM (and to a lesser extent ridge) is meaningfully worse than
  naive persistence at detecting real hypoglycemia**, despite having the
  best average RMSE — a regression-to-the-mean effect from hypoglycemia
  being ~2.8% of training data. A hypo-weighted loss (already on plan.md's
  own "what I'd do next" list) would directly target this, not implemented
  here — this project reports the failure mode rather than papering over
  it with a leaderboard-only view of RMSE.
- Regime definitions (120min postprandial lookback, `[0,6)` overnight) are
  one reasonable, data-justified choice each, not the only valid one.

## 8. Not yet started

- Results table, Clarke-grid figure, README results/limitations sections,
  final polish — Days 10–12. The Day 9 hypo-detection finding should be
  the headline of the writeup, not a footnote.
