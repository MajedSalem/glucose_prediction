"""Days 10-11: generate the public writeup deliverables. Prints markdown
tables (headline RMSE/MAE, regime breakdown, hypo-detection) to stdout for
copy-paste into README.md, and saves the Clarke Error Grid comparison figure
(persistence vs lstm_pooled, 30 vs 60min) to disk. Pure aggregation/plotting
over already-tested code -- no new metrics, no new training logic.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from glucose_prediction.config import (
    DATA_PROCESSED_DIR, FIGURES_DIR, HORIZONS_MINUTES,
    HYPERGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, RESULTS_DIR,
)
from glucose_prediction.data.ohio_t1dm import PATIENT_IDS
from glucose_prediction.evaluation.clarke import ZONES, clarke_zones, zone_percentages
from scripts.run_error_analysis import collect_predictions, hypo_detection_table, pooled_regime_table

MODEL_LABELS = {
    "persistence": "Persistence",
    "linear_extrapolation": "Linear extrapolation",
    "ridge": "Ridge",
    "lstm_pooled": "LSTM (pooled)",
}
ZONE_COLORS = {"A": "#c7e9c0", "B": "#fdd0a2", "C": "#fdae6b", "D": "#fc9272", "E": "#de2d26"}


def print_headline_table(results_dir: Path):
    """Read the already-generated per-model summary CSVs (mean+-std RMSE/MAE
    across patients, the exact numbers already verified in progress.md) and
    print one combined markdown table.
    """
    baselines = pd.read_csv(results_dir / "baselines_summary.csv", header=[0, 1], index_col=[0, 1])
    sequence = pd.read_csv(results_dir / "sequence_model_summary.csv", header=[0, 1], index_col=[0, 1])
    summary = pd.concat([baselines, sequence])

    print("\n| Model | Horizon | RMSE | MAE |")
    print("|---|---|---|---|")
    for model in MODEL_LABELS:
        for h in HORIZONS_MINUTES:
            row = summary.loc[(model, h)]
            rmse = f"{row[('rmse', 'mean')]:.2f} ± {row[('rmse', 'std')]:.2f}"
            mae = f"{row[('mae', 'mean')]:.2f} ± {row[('mae', 'std')]:.2f}"
            print(f"| {MODEL_LABELS[model]} | {h}min | {rmse} | {mae} |")


def print_regime_table(by_regime: pd.DataFrame):
    print("\n| Horizon | Regime (n) | Persistence | Linear extrap. | Ridge | LSTM (pooled) |")
    print("|---|---|---|---|---|---|")
    for h in HORIZONS_MINUTES:
        for regime in ("postprandial", "overnight", "hypoglycemia"):
            sub = by_regime[(by_regime["horizon_minutes"] == h) & (by_regime["regime"] == regime)]
            n = int(sub["n"].iloc[0])
            vals = {r.model: r.rmse for r in sub.itertuples()}
            print(f"| {h}min | {regime} ({n}) | {vals['persistence']:.2f} | "
                  f"{vals['linear_extrapolation']:.2f} | {vals['ridge']:.2f} | {vals['lstm_pooled']:.2f} |")


def print_hypo_table(hypo: pd.DataFrame):
    print("\n| Model | Horizon | Hit rate | Missed -> normal | Missed -> high |")
    print("|---|---|---|---|---|")
    for row in hypo.itertuples():
        print(f"| {MODEL_LABELS[row.model]} | {row.horizon_minutes}min | "
              f"{row.hit_rate_pct:.2f}% | {row.missed_as_normal_pct:.2f}% | {row.missed_as_high_pct:.2f}% |")


def _zone_background(ax, grid_max=400, resolution=300):
    """Shade each pixel by the zone clarke_zones() actually assigns it --
    reuses the tested classifier instead of hand-transcribing boundary-line
    coordinates that could drift from clarke.py's real priority-ordered
    inequalities.
    """
    x = np.linspace(0, grid_max, resolution)
    y = np.linspace(0, grid_max, resolution)
    xx, yy = np.meshgrid(x, y)
    zones = clarke_zones(xx.ravel(), yy.ravel()).reshape(xx.shape)
    zone_idx = np.searchsorted(np.array(ZONES), zones)
    cmap = ListedColormap([ZONE_COLORS[z] for z in ZONES])
    ax.pcolormesh(xx, yy, zone_idx, cmap=cmap, vmin=0, vmax=len(ZONES) - 1,
                  shading="auto", alpha=0.5, zorder=0)


def _panel(ax, y_true, y_pred, title):
    _zone_background(ax)
    ax.scatter(y_true, y_pred, s=4, alpha=0.15, color="black", zorder=1, linewidths=0)
    ax.axvline(HYPOGLYCEMIA_THRESHOLD, color="black", linestyle="--", linewidth=0.7, zorder=2)
    ax.axvline(HYPERGLYCEMIA_THRESHOLD, color="black", linestyle="--", linewidth=0.7, zorder=2)
    pct = zone_percentages(clarke_zones(y_true, y_pred))
    ann = " / ".join(f"{z}:{pct[f'zone_{z}_pct']:.1f}%" for z in ZONES)
    ax.set_title(f"{title}\n{ann}", fontsize=9)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Reference glucose (mg/dL)")
    ax.set_ylabel("Predicted glucose (mg/dL)")


def build_clarke_grid_figure(predictions, patient_ids, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for row, model in enumerate(("persistence", "lstm_pooled")):
        for col, h in enumerate(HORIZONS_MINUTES):
            yt_parts, yp_parts = [], []
            for patient_id in patient_ids:
                d = predictions[(patient_id, h)]
                yt_parts.append(d["y_true"])
                yp_parts.append(d["preds"][model])
            y_true, y_pred = np.concatenate(yt_parts), np.concatenate(yp_parts)
            _panel(axes[row, col], y_true, y_pred, f"{MODEL_LABELS[model]}, {h}min")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="+", default=list(PATIENT_IDS))
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR / "ohio_t1dm")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print_headline_table(args.results_dir)

    predictions = collect_predictions(args.patients, args.processed_dir, seed=args.seed)
    by_regime = pooled_regime_table(predictions, args.patients)
    hypo = hypo_detection_table(by_regime)
    print_regime_table(by_regime)
    print_hypo_table(hypo)

    output_path = args.output_dir / "clarke_grid.png"
    build_clarke_grid_figure(predictions, args.patients, output_path)
    print(f"\nSaved Clarke grid figure to {output_path}")


if __name__ == "__main__":
    main()
