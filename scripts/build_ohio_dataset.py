"""Batch-run the OhioT1DM loader across patients and splits into clean CSVs."""
import argparse
from pathlib import Path

from glucose_prediction.config import DATA_PROCESSED_DIR
from glucose_prediction.data.ohio_t1dm import DEFAULT_ROOT, PATIENT_IDS, load_patient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="+", default=list(PATIENT_IDS))
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DATA_PROCESSED_DIR / "ohio_t1dm")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for patient_id in args.patients:
        for split in args.splits:
            df = load_patient(patient_id, split, root=args.source_dir)
            out_path = args.output_dir / f"{patient_id}_{split}.csv"
            df.to_csv(out_path, index_label="timestamp")
            print(f"{patient_id} {split}: {len(df)} rows, "
                  f"{df['glucose_missing'].sum()} missing -> {out_path}")


if __name__ == "__main__":
    main()
