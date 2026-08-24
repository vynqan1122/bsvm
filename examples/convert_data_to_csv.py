"""Convert local datasets in data/ to a standard CSV layout.

By default, the script scans data/, skips cache/ and data/csv/, and writes:

    data/csv/<dataset>.csv
    data/csv/manifest.csv

Each output CSV has the label column renamed to "target" and moved to the end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roch_bsvm.data_conversion import convert_directory_to_csv, load_target_overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output-dir", default="data/csv")
    parser.add_argument(
        "--target-column",
        default=None,
        help=(
            "Optional label column name. If omitted, the script tries target, "
            "class, binaryClass, label, y, then falls back to the last column."
        ),
    )
    parser.add_argument("--target-output-name", default="target")
    parser.add_argument(
        "--target-overrides",
        default="data/target_overrides.csv",
        help="CSV with columns dataset,target_column for datasets without clear labels.",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail when an output CSV already exists.",
    )
    args = parser.parse_args()

    manifest = convert_directory_to_csv(
        ROOT / args.input_dir,
        ROOT / args.output_dir,
        target_column=args.target_column,
        target_output_name=args.target_output_name,
        recursive=args.recursive,
        overwrite=not args.no_overwrite,
        target_overrides=load_target_overrides(ROOT / args.target_overrides),
    )
    print(manifest.to_string(index=False))
    print(f"\nWrote {(ROOT / args.output_dir / 'manifest.csv').resolve()}")


if __name__ == "__main__":
    main()
