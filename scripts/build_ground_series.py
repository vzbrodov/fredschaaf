#!/usr/bin/env python3
"""Convert current ground-based reductions to the common series table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.series import load_ground_series_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=Path("outputs/two_zone_result_20250903_R.csv"),
    )
    parser.add_argument("--series-id", default="20250903_R")
    parser.add_argument("--filter", default="R")
    parser.add_argument("--observatory-code", default="217")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/pulkovo_series.csv")
    )
    args = parser.parse_args()

    table = load_ground_series_result(
        args.result,
        series_id=args.series_id,
        filter_name=args.filter,
        observatory_code=args.observatory_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
