#!/usr/bin/env python3
"""Build a covariance-complete common row for one ground observing series."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.series import build_ground_series, load_series_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/20250903_R.toml"),
        help="TOML configuration for the observing series",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/ground_series.csv")
    )
    args = parser.parse_args()

    table = build_ground_series(load_series_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
