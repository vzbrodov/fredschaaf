#!/usr/bin/env python3
"""Build the orbit-fit ground table with one independent row per series."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.series import build_orbit_ground_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-stack",
        type=Path,
        default=Path("outputs/ground_full_stack_series.csv"),
    )
    parser.add_argument(
        "--appulse",
        type=Path,
        default=Path("outputs/ground_appulse_series_20250903_R.csv"),
        help="specialized measurement that replaces the 20250903_R full stack",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/ground_orbit_series.csv")
    )
    args = parser.parse_args()

    full_stack = pd.read_csv(args.full_stack)
    appulse = pd.read_csv(args.appulse)
    result = build_orbit_ground_series(
        full_stack, {"20250903_R": appulse}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    display = result.copy()
    display["sigma_xi_mas"] = display.cov_xi_xi_mas2.pow(0.5)
    display["sigma_eta_mas"] = display.cov_eta_eta_mas2.pow(0.5)
    print(
        display[
            ["series_id", "source", "central_utc", "sigma_xi_mas", "sigma_eta_mas"]
        ].to_string(index=False)
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
