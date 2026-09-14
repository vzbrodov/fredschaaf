#!/usr/bin/env python3
"""Build one conservative astrometric row per detected full-series stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.series import load_series_config


DEFAULT_CONFIGS = (
    Path("configs/20250902_R.toml"),
    Path("configs/20250903_R.toml"),
    Path("configs/20251010_R.toml"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/ground_full_stack_series.csv")
    )
    args = parser.parse_args()
    rows = []
    for config_path in args.config or DEFAULT_CONFIGS:
        config = load_series_config(config_path)
        series_id = str(config.series["id"])
        detection = pd.read_csv(
            args.input_dir / f"full_stack_detection_{series_id}.csv"
        ).iloc[0]
        if str(detection.detection_ok).lower() != "true":
            raise SystemExit(f"Full-stack detection failed QC for {series_id}")
        split = pd.read_csv(
            args.input_dir / f"full_stack_split_check_{series_id}.csv"
        )
        if len(split) != 2 or split.at_search_boundary.astype(bool).any():
            raise SystemExit(f"Split-stack validation failed for {series_id}")
        split_delta = (
            split.iloc[0][["offset_xi_mas", "offset_eta_mas"]].to_numpy(float)
            - split.iloc[1][["offset_xi_mas", "offset_eta_mas"]].to_numpy(float)
        ) / 2.0
        formal = np.array(
            [
                [detection.formal_error_xi_mas**2, detection.cov_xi_eta_mas2],
                [detection.cov_xi_eta_mas2, detection.formal_error_eta_mas**2],
            ]
        )
        covariance = formal + np.outer(split_delta, split_delta)
        rows.append({
            "source": "ground_full_motion_stack",
            "series_id": series_id,
            "observatory_code": str(config.series["observatory_code"]),
            "filter": str(config.series["filter"]),
            "central_utc": detection.central_utc,
            "jd_utc": float(Time(str(detection.central_utc), scale="utc").jd),
            "ra_deg": detection.ra_deg,
            "dec_deg": detection.dec_deg,
            "oc_xi_mas": detection.offset_xi_mas,
            "oc_eta_mas": detection.offset_eta_mas,
            "cov_xi_xi_mas2": covariance[0, 0],
            "cov_xi_eta_mas2": covariance[0, 1],
            "cov_eta_eta_mas2": covariance[1, 1],
            "formal_error_xi_mas": detection.formal_error_xi_mas,
            "formal_error_eta_mas": detection.formal_error_eta_mas,
            "split_systematic_xi_mas": split_delta[0],
            "split_systematic_eta_mas": split_delta[1],
            "split_difference_2d_mas": split.split_difference_2d_mas.iloc[0],
            "fitted_flux_snr": detection.fitted_flux_snr,
            "frames": int(detection.frames),
            "span_minutes": detection.span_minutes,
            "ephemeris_id": str(config.ephemeris["id"]),
            "covariance_method": "matched-psf-formal+odd-even-half-difference",
            "quality_ok": True,
            "quality_flags": "ok",
        })
    result = pd.DataFrame(rows)
    for row in result.itertuples():
        covariance = np.array(
            [
                [row.cov_xi_xi_mas2, row.cov_xi_eta_mas2],
                [row.cov_xi_eta_mas2, row.cov_eta_eta_mas2],
            ]
        )
        if np.any(np.linalg.eigvalsh(covariance) <= 0):
            raise SystemExit(f"Non-positive covariance for {row.series_id}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    display = result.copy()
    display["sigma_xi_mas"] = np.sqrt(display.cov_xi_xi_mas2)
    display["sigma_eta_mas"] = np.sqrt(display.cov_eta_eta_mas2)
    print(
        display[
            [
                "series_id",
                "central_utc",
                "frames",
                "fitted_flux_snr",
                "ra_deg",
                "dec_deg",
                "sigma_xi_mas",
                "sigma_eta_mas",
                "split_difference_2d_mas",
            ]
        ].to_string(index=False)
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
