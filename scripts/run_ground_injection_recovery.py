#!/usr/bin/env python3
"""Inject moving sources into real frames and measure recovery bias."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.wcs import FITSFixedWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.detection import build_full_stack_detection
from fredschaaf_astrometry.ground import MAS_PER_DEGREE
from fredschaaf_astrometry.ground_pipeline import reduce_series
from fredschaaf_astrometry.series import load_series_config


def main() -> None:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    warnings.filterwarnings("ignore", message="The fit may not have converged.*")
    warnings.filterwarnings("ignore", message="Input data contains invalid values.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/20250903_R.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--injections", type=int, default=8)
    parser.add_argument("--radius-px", type=float, default=15.0)
    parser.add_argument("--flux-scale", type=float, default=1.0)
    parser.add_argument("--search-radius-px", type=float, default=1.5)
    parser.add_argument("--minimum-snr", type=float, default=8.0)
    args = parser.parse_args()
    if args.injections < 4 or args.radius_px <= 0 or args.flux_scale <= 0:
        raise SystemExit("Need at least four injections and positive radius/flux scale")

    config = load_series_config(args.config)
    products = reduce_series(config)
    psf_half = int(config.reduction["psf_half_size_px"])
    stack_half = max(
        int(config.reduction["stack_half_size_px"]),
        int(np.ceil(args.radius_px + psf_half + args.search_radius_px + 3)),
    )
    real = build_full_stack_detection(
        products, config, maximum_offset_px=8.0, stack_half_size_px=stack_half
    )
    injection_flux = float(real.flux * args.flux_scale)
    accepted = real.frame_indices
    mean_jacobian = np.mean(
        [products.frame_models[index].coefficients[1:].T for index in accepted], axis=0
    )

    rows = []
    for injection_id, angle in enumerate(
        np.linspace(0.0, 2.0 * np.pi, args.injections, endpoint=False), 1
    ):
        offset_px = args.radius_px * np.array([np.cos(angle), np.sin(angle)])
        offset_mas = mean_jacobian @ offset_px * MAS_PER_DEGREE
        recovered = build_full_stack_detection(
            products,
            config,
            maximum_offset_px=args.search_radius_px,
            stack_half_size_px=stack_half,
            search_center_offset_mas=offset_mas,
            injection_offset_mas=offset_mas,
            injection_flux=injection_flux,
        )
        error = recovered.offset_mas - offset_mas
        covariance = recovered.covariance_mas2
        mahalanobis2 = float(error @ np.linalg.pinv(covariance) @ error)
        rows.append(
            {
                "injection": injection_id,
                "angle_deg": np.rad2deg(angle),
                "injected_xi_mas": offset_mas[0],
                "injected_eta_mas": offset_mas[1],
                "recovered_xi_mas": recovered.offset_mas[0],
                "recovered_eta_mas": recovered.offset_mas[1],
                "error_xi_mas": error[0],
                "error_eta_mas": error[1],
                "error_2d_mas": np.hypot(*error),
                "formal_error_xi_mas": np.sqrt(covariance[0, 0]),
                "formal_error_eta_mas": np.sqrt(covariance[1, 1]),
                "mahalanobis2": mahalanobis2,
                "injected_flux": injection_flux,
                "recovered_flux": recovered.flux,
                "flux_ratio": recovered.flux / injection_flux,
                "fitted_flux_snr": recovered.fitted_snr,
                "at_search_boundary": recovered.at_search_boundary,
                "recovered": bool(
                    recovered.fitted_snr >= args.minimum_snr
                    and not recovered.at_search_boundary
                ),
            }
        )

    table = pd.DataFrame(rows)
    errors = table[["error_xi_mas", "error_eta_mas"]].to_numpy(float)
    summary = {
        "series_id": str(config.series["id"]),
        "frames": len(accepted),
        "injections": len(table),
        "stack_half_size_px": stack_half,
        "radius_px": args.radius_px,
        "flux_scale": args.flux_scale,
        "injected_flux": injection_flux,
        "recovery_fraction": float(table.recovered.mean()),
        "bias_xi_mas": float(errors[:, 0].mean()),
        "bias_eta_mas": float(errors[:, 1].mean()),
        "rms_xi_mas": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "rms_eta_mas": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "rms_2d_mas": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "median_flux_ratio": float(table.flux_ratio.median()),
        "median_fitted_flux_snr": float(table.fitted_flux_snr.median()),
        "fraction_inside_formal_95pct_ellipse": float((table.mahalanobis2 <= 5.991).mean()),
    }
    series_id = str(config.series["id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / f"injection_recovery_{series_id}.csv"
    summary_path = args.output_dir / f"injection_recovery_{series_id}_summary.json"
    table.to_csv(table_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"Saved {table_path} and {summary_path}")


if __name__ == "__main__":
    main()
