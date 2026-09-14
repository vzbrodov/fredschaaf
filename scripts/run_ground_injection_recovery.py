#!/usr/bin/env python3
"""Inject moving sources into real frames and measure recovery bias."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fredschaaf-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.wcs import FITSFixedWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.detection import build_full_stack_detection
from fredschaaf_astrometry.ground import MAS_PER_DEGREE
from fredschaaf_astrometry.ground_pipeline import reduce_series
from fredschaaf_astrometry.series import load_series_config


def plot_recovery_summary(
    summary_table: pd.DataFrame,
    *,
    series_id: str,
    minimum_snr: float,
    output: Path,
) -> None:
    """Plot completeness, positional RMS and flux recovery."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    x = summary_table.median_fitted_flux_snr
    axes[0].errorbar(
        x,
        summary_table.recovery_fraction,
        yerr=np.vstack(
            (
                summary_table.recovery_fraction - summary_table.recovery_fraction_wilson_low,
                summary_table.recovery_fraction_wilson_high - summary_table.recovery_fraction,
            )
        ),
        fmt="o-",
        capsize=3,
    )
    axes[0].axvline(minimum_snr, color="0.5", ls="--", lw=1)
    axes[0].set(xlabel="median measured fitted S/N", ylabel="recovery fraction", ylim=(-0.05, 1.05))
    axes[1].plot(x, summary_table.rms_2d_mas, "o-")
    axes[1].set(xlabel="median measured fitted S/N", ylabel="position RMS, mas")
    axes[2].plot(x, summary_table.median_flux_ratio, "o-")
    axes[2].axhline(1.0, color="0.5", ls="--", lw=1)
    axes[2].set(xlabel="median measured fitted S/N", ylabel="median recovered / injected flux")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
    figure.suptitle(f"{series_id}: moving-source injection recovery")
    figure.savefig(output, dpi=170)
    plt.close(figure)


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
    parser.add_argument(
        "--flux-scales",
        type=float,
        nargs="+",
        help="flux grid relative to the measured asteroid; overrides --flux-scale",
    )
    parser.add_argument(
        "--expected-snrs",
        type=float,
        nargs="+",
        help="target expected S/N grid; also retains the measured-object flux",
    )
    parser.add_argument("--search-radius-px", type=float, default=1.5)
    parser.add_argument("--minimum-snr", type=float, default=8.0)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw the PNG from an existing summary CSV without rerunning injections",
    )
    args = parser.parse_args()
    requested_scales = args.flux_scales or [args.flux_scale]
    if args.injections < 4 or args.radius_px <= 0:
        raise SystemExit("Need at least four injections and positive radius/flux scale")

    config = load_series_config(args.config)
    series_id = str(config.series["id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / f"injection_recovery_{series_id}.csv"
    summary_path = args.output_dir / f"injection_recovery_{series_id}_summary.json"
    summary_table_path = args.output_dir / f"injection_recovery_{series_id}_summary.csv"
    figure_path = args.output_dir / f"injection_recovery_{series_id}.png"
    if args.plot_only:
        plot_recovery_summary(
            pd.read_csv(summary_table_path),
            series_id=series_id,
            minimum_snr=args.minimum_snr,
            output=figure_path,
        )
        print(f"Saved {figure_path}")
        return
    products = reduce_series(config)
    psf_half = int(config.reduction["psf_half_size_px"])
    stack_half = max(
        int(config.reduction["stack_half_size_px"]),
        int(np.ceil(args.radius_px + psf_half + args.search_radius_px + 3)),
    )
    real = build_full_stack_detection(
        products, config, maximum_offset_px=8.0, stack_half_size_px=stack_half
    )
    if args.expected_snrs is None:
        flux_scales = np.asarray(requested_scales, dtype=float)
    else:
        flux_scales = np.r_[
            np.asarray(args.expected_snrs, dtype=float) / real.fitted_snr,
            1.0,
        ]
    if np.any(~np.isfinite(flux_scales)) or np.any(flux_scales <= 0):
        raise SystemExit("Flux scales and expected S/N values must be finite and positive")
    flux_scales = np.unique(flux_scales)
    accepted = real.frame_indices
    mean_jacobian = np.mean(
        [products.frame_models[index].coefficients[1:].T for index in accepted], axis=0
    )

    rows = []
    angles = np.linspace(0.0, 2.0 * np.pi, args.injections, endpoint=False)
    for flux_scale in flux_scales:
        injection_flux = float(real.flux * flux_scale)
        for injection_id, angle in enumerate(angles, 1):
            offset_px = args.radius_px * np.array([np.cos(angle), np.sin(angle)])
            offset_mas = mean_jacobian @ offset_px * MAS_PER_DEGREE
            common = {
                "injection": injection_id,
                "angle_deg": np.rad2deg(angle),
                "flux_scale": flux_scale,
                "expected_snr_scaled": real.fitted_snr * flux_scale,
                "injected_xi_mas": offset_mas[0],
                "injected_eta_mas": offset_mas[1],
                "injected_flux": injection_flux,
            }
            try:
                fitted = build_full_stack_detection(
                    products,
                    config,
                    maximum_offset_px=args.search_radius_px,
                    stack_half_size_px=stack_half,
                    search_center_offset_mas=offset_mas,
                    injection_offset_mas=offset_mas,
                    injection_flux=injection_flux,
                )
            except ValueError as error:
                rows.append(
                    {
                        **common,
                        "fit_error": str(error),
                        "recovered": False,
                    }
                )
                continue
            error = fitted.offset_mas - offset_mas
            covariance = fitted.covariance_mas2
            mahalanobis2 = float(error @ np.linalg.pinv(covariance) @ error)
            rows.append(
                {
                    **common,
                    "recovered_xi_mas": fitted.offset_mas[0],
                    "recovered_eta_mas": fitted.offset_mas[1],
                    "error_xi_mas": error[0],
                    "error_eta_mas": error[1],
                    "error_2d_mas": np.hypot(*error),
                    "formal_error_xi_mas": np.sqrt(covariance[0, 0]),
                    "formal_error_eta_mas": np.sqrt(covariance[1, 1]),
                    "mahalanobis2": mahalanobis2,
                    "recovered_flux": fitted.flux,
                    "flux_ratio": fitted.flux / injection_flux,
                    "fitted_flux_snr": fitted.fitted_snr,
                    "at_search_boundary": fitted.at_search_boundary,
                    "recovered": bool(
                        fitted.fitted_snr >= args.minimum_snr
                        and not fitted.at_search_boundary
                    ),
                }
            )

    table = pd.DataFrame(rows)
    summary_rows = []
    for flux_scale, group in table.groupby("flux_scale", sort=True):
        valid = group.dropna(subset=["error_xi_mas", "error_eta_mas"])
        errors = valid[["error_xi_mas", "error_eta_mas"]].to_numpy(float)
        recovered_count = int(group.recovered.sum())
        count = len(group)
        fraction = recovered_count / count
        denominator = 1.0 + 1.96**2 / count
        center = (fraction + 1.96**2 / (2 * count)) / denominator
        half_width = (
            1.96
            * np.sqrt(fraction * (1 - fraction) / count + 1.96**2 / (4 * count**2))
            / denominator
        )
        summary_rows.append(
            {
                "flux_scale": flux_scale,
                "expected_snr_scaled": real.fitted_snr * flux_scale,
                "injections": count,
                "successful_recoveries": recovered_count,
                "recovery_fraction": fraction,
                "recovery_fraction_wilson_low": center - half_width,
                "recovery_fraction_wilson_high": center + half_width,
                "bias_xi_mas": float(errors[:, 0].mean()) if len(valid) else np.nan,
                "bias_eta_mas": float(errors[:, 1].mean()) if len(valid) else np.nan,
                "rms_xi_mas": float(np.sqrt(np.mean(errors[:, 0] ** 2))) if len(valid) else np.nan,
                "rms_eta_mas": float(np.sqrt(np.mean(errors[:, 1] ** 2))) if len(valid) else np.nan,
                "rms_2d_mas": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))) if len(valid) else np.nan,
                "median_flux_ratio": float(valid.flux_ratio.median()) if len(valid) else np.nan,
                "median_fitted_flux_snr": float(valid.fitted_flux_snr.median()) if len(valid) else np.nan,
                "snr_transfer_ratio": float(
                    valid.fitted_flux_snr.median() / (real.fitted_snr * flux_scale)
                ) if len(valid) else np.nan,
                "fraction_inside_formal_95pct_ellipse": float(
                    (valid.mahalanobis2 <= 5.991).mean()
                ) if len(valid) else np.nan,
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    summary = {
        "series_id": str(config.series["id"]),
        "frames": len(accepted),
        "injections_per_flux": args.injections,
        "total_injections": len(table),
        "stack_half_size_px": stack_half,
        "radius_px": args.radius_px,
        "reference_flux": float(real.flux),
        "reference_fitted_snr": float(real.fitted_snr),
        "minimum_detection_snr": args.minimum_snr,
        "by_flux_scale": summary_table.to_dict(orient="records"),
    }
    table.to_csv(table_path, index=False)
    summary_table.to_csv(summary_table_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_recovery_summary(
        summary_table,
        series_id=series_id,
        minimum_snr=args.minimum_snr,
        output=figure_path,
    )
    print(summary_table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"Saved {table_path}, {summary_table_path}, {summary_path}, and {figure_path}")


if __name__ == "__main__":
    main()
