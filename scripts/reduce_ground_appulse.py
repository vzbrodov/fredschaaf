#!/usr/bin/env python3
"""Run direct asteroid--Gaia-star differential astrometry on one series."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.wcs import FITSFixedWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.appulse import fit_appulse_trajectory
from fredschaaf_astrometry.ground import (
    MAS_PER_DEGREE,
    inverse_tangent_plane,
    linear_ephemeris_coordinates,
    tangent_plane,
)
from fredschaaf_astrometry.ground_pipeline import reduce_series
from fredschaaf_astrometry.series import load_series_config


def main() -> None:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    warnings.filterwarnings("ignore", message="The fit may not have converged.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/20250903_R.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--event-source-id",
        action="append",
        type=int,
        help="Gaia event star; repeat to run independent reference-star checks",
    )
    parser.add_argument(
        "--skip-group-psf-comparison",
        action="store_true",
        help="Skip the diagnostic primary-star refit with the older group PSF",
    )
    parser.add_argument(
        "--skip-jackknife",
        action="store_true",
        help="Skip leave-one-stack-out fits for a quick low-S/N diagnostic",
    )
    args = parser.parse_args()
    config = load_series_config(args.config)
    source_ids = args.event_source_id or [
        int(config.reduction["appulse_source_id"]),
        *map(int, config.reduction.get("appulse_check_source_ids", [])),
    ]
    catalog_path = (config.path.parent / str(config.reduction["catalog"])).resolve()
    catalog = pd.read_csv(catalog_path).dropna(
        subset=["ra", "dec", "pmra", "pmdec", "ref_epoch"]
    )
    products = reduce_series(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series_id = str(config.series["id"])
    diagnostics_path = args.output_dir / f"appulse_frames_{series_id}.csv"
    pseudo_path = args.output_dir / f"appulse_pseudo_stars_{series_id}.csv"
    jackknife_path = args.output_dir / f"appulse_jackknife_{series_id}.csv"
    systematics_path = args.output_dir / f"appulse_systematics_{series_id}.csv"
    psf_comparison_path = args.output_dir / f"appulse_psf_comparison_{series_id}.csv"
    summary_path = args.output_dir / f"appulse_result_{series_id}.csv"
    ground_path = args.output_dir / f"ground_appulse_series_{series_id}.csv"
    summaries = []
    diagnostics = []
    pseudo_stars = []
    jackknives = []
    systematics = []
    for source_id in source_ids:
        fit = fit_appulse_trajectory(
            products,
            config,
            catalog,
            event_source_id=source_id,
            psf_mode="frame",
            compute_jackknife=not args.skip_jackknife,
        )
        frame_table = fit.frame_diagnostics.copy()
        frame_table.insert(0, "event_source_id", source_id)
        diagnostics.append(frame_table)
        pseudo_table = fit.pseudo_stars.copy()
        pseudo_table.insert(0, "event_source_id", source_id)
        pseudo_stars.append(pseudo_table)
        jackknife_table = fit.jackknife.copy()
        jackknife_table.insert(0, "event_source_id", source_id)
        jackknives.append(jackknife_table)
        systematics_table = fit.systematics.copy()
        systematics_table.insert(0, "event_source_id", source_id)
        systematics.append(systematics_table)
        covariance = fit.formal_covariance_mas2
        summaries.append({
            "series_id": series_id,
            "central_utc": Time(fit.reference_jd, format="jd", scale="utc").isot,
            "event_source_id": fit.event_source_id,
            "event_separation_arcsec": fit.event_separation_arcsec,
            "oc_xi_mas": fit.position_mas[0],
            "oc_eta_mas": fit.position_mas[1],
            "velocity_correction_xi_mas_per_minute": fit.velocity_mas_per_minute[0],
            "velocity_correction_eta_mas_per_minute": fit.velocity_mas_per_minute[1],
            "formal_error_xi_mas": np.sqrt(covariance[0, 0]),
            "formal_error_eta_mas": np.sqrt(covariance[1, 1]),
            "formal_cov_xi_eta_mas2": covariance[0, 1],
            "frames": len(fit.frame_diagnostics),
            "median_frame_fitted_flux_snr": fit.frame_diagnostics.fitted_flux_snr.median(),
            "combined_fitted_flux_snr": np.sqrt(
                np.sum(np.maximum(fit.frame_diagnostics.fitted_flux_snr, 0) ** 2)
            ),
            "boundary_frames": int(fit.frame_diagnostics.at_offset_boundary.sum()),
            "frame_psf_frames": int((fit.frame_diagnostics.psf_scope == "frame").sum()),
            "group_psf_fallback_frames": int(
                (fit.frame_diagnostics.psf_scope == "group").sum()
            ),
            "jackknife_scatter_xi_mas": (
                fit.jackknife.position_xi_mas.std(ddof=1) if len(fit.jackknife) else np.nan
            ),
            "jackknife_scatter_eta_mas": (
                fit.jackknife.position_eta_mas.std(ddof=1) if len(fit.jackknife) else np.nan
            ),
        })
    summary = pd.DataFrame(summaries)
    comparison_rows = []
    if not args.skip_group_psf_comparison:
        group_fit = fit_appulse_trajectory(
            products,
            config,
            catalog,
            event_source_id=source_ids[0],
            psf_mode="group",
            compute_jackknife=False,
        )
        frame_row = summary.loc[summary.event_source_id == source_ids[0]].iloc[0]
        comparison_rows = [
            {
                "event_source_id": source_ids[0],
                "frame_psf_xi_mas": frame_row.oc_xi_mas,
                "frame_psf_eta_mas": frame_row.oc_eta_mas,
                "group_psf_xi_mas": group_fit.position_mas[0],
                "group_psf_eta_mas": group_fit.position_mas[1],
                "difference_xi_mas": frame_row.oc_xi_mas - group_fit.position_mas[0],
                "difference_eta_mas": frame_row.oc_eta_mas - group_fit.position_mas[1],
                "difference_2d_mas": float(
                    np.hypot(*(frame_row[["oc_xi_mas", "oc_eta_mas"]].to_numpy(float) - group_fit.position_mas))
                ),
            }
        ]
    primary = summary.iloc[0]
    psf_model_delta = np.array([np.nan, np.nan])
    if comparison_rows:
        psf_model_delta = np.array(
            [
                comparison_rows[0]["difference_xi_mas"],
                comparison_rows[0]["difference_eta_mas"],
            ]
        )
    formal_covariance = np.array(
        [
            [primary.formal_error_xi_mas**2, primary.formal_cov_xi_eta_mas2],
            [primary.formal_cov_xi_eta_mas2, primary.formal_error_eta_mas**2],
        ]
    )
    covariance = formal_covariance.copy()
    if np.all(np.isfinite(psf_model_delta)):
        covariance += np.outer(psf_model_delta, psf_model_delta)
    central_time = Time(primary.central_utc, scale="utc")
    ephemeris_ra, ephemeris_dec = linear_ephemeris_coordinates(
        central_time,
        reference_time=Time(str(config.ephemeris["reference_utc"]), scale="utc"),
        reference_ra_deg=float(config.ephemeris["ra_deg"]),
        reference_dec_deg=float(config.ephemeris["dec_deg"]),
        ra_rate_arcsec_per_hour=float(config.ephemeris["ra_rate_arcsec_per_hour"]),
        dec_rate_arcsec_per_hour=float(config.ephemeris["dec_rate_arcsec_per_hour"]),
    )
    measured_tangent = tangent_plane(
        [ephemeris_ra], [ephemeris_dec],
        origin_ra_deg=float(config.ephemeris["ra_deg"]),
        origin_dec_deg=float(config.ephemeris["dec_deg"]),
    )[0] + primary[["oc_xi_mas", "oc_eta_mas"]].to_numpy(float) / MAS_PER_DEGREE
    measured_ra, measured_dec = inverse_tangent_plane(
        *measured_tangent,
        origin_ra_deg=float(config.ephemeris["ra_deg"]),
        origin_dec_deg=float(config.ephemeris["dec_deg"]),
    )
    reference_difference = (
        float(np.hypot(
            summary.iloc[1].oc_xi_mas - primary.oc_xi_mas,
            summary.iloc[1].oc_eta_mas - primary.oc_eta_mas,
        )) if len(summary) > 1 else np.nan
    )
    psf_model_difference = float(np.hypot(*psf_model_delta))
    psf_model_ok = (
        not np.isfinite(psf_model_difference)
        or psf_model_difference
        <= float(config.quality.get("max_appulse_psf_model_difference_mas", 50.0))
    )
    ground = pd.DataFrame([{
        "source": "ground_appulse",
        "series_id": f"{series_id}_appulse",
        "observatory_code": str(config.series["observatory_code"]),
        "filter": str(config.series["filter"]),
        "central_utc": primary.central_utc,
        "jd_utc": central_time.jd,
        "ra_deg": measured_ra,
        "dec_deg": measured_dec,
        "oc_xi_mas": primary.oc_xi_mas,
        "oc_eta_mas": primary.oc_eta_mas,
        "cov_xi_xi_mas2": covariance[0, 0],
        "cov_xi_eta_mas2": covariance[0, 1],
        "cov_eta_eta_mas2": covariance[1, 1],
        "ephemeris_id": str(config.ephemeris["id"]),
        "covariance_method": "joint-pixel-forward-fit-formal+psf-model-spread",
        "event_source_id": int(primary.event_source_id),
        "event_separation_arcsec": primary.event_separation_arcsec,
        "reference_check_difference_mas": reference_difference,
        "combined_fitted_flux_snr": primary.combined_fitted_flux_snr,
        "psf_model_difference_mas": psf_model_difference,
        "quality_ok": bool(
            primary.combined_fitted_flux_snr >= float(config.quality["min_median_stack_snr"])
            and primary.boundary_frames == 0
            and (not np.isfinite(reference_difference) or reference_difference <= 50.0)
            and psf_model_ok
        ),
        "quality_flags": "ok" if (
            primary.combined_fitted_flux_snr >= float(config.quality["min_median_stack_snr"])
            and primary.boundary_frames == 0
            and (not np.isfinite(reference_difference) or reference_difference <= 50.0)
            and psf_model_ok
        ) else "appulse_validation_failed",
    }])
    pd.concat(diagnostics, ignore_index=True).to_csv(diagnostics_path, index=False)
    pd.concat(pseudo_stars, ignore_index=True).to_csv(pseudo_path, index=False)
    pd.concat(jackknives, ignore_index=True).to_csv(jackknife_path, index=False)
    pd.concat(systematics, ignore_index=True).to_csv(systematics_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(psf_comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    ground.to_csv(ground_path, index=False)
    print(summary.to_string(index=False))
    print(
        f"Saved {summary_path}, {diagnostics_path}, {pseudo_path}, "
        f"{jackknife_path}, {systematics_path}, {psf_comparison_path}, and {ground_path}"
    )


if __name__ == "__main__":
    main()
