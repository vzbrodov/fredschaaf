#!/usr/bin/env python3
"""Fit adaptive motion-compensated substacks and a linear series trajectory."""

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

from fredschaaf_astrometry.detection import (
    build_full_stack_detection,
    fit_stack_flux_at_offset,
)
from fredschaaf_astrometry.ground import MAS_PER_DEGREE
from fredschaaf_astrometry.ground_pipeline import reduce_series
from fredschaaf_astrometry.series import load_series_config


def adaptive_groups(indices, full_snr, target_snr, minimum_frames, maximum_groups):
    """Split chronologically while keeping the predicted S/N near target."""
    possible = max(1, len(indices) // minimum_frames)
    by_signal = max(1, int((max(full_snr, 0.0) / target_snr) ** 2))
    count = min(possible, by_signal, maximum_groups)
    return [tuple(map(int, group)) for group in np.array_split(indices, count)]


def split_time_runs(indices, frame_models, maximum_gap_minutes):
    """Never let a substack bridge a long interruption in observations."""
    runs = []
    start = 0
    for position in range(1, len(indices)):
        gap = (
            frame_models[indices[position]].time
            - frame_models[indices[position - 1]].time
        ).to_value("min")
        if gap > maximum_gap_minutes:
            runs.append(indices[start:position])
            start = position
    runs.append(indices[start:])
    return runs


def fit_linear_trajectory(table: pd.DataFrame, reference_jd: float):
    """Generalized least-squares fit of xi/eta offsets and velocity corrections."""
    good = table.detection_ok.to_numpy(bool)
    used = table.loc[good].reset_index(drop=True)
    if len(used) < 2:
        return None
    normal = np.zeros((4, 4))
    rhs = np.zeros(4)
    blocks = []
    for row in used.itertuples():
        dt_hours = (row.reference_jd - reference_jd) * 24.0
        design = np.array([[1.0, 0.0, dt_hours, 0.0], [0.0, 1.0, 0.0, dt_hours]])
        covariance = np.array(
            [[row.cov_xi_xi_mas2, row.cov_xi_eta_mas2],
             [row.cov_xi_eta_mas2, row.cov_eta_eta_mas2]]
        )
        weight = np.linalg.pinv(covariance)
        value = np.array([row.offset_xi_mas, row.offset_eta_mas])
        normal += design.T @ weight @ design
        rhs += design.T @ weight @ value
        blocks.append((design, value, weight))
    covariance = np.linalg.pinv(normal)
    parameters = covariance @ rhs
    chi2 = sum(
        float((value - design @ parameters).T @ weight @ (value - design @ parameters))
        for design, value, weight in blocks
    )
    dof = max(1, 2 * len(used) - 4)
    scale = np.sqrt(max(1.0, chi2 / dof))
    residuals = []
    for design, value, _ in blocks:
        residuals.append(value - design @ parameters)
    residuals = np.asarray(residuals)
    return {
        "used_substacks": len(used),
        "reference_utc": Time(reference_jd, format="jd", scale="utc").isot,
        "offset_xi_mas": parameters[0],
        "offset_eta_mas": parameters[1],
        "velocity_correction_xi_mas_per_hour": parameters[2],
        "velocity_correction_eta_mas_per_hour": parameters[3],
        "error_xi_mas": np.sqrt(covariance[0, 0]),
        "error_eta_mas": np.sqrt(covariance[1, 1]),
        "error_velocity_xi_mas_per_hour": np.sqrt(covariance[2, 2]),
        "error_velocity_eta_mas_per_hour": np.sqrt(covariance[3, 3]),
        "scatter_scaled_error_xi_mas": np.sqrt(covariance[0, 0]) * scale,
        "scatter_scaled_error_eta_mas": np.sqrt(covariance[1, 1]) * scale,
        "scatter_scaled_error_velocity_xi_mas_per_hour": np.sqrt(covariance[2, 2]) * scale,
        "scatter_scaled_error_velocity_eta_mas_per_hour": np.sqrt(covariance[3, 3]) * scale,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": chi2 / dof,
        "residual_rms_xi_mas": np.sqrt(np.mean(residuals[:, 0] ** 2)),
        "residual_rms_eta_mas": np.sqrt(np.mean(residuals[:, 1] ** 2)),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    warnings.filterwarnings("ignore", message="The fit may not have converged.*")
    warnings.filterwarnings("ignore", message="Input data contains invalid values.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--target-snr", type=float, default=15.0)
    parser.add_argument("--minimum-detection-snr", type=float, default=8.0)
    parser.add_argument("--minimum-frames", type=int, default=3)
    parser.add_argument("--maximum-groups", type=int, default=12)
    parser.add_argument("--search-radius-px", type=float, default=1.5)
    parser.add_argument(
        "--maximum-gap-minutes", type=float,
        help="do not bridge longer gaps; defaults to config value or 5 minutes",
    )
    parser.add_argument(
        "--seed-offset-mas", type=float, nargs=2, metavar=("XI", "ETA"),
        help="external fixed search seed; useful for forced H-alpha photometry",
    )
    parser.add_argument(
        "--force-groups", type=int,
        help="diagnostic override; groups may fall below the target S/N",
    )
    args = parser.parse_args()
    if args.target_snr <= 0 or args.minimum_frames < 1 or args.maximum_groups < 1:
        raise SystemExit("S/N and grouping parameters must be positive")

    config = load_series_config(args.config)
    products = reduce_series(config)
    full = build_full_stack_detection(products, config, maximum_offset_px=8.0)
    seed_mas = np.array(
        full.offset_mas if args.seed_offset_mas is None else args.seed_offset_mas,
        float,
    )
    accepted = np.array(full.frame_indices, int)
    full_jacobian = np.mean(
        [products.frame_models[index].coefficients[1:].T for index in accepted], axis=0
    )
    full_seed_px = np.linalg.solve(full_jacobian, seed_mas / MAS_PER_DEGREE)
    full_forced_flux, full_forced_flux_error, _ = fit_stack_flux_at_offset(
        full.image, full.psf, full_seed_px
    )
    maximum_gap = float(
        args.maximum_gap_minutes
        if args.maximum_gap_minutes is not None
        else config.reduction.get("max_inter_frame_gap_minutes", 5.0)
    )
    runs = split_time_runs(accepted, products.frame_models, maximum_gap)
    groups = []
    for run in runs:
        run_snr = full.fitted_snr * np.sqrt(len(run) / len(accepted))
        groups.extend(adaptive_groups(
            run,
            run_snr,
            args.target_snr,
            args.minimum_frames,
            max(1, args.maximum_groups - len(groups)),
        ))
    if args.force_groups is not None:
        count = min(args.force_groups, len(accepted) // args.minimum_frames)
        groups = [tuple(map(int, group)) for group in np.array_split(accepted, max(1, count))]

    rows = []
    for group_id, indices in enumerate(groups, 1):
        result = build_full_stack_detection(
            products,
            config,
            maximum_offset_px=args.search_radius_px,
            frame_indices=indices,
            search_center_offset_mas=seed_mas,
        )
        jacobian = np.mean(
            [products.frame_models[index].coefficients[1:].T for index in indices], axis=0
        )
        seed_px = np.linalg.solve(jacobian, seed_mas / MAS_PER_DEGREE)
        forced_flux, forced_flux_error, _ = fit_stack_flux_at_offset(
            result.image, result.psf, seed_px
        )
        covariance = result.covariance_mas2
        rows.append({
            "group": group_id,
            "first_frame_index": indices[0],
            "last_frame_index": indices[-1],
            "frames": len(indices),
            "reference_jd": result.reference_jd,
            "central_utc": Time(result.reference_jd, format="jd", scale="utc").isot,
            "span_minutes": result.span_minutes,
            "offset_xi_mas": result.offset_mas[0],
            "offset_eta_mas": result.offset_mas[1],
            "cov_xi_xi_mas2": covariance[0, 0],
            "cov_xi_eta_mas2": covariance[0, 1],
            "cov_eta_eta_mas2": covariance[1, 1],
            "formal_error_xi_mas": np.sqrt(covariance[0, 0]),
            "formal_error_eta_mas": np.sqrt(covariance[1, 1]),
            "fitted_flux_snr": result.fitted_snr,
            "forced_flux_snr_at_seed": forced_flux / forced_flux_error,
            "at_search_boundary": result.at_search_boundary,
            "detection_ok": len(indices) >= args.minimum_frames
            and result.fitted_snr >= args.minimum_detection_snr
            and not result.at_search_boundary,
        })

    table = pd.DataFrame(rows)
    trajectory = fit_linear_trajectory(table, full.reference_jd)
    series_id = str(config.series["id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / f"adaptive_substacks_{series_id}.csv"
    summary_path = args.output_dir / f"adaptive_trajectory_{series_id}.csv"
    diagnostic_path = args.output_dir / f"adaptive_full_diagnostic_{series_id}.csv"
    table.to_csv(table_path, index=False)
    pd.DataFrame([] if trajectory is None else [trajectory]).to_csv(summary_path, index=False)
    pd.DataFrame([{
        "series_id": series_id,
        "frames": len(accepted),
        "blind_full_offset_xi_mas": full.offset_mas[0],
        "blind_full_offset_eta_mas": full.offset_mas[1],
        "blind_full_fitted_snr": full.fitted_snr,
        "forced_seed_offset_xi_mas": seed_mas[0],
        "forced_seed_offset_eta_mas": seed_mas[1],
        "forced_full_flux_snr": full_forced_flux / full_forced_flux_error,
        "groups": len(groups),
        "time_runs": len(runs),
        "maximum_gap_minutes": maximum_gap,
    }]).to_csv(diagnostic_path, index=False)
    print(table.to_string(index=False))
    if trajectory is None:
        print("No linear trajectory: fewer than two substacks passed detection QC")
    else:
        print(pd.DataFrame([trajectory]).to_string(index=False))
    print(
        f"Full-stack blind S/N={full.fitted_snr:.2f}; "
        f"forced S/N at seed={full_forced_flux / full_forced_flux_error:.2f}"
    )
    print(f"Saved {table_path}, {summary_path}, and {diagnostic_path}")


if __name__ == "__main__":
    main()
