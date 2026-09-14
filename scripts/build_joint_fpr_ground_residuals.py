#!/usr/bin/env python3
"""Jointly refit the six-parameter orbit to Gaia FPR and ground astrometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import assist
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation
from astropy.table import Table
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.fpr import (
    add_ccd_residuals,
    aggregate_transit_residuals,
    aggregate_transits,
    prepare_ccd_observations,
    tangent_plane_residuals,
)
from fredschaaf_astrometry.orbit import FPR_REFERENCE_JD, propagate_fpr_state


def mpc_217_location() -> EarthLocation:
    longitude = np.deg2rad(77.87114)
    radius = 6378.137 * u.km
    return EarthLocation.from_geocentric(
        radius * 0.730114 * np.cos(longitude),
        radius * 0.730114 * np.sin(longitude),
        radius * 0.681643,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=7065)
    parser.add_argument(
        "--ground",
        type=Path,
        default=Path("outputs/ground_orbit_series.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--planets", type=Path, default=Path("data/assist/linux_p1550p2650.440"))
    parser.add_argument("--asteroids", type=Path, default=Path("data/assist/sb441-n16.bsp"))
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()

    stem = f"gaia_fpr_{args.number}"
    source = Table.read(args.output_dir / f"{stem}_source.ecsv")
    observations = Table.read(args.output_dir / f"{stem}_observations.ecsv")
    ground = pd.read_csv(args.ground)
    if len(source) != 1 or ground.empty:
        raise SystemExit("Expected one FPR source and at least one ground series")
    if "quality_ok" in ground:
        quality = ground.quality_ok.astype(str).str.lower().map({"true": True, "false": False})
        if quality.isna().any() or not quality.all():
            raise SystemExit("All ground rows must pass quality checks")

    frame = observations.to_pandas()
    prepared = prepare_ccd_observations(frame)
    good = prepared.good_for_analysis
    gaia_position = frame[["x_gaia", "y_gaia", "z_gaia"]].to_numpy(float)
    transit_uncertainty = aggregate_transits(prepared).set_index("transit_id")
    transit_ids = transit_uncertainty.index.to_numpy()
    transit_sigma = transit_uncertainty.sigma_al_total_mas.to_numpy(float)
    ground_covariance = np.zeros((2 * len(ground), 2 * len(ground)))
    for index, row in ground.iterrows():
        block = np.array([
            [row.cov_xi_xi_mas2, row.cov_xi_eta_mas2],
            [row.cov_xi_eta_mas2, row.cov_eta_eta_mas2],
        ], dtype=float)
        if np.any(np.linalg.eigvalsh(block) <= 0):
            raise SystemExit(f"Invalid ground covariance for {row.series_id}")
        ground_covariance[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = block
    covariance = np.zeros((len(transit_ids) + 2 * len(ground),) * 2)
    covariance[: len(transit_ids), : len(transit_ids)] = np.diag(transit_sigma**2)
    covariance[len(transit_ids) :, len(transit_ids) :] = ground_covariance
    cholesky = np.linalg.cholesky(covariance)

    ephem = assist.Ephem(str(args.planets), str(args.asteroids))
    location = mpc_217_location()
    ground_times = Time(ground.central_utc.to_list(), scale="utc")
    ground_observers = []
    for epoch in ground_times:
        earth = ephem.get_particle("Earth", epoch.tdb.jd - ephem.jd_ref)
        topocentric = location.get_gcrs_posvel(epoch)[0].xyz.to_value(u.au)
        ground_observers.append(np.array([earth.x, earth.y, earth.z]) + topocentric)
    ground_observers = np.asarray(ground_observers)
    ground_offsets = ground_times.tcb.jd - FPR_REFERENCE_JD
    state_epoch = float(source["epoch_state_vector"][0])
    state = np.asarray(
        json.loads((args.output_dir / f"{stem}_residual_summary.json").read_text())["fitted_state_fpr"],
        dtype=float,
    )

    def evaluate(trial_state):
        gaia_ra, gaia_dec, emission = propagate_fpr_state(
            trial_state,
            state_epoch,
            frame.epoch.to_numpy(float),
            gaia_position,
            planets_path=args.planets,
            asteroids_path=args.asteroids,
        )
        ccd = add_ccd_residuals(prepared, gaia_ra, gaia_dec)
        transit = (
            ccd.loc[good].groupby("transit_id").residual_al_mas.mean()
            .reindex(transit_ids).to_numpy(float)
        )
        model_ra, model_dec, _ = propagate_fpr_state(
            trial_state,
            state_epoch,
            ground_offsets,
            ground_observers,
            planets_path=args.planets,
            asteroids_path=args.asteroids,
            observer_positions_are_tcb=False,
            solar_light_deflection_mode="differential",
        )
        xi, eta = tangent_plane_residuals(
            ground.ra_deg.to_numpy(float),
            ground.dec_deg.to_numpy(float),
            model_ra,
            model_dec,
        )
        ground_residual = np.column_stack((xi, eta)).ravel()
        return np.r_[transit, ground_residual], ccd, emission, model_ra, model_dec

    state_steps = np.array([1e-7, 1e-7, 1e-7, 1e-10, 1e-10, 1e-10])
    initial_state = state.copy()
    initial_residual, _, _, _, _ = evaluate(state)
    for _ in range(args.iterations):
        residual, _, _, _, _ = evaluate(state)
        jacobian = np.empty((len(residual), 6))
        for parameter in range(6):
            perturbed = state.copy()
            perturbed[parameter] += state_steps[parameter]
            perturbed_residual, _, _, _, _ = evaluate(perturbed)
            jacobian[:, parameter] = (perturbed_residual - residual) / state_steps[parameter]
        white_jacobian = np.linalg.solve(cholesky, jacobian)
        white_residual = np.linalg.solve(cholesky, residual)
        correction = np.linalg.lstsq(white_jacobian, -white_residual, rcond=None)[0]
        state += correction

    residual, ccd, emission, model_ra, model_dec = evaluate(state)
    final_jacobian = np.empty((len(residual), 6))
    for parameter in range(6):
        perturbed = state.copy()
        perturbed[parameter] += state_steps[parameter]
        perturbed_residual, _, _, _, _ = evaluate(perturbed)
        final_jacobian[:, parameter] = (
            perturbed_residual - residual
        ) / state_steps[parameter]
    gaia_white_jacobian = final_jacobian[: len(transit_ids)] / transit_sigma[:, None]
    joint_white_jacobian = np.linalg.solve(cholesky, final_jacobian)
    gaia_state_covariance = np.linalg.inv(
        gaia_white_jacobian.T @ gaia_white_jacobian
    )
    joint_state_covariance = np.linalg.inv(
        joint_white_jacobian.T @ joint_white_jacobian
    )
    ground_model_jacobian = -final_jacobian[len(transit_ids) :]
    gaia_prediction_covariance = (
        ground_model_jacobian @ gaia_state_covariance @ ground_model_jacobian.T
    )
    joint_prediction_covariance = (
        ground_model_jacobian @ joint_state_covariance @ ground_model_jacobian.T
    )
    ccd["emission_jd_tdb"] = emission
    transits = aggregate_transit_residuals(ccd)
    ground_result = ground.copy()
    ground_result["model_ra_deg"] = model_ra
    ground_result["model_dec_deg"] = model_dec
    ground_result["oc_xi_mas"] = residual[-2 * len(ground) :: 2]
    ground_result["oc_eta_mas"] = residual[-2 * len(ground) + 1 :: 2]
    ground_result["ephemeris_id"] = "joint-gaia-fpr-ground-assist-de440-de441"
    prefix = args.output_dir / f"joint_fpr_ground_{args.number}"
    ccd.to_csv(f"{prefix}_ccd_residuals.csv", index=False)
    transits.to_csv(f"{prefix}_transit_residuals.csv", index=False)
    ground_result.to_csv(f"{prefix}_ground_residuals.csv", index=False)
    summary = {
        "gaia_transits": len(transit_ids),
        "ground_series": len(ground),
        "iterations": args.iterations,
        "initial_chi2": float(np.sum(np.linalg.solve(cholesky, initial_residual) ** 2)),
        "joint_chi2": float(np.sum(np.linalg.solve(cholesky, residual) ** 2)),
        "initial_state_fpr": initial_state.tolist(),
        "joint_state_fpr": state.tolist(),
        "state_correction_fpr": (state - initial_state).tolist(),
        "ground_epoch_orbit_covariance_from_gaia_mas2": gaia_prediction_covariance.tolist(),
        "ground_epoch_orbit_covariance_joint_mas2": joint_prediction_covariance.tolist(),
        "ground_epoch_orbit_sigma_from_gaia_mas": np.sqrt(
            np.diag(gaia_prediction_covariance)
        ).tolist(),
        "ground_epoch_orbit_sigma_joint_mas": np.sqrt(
            np.diag(joint_prediction_covariance)
        ).tolist(),
        "ground_residuals_mas": ground_result[
            ["series_id", "oc_xi_mas", "oc_eta_mas"]
        ].to_dict(orient="records"),
    }
    Path(f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
