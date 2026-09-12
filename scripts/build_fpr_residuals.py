#!/usr/bin/env python3
"""Propagate the FPR orbit and build CCD/transit O-C tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.fpr import (
    add_ccd_residuals,
    aggregate_transit_residuals,
    aggregate_transits,
    prepare_ccd_observations,
)
from fredschaaf_astrometry.orbit import propagate_fpr_state


def robust_sigma(values) -> float:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - median)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=7065)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--planets", type=Path,
        default=Path("data/assist/linux_p1550p2650.440"),
    )
    parser.add_argument(
        "--asteroids", type=Path,
        default=Path("data/assist/sb441-n16.bsp"),
    )
    parser.add_argument("--refit-iterations", type=int, default=1)
    args = parser.parse_args()

    stem = f"gaia_fpr_{args.number}"
    source = Table.read(args.output_dir / f"{stem}_source.ecsv")
    observations = Table.read(args.output_dir / f"{stem}_observations.ecsv")
    if len(source) != 1:
        raise RuntimeError("Expected exactly one FPR source row")
    frame = observations.to_pandas()
    gaia_position = frame[["x_gaia", "y_gaia", "z_gaia"]].to_numpy(float)
    prepared = prepare_ccd_observations(frame)
    good = prepared["good_for_analysis"]
    # The published state was fitted with Gaia's own force model. Refit a tiny
    # six-parameter correction so that the ASSIST trajectory is the post-fit
    # reference orbit rather than leaving force-model differences in O-C.
    state_published = np.array(source["h_state_vector"][0], dtype=float, copy=True)
    state_fitted = state_published.copy()
    state_epoch = float(source["epoch_state_vector"][0])
    uncertainty = aggregate_transits(prepared).set_index("transit_id")
    transit_ids = uncertainty.index.to_numpy()
    transit_sigma = uncertainty["sigma_al_total_mas"].to_numpy(float)

    def evaluate(state):
        ra, dec, emission = propagate_fpr_state(
            state, state_epoch, frame["epoch"].to_numpy(float), gaia_position,
            planets_path=args.planets, asteroids_path=args.asteroids,
        )
        table = add_ccd_residuals(prepared, ra, dec)
        means = (
            table.loc[good]
            .groupby("transit_id")["residual_al_mas"]
            .mean()
            .reindex(transit_ids)
            .to_numpy(float)
        )
        return means, table, emission

    state_steps = np.array([1e-7, 1e-7, 1e-7, 1e-10, 1e-10, 1e-10])
    initial_means, _, _ = evaluate(state_fitted)
    for _ in range(args.refit_iterations):
        means, _, _ = evaluate(state_fitted)
        jacobian = np.empty((len(means), 6), dtype=float)
        for parameter in range(6):
            perturbed = state_fitted.copy()
            perturbed[parameter] += state_steps[parameter]
            perturbed_means, _, _ = evaluate(perturbed)
            jacobian[:, parameter] = perturbed_means - means
        correction_in_steps = np.linalg.lstsq(
            jacobian / transit_sigma[:, None],
            -means / transit_sigma,
            rcond=None,
        )[0]
        state_fitted += state_steps * correction_in_steps

    fitted_means, residuals, emission_tdb = evaluate(state_fitted)
    residuals["emission_jd_tdb"] = emission_tdb
    transits = aggregate_transit_residuals(residuals)
    residuals.to_csv(args.output_dir / f"{stem}_ccd_residuals.csv", index=False)
    transits.to_csv(args.output_dir / f"{stem}_transit_residuals.csv", index=False)

    summary = {
        "good_ccd": int(good.sum()),
        "good_transits": len(transits),
        "median_ccd_al_residual_mas": float(residuals.loc[good, "residual_al_mas"].median()),
        "robust_sigma_ccd_al_residual_mas": robust_sigma(residuals.loc[good, "residual_al_mas"]),
        "median_transit_al_residual_mas": float(transits["residual_al_mas"].median()),
        "robust_sigma_transit_al_residual_mas": robust_sigma(transits["residual_al_mas"]),
        "robust_sigma_transit_al_before_refit_mas": robust_sigma(initial_means),
        "assist_state_refit_iterations": args.refit_iterations,
        "published_state_fpr": state_published.tolist(),
        "state_epoch_tcb_offset_days": state_epoch,
        "fitted_state_fpr": state_fitted.tolist(),
        "state_correction_fpr": (state_fitted - state_published).tolist(),
        "solar_light_deflection_applied": True,
    }
    path = args.output_dir / f"{stem}_residual_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
