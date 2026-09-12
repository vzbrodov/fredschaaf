#!/usr/bin/env python3
"""Compute the Assy-Turgen series O-C relative to the fitted FPR orbit."""

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
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.fpr import tangent_plane_residuals
from fredschaaf_astrometry.orbit import FPR_REFERENCE_JD, propagate_fpr_state


# MPC 217 (Assah / Assy-Turgen): east longitude and parallax constants.
MPC_217_LONGITUDE_DEG = 77.87114
MPC_217_RHO_COS_PHI = 0.730114
MPC_217_RHO_SIN_PHI = 0.681643


def mpc_217_location() -> EarthLocation:
    longitude = np.deg2rad(MPC_217_LONGITUDE_DEG)
    radius = 6378.137 * u.km
    return EarthLocation.from_geocentric(
        radius * MPC_217_RHO_COS_PHI * np.cos(longitude),
        radius * MPC_217_RHO_COS_PHI * np.sin(longitude),
        radius * MPC_217_RHO_SIN_PHI,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=7065)
    parser.add_argument("--input", type=Path, default=Path("outputs/pulkovo_series.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pulkovo_series_fpr.csv"))
    parser.add_argument("--planets", type=Path, default=Path("data/assist/linux_p1550p2650.440"))
    parser.add_argument("--asteroids", type=Path, default=Path("data/assist/sb441-n16.bsp"))
    args = parser.parse_args()

    series = pd.read_csv(args.input)
    summary = json.loads(
        Path(f"outputs/gaia_fpr_{args.number}_residual_summary.json").read_text(encoding="utf-8")
    )
    state = np.asarray(summary["fitted_state_fpr"], dtype=float)
    ephem = assist.Ephem(str(args.planets), str(args.asteroids))
    location = mpc_217_location()
    model_ra = []
    model_dec = []
    for _, row in series.iterrows():
        epoch = Time(row["central_utc"], scale="utc")
        earth = ephem.get_particle("Earth", epoch.tdb.jd - ephem.jd_ref)
        topocentric = location.get_gcrs_posvel(epoch)[0].xyz.to_value(u.au)
        observer = np.array([earth.x, earth.y, earth.z]) + topocentric
        ra, dec, _ = propagate_fpr_state(
            state,
            float(summary["state_epoch_tcb_offset_days"]),
            np.array([epoch.tcb.jd - FPR_REFERENCE_JD]),
            observer[None, :],
            planets_path=args.planets,
            asteroids_path=args.asteroids,
            observer_positions_are_tcb=False,
            # The Gaia-star plate solution removes the local stellar
            # (infinite-distance) deflection, leaving the finite-minus-infinite
            # differential term for the asteroid.
            solar_light_deflection_mode="differential",
        )
        model_ra.append(ra[0])
        model_dec.append(dec[0])
    xi, eta = tangent_plane_residuals(
        series["ra_deg"], series["dec_deg"], model_ra, model_dec
    )
    result = series.copy()
    result["model_ra_deg"] = model_ra
    result["model_dec_deg"] = model_dec
    result["oc_xi_mas"] = xi
    result["oc_eta_mas"] = eta
    result["ephemeris_id"] = "gaia-fpr-refit-assist-de440-de441"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[["series_id", "oc_xi_mas", "oc_eta_mas", "ephemeris_id"]].to_string(index=False))


if __name__ == "__main__":
    main()
