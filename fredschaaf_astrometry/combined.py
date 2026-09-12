"""Build one projected-observation system from Gaia and ground astrometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GAIA_REFERENCE_JD = 2_455_197.5


@dataclass(frozen=True)
class ProjectedAstrometry:
    """Scalar observations and their projections onto east/north motion."""

    time_jd: np.ndarray
    values_mas: np.ndarray
    covariance_mas2: np.ndarray
    projection_east_north: np.ndarray
    index: pd.DataFrame


def combine_gaia_and_ground(
    gaia_transit_residuals: pd.DataFrame,
    ground_series: pd.DataFrame,
    *,
    assume_zero_missing_ground_cross_covariance: bool = False,
) -> ProjectedAstrometry:
    """Combine Gaia AL residuals and ground ξ/η O-C into one GLS system.

    Gaia contributes only its accurately measured AL projection. Each ground
    series contributes a correlated two-coordinate block. This routine assumes
    residuals from both inputs are already relative to compatible trajectory
    models; it does not remove independent per-night offsets.
    """
    gaia_required = {
        "transit_id",
        "epoch_utc_offset_jd",
        "position_angle_scan_deg",
        "residual_al_mas",
        "sigma_al_total_mas",
    }
    ground_required = {
        "series_id",
        "jd_utc",
        "oc_xi_mas",
        "oc_eta_mas",
        "cov_xi_xi_mas2",
        "cov_xi_eta_mas2",
        "cov_eta_eta_mas2",
    }
    missing_gaia = sorted(gaia_required.difference(gaia_transit_residuals.columns))
    missing_ground = sorted(ground_required.difference(ground_series.columns))
    if missing_gaia or missing_ground:
        raise ValueError(
            f"Missing Gaia columns: {missing_gaia}; missing ground columns: {missing_ground}"
        )

    values: list[float] = []
    times: list[float] = []
    projections: list[tuple[float, float]] = []
    blocks: list[np.ndarray] = []
    labels: list[dict[str, str]] = []

    for _, row in gaia_transit_residuals.iterrows():
        theta = np.deg2rad(float(row["position_angle_scan_deg"]))
        values.append(float(row["residual_al_mas"]))
        times.append(GAIA_REFERENCE_JD + float(row["epoch_utc_offset_jd"]))
        projections.append((float(np.sin(theta)), float(np.cos(theta))))
        blocks.append(np.array([[float(row["sigma_al_total_mas"]) ** 2]]))
        labels.append(
            {
                "source": "gaia_fpr",
                "observation_id": str(int(row["transit_id"])),
                "component": "al",
            }
        )

    for _, row in ground_series.iterrows():
        cross = float(row["cov_xi_eta_mas2"])
        if not np.isfinite(cross):
            if not assume_zero_missing_ground_cross_covariance:
                raise ValueError(
                    f"Ground series {row['series_id']} has no ξ/η cross covariance"
                )
            cross = 0.0
        block = np.array(
            [
                [float(row["cov_xi_xi_mas2"]), cross],
                [cross, float(row["cov_eta_eta_mas2"])],
            ]
        )
        if not np.all(np.linalg.eigvalsh(block) > 0):
            raise ValueError(f"Ground covariance is not positive definite: {row['series_id']}")
        values.extend([float(row["oc_xi_mas"]), float(row["oc_eta_mas"])])
        times.extend([float(row["jd_utc"]), float(row["jd_utc"])])
        projections.extend([(1.0, 0.0), (0.0, 1.0)])
        blocks.append(block)
        for component in ("xi", "eta"):
            labels.append(
                {
                    "source": "pulkovo",
                    "observation_id": str(row["series_id"]),
                    "component": component,
                }
            )

    size = sum(len(block) for block in blocks)
    covariance = np.zeros((size, size), dtype=float)
    start = 0
    for block in blocks:
        stop = start + len(block)
        covariance[start:stop, start:stop] = block
        start = stop
    return ProjectedAstrometry(
        time_jd=np.asarray(times),
        values_mas=np.asarray(values),
        covariance_mas2=covariance,
        projection_east_north=np.asarray(projections),
        index=pd.DataFrame(labels),
    )
