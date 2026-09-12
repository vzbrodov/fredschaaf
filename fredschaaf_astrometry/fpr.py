"""Gaia FPR uncertainty handling at CCD and transit level.

Gaia publishes errors in the local equatorial basis
``(RA*cos(Dec), Dec)``.  Random errors are independent between CCDs, while
the systematic component is common to all CCDs in one transit.  The two
components therefore have to be propagated separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ERROR_COMPONENTS = ("random", "systematic")


def covariance_from_errors(
    east_error_mas: np.ndarray,
    north_error_mas: np.ndarray,
    correlation: np.ndarray,
) -> np.ndarray:
    """Return Nx2x2 covariance matrices in the east/north tangent basis."""
    east = np.asarray(east_error_mas, dtype=float)
    north = np.asarray(north_error_mas, dtype=float)
    rho = np.asarray(correlation, dtype=float)
    covariance = np.empty((east.size, 2, 2), dtype=float)
    covariance[:, 0, 0] = east**2
    covariance[:, 1, 1] = north**2
    covariance[:, 0, 1] = covariance[:, 1, 0] = rho * east * north
    return covariance


def project_covariance(
    covariance: np.ndarray, position_angle_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project covariance onto nominal Gaia AL and AC axes.

    The scan position angle is measured from North towards East, hence
    ``u_AL=(sin(theta), cos(theta))`` in the east/north basis.  The nominal
    perpendicular axis is sufficient for uncertainty diagnostics; the archive
    notes that the aberration-corrected AC direction is not exactly orthogonal.
    """
    matrices = np.asarray(covariance, dtype=float)
    theta = np.deg2rad(np.asarray(position_angle_deg, dtype=float))
    along = np.column_stack((np.sin(theta), np.cos(theta)))
    across = np.column_stack((-np.cos(theta), np.sin(theta)))
    variance_al = np.einsum("ni,nij,nj->n", along, matrices, along)
    variance_ac = np.einsum("ni,nij,nj->n", across, matrices, across)
    return variance_al, variance_ac


def prepare_ccd_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Add quality flags and projected FPR uncertainties to CCD observations."""
    required = {
        "transit_id",
        "epoch_utc",
        "position_angle_scan",
        "astrometric_outcome_ccd",
        "is_rejected",
    }
    for component in ERROR_COMPONENTS:
        required.update(
            {
                f"ra_error_{component}",
                f"dec_error_{component}",
                f"ra_dec_correlation_{component}",
            }
        )
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Missing Gaia FPR columns: {missing}")

    result = observations.copy()
    rejected = result["is_rejected"].fillna(False).astype(bool)
    result["good_for_analysis"] = (
        result["astrometric_outcome_ccd"].astype("Int64").eq(1) & ~rejected
    )

    covariances: dict[str, np.ndarray] = {}
    for component in ERROR_COMPONENTS:
        covariance = covariance_from_errors(
            result[f"ra_error_{component}"],
            result[f"dec_error_{component}"],
            result[f"ra_dec_correlation_{component}"],
        )
        covariances[component] = covariance
        variance_al, variance_ac = project_covariance(
            covariance, result["position_angle_scan"]
        )
        result[f"sigma_al_{component}_mas"] = np.sqrt(
            np.maximum(variance_al, 0)
        )
        result[f"sigma_ac_{component}_mas"] = np.sqrt(
            np.maximum(variance_ac, 0)
        )

    total = covariances["random"] + covariances["systematic"]
    variance_al, variance_ac = project_covariance(
        total, result["position_angle_scan"]
    )
    result["sigma_al_total_ccd_mas"] = np.sqrt(np.maximum(variance_al, 0))
    result["sigma_ac_total_ccd_mas"] = np.sqrt(np.maximum(variance_ac, 0))
    return result


def _matrix_from_row(row: pd.Series, component: str) -> np.ndarray:
    return covariance_from_errors(
        np.array([row[f"ra_error_{component}"]], dtype=float),
        np.array([row[f"dec_error_{component}"]], dtype=float),
        np.array([row[f"ra_dec_correlation_{component}"]], dtype=float),
    )[0]


def _combine_random_covariances(matrices: list[np.ndarray]) -> np.ndarray:
    valid = [
        matrix
        for matrix in matrices
        if np.all(np.isfinite(matrix)) and np.all(np.linalg.eigvalsh(matrix) > 0)
    ]
    if not valid:
        return np.full((2, 2), np.nan)
    information = sum(np.linalg.inv(matrix) for matrix in valid)
    return np.linalg.inv(information)


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Return a wrap-safe mean direction in degrees."""
    angles = np.deg2rad(np.asarray(angles_deg, dtype=float))
    valid = np.isfinite(angles)
    if not np.any(valid):
        return float("nan")
    vector = np.mean(np.exp(1j * angles[valid]))
    return float(np.rad2deg(np.angle(vector)) % 360.0)


def tangent_plane_residuals(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    model_ra_deg: np.ndarray,
    model_dec_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed-minus-model residuals in the east/north basis, mas."""
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    model_ra = np.asarray(model_ra_deg, dtype=float)
    model_dec = np.asarray(model_dec_deg, dtype=float)
    delta_ra = (ra - model_ra + 180.0) % 360.0 - 180.0
    east = delta_ra * np.cos(np.deg2rad(model_dec)) * 3_600_000.0
    north = (dec - model_dec) * 3_600_000.0
    return east, north


def add_ccd_residuals(
    prepared_ccd: pd.DataFrame,
    model_ra_deg: np.ndarray,
    model_dec_deg: np.ndarray,
) -> pd.DataFrame:
    """Attach O-C residuals for an orbit model evaluated at every CCD epoch."""
    if len(prepared_ccd) != len(model_ra_deg) or len(prepared_ccd) != len(model_dec_deg):
        raise ValueError("The orbit model must contain one coordinate per CCD row")
    if not {"ra", "dec"}.issubset(prepared_ccd.columns):
        raise ValueError("Gaia CCD table must contain ra and dec")
    result = prepared_ccd.copy()
    east, north = tangent_plane_residuals(
        result["ra"], result["dec"], model_ra_deg, model_dec_deg
    )
    result["model_ra_deg"] = np.asarray(model_ra_deg, dtype=float)
    result["model_dec_deg"] = np.asarray(model_dec_deg, dtype=float)
    result["residual_east_mas"] = east
    result["residual_north_mas"] = north
    theta = np.deg2rad(result["position_angle_scan"].to_numpy(float))
    result["residual_al_mas"] = east * np.sin(theta) + north * np.cos(theta)
    result["residual_ac_nominal_mas"] = -east * np.cos(theta) + north * np.sin(theta)
    return result


def aggregate_transits(prepared_ccd: pd.DataFrame) -> pd.DataFrame:
    """Propagate CCD covariances to transit-level uncertainty summaries.

    This function deliberately does not average RA/Dec.  A moving object's CCD
    coordinates must first be reduced by an orbital/trajectory model; only the
    resulting residuals may be combined into a transit astrometric point.
    """
    if "good_for_analysis" not in prepared_ccd:
        prepared_ccd = prepare_ccd_observations(prepared_ccd)

    rows: list[dict[str, float | int]] = []
    for transit_id, all_group in prepared_ccd.groupby("transit_id", sort=False):
        group = all_group[all_group["good_for_analysis"]]
        if group.empty:
            continue

        random_covariance = _combine_random_covariances(
            [_matrix_from_row(row, "random") for _, row in group.iterrows()]
        )
        systematic_stack = np.stack(
            [_matrix_from_row(row, "systematic") for _, row in group.iterrows()]
        )
        systematic_covariance = np.nanmedian(systematic_stack, axis=0)
        total_covariance = random_covariance + systematic_covariance
        angle = _circular_mean_deg(group["position_angle_scan"].to_numpy(float))
        variance_al, variance_ac = project_covariance(
            total_covariance[None, :, :], np.array([angle])
        )
        rows.append(
            {
                "transit_id": int(transit_id),
                "epoch_utc_offset_jd": float(np.nanmean(group["epoch_utc"])),
                "position_angle_scan_deg": angle,
                "n_ccd_total": int(len(all_group)),
                "n_ccd_good": int(len(group)),
                "random_cov_east_east_mas2": random_covariance[0, 0],
                "random_cov_east_north_mas2": random_covariance[0, 1],
                "random_cov_north_north_mas2": random_covariance[1, 1],
                "systematic_cov_east_east_mas2": systematic_covariance[0, 0],
                "systematic_cov_east_north_mas2": systematic_covariance[0, 1],
                "systematic_cov_north_north_mas2": systematic_covariance[1, 1],
                "total_cov_east_east_mas2": total_covariance[0, 0],
                "total_cov_east_north_mas2": total_covariance[0, 1],
                "total_cov_north_north_mas2": total_covariance[1, 1],
                "sigma_al_total_mas": float(np.sqrt(max(variance_al[0], 0))),
                "sigma_ac_total_mas": float(np.sqrt(max(variance_ac[0], 0))),
            }
        )
    return pd.DataFrame(rows).sort_values("epoch_utc_offset_jd").reset_index(drop=True)


def aggregate_transit_residuals(prepared_ccd: pd.DataFrame) -> pd.DataFrame:
    """Combine model-reduced CCD positions into transit O-C measurements.

    Random CCD covariances determine the weights. The systematic covariance,
    common within a transit, is added only after averaging. Orbit coordinates
    therefore have to be evaluated at every CCD epoch beforehand.
    """
    required = {"residual_east_mas", "residual_north_mas"}
    missing = sorted(required.difference(prepared_ccd.columns))
    if missing:
        raise ValueError(f"Missing model-reduced columns: {missing}")
    if "good_for_analysis" not in prepared_ccd:
        prepared_ccd = prepare_ccd_observations(prepared_ccd)

    uncertainty = aggregate_transits(prepared_ccd).set_index("transit_id")
    residual_rows: list[dict[str, float | int]] = []
    for transit_id, all_group in prepared_ccd.groupby("transit_id", sort=False):
        group = all_group[all_group["good_for_analysis"]]
        if group.empty:
            continue
        information = np.zeros((2, 2), dtype=float)
        weighted_sum = np.zeros(2, dtype=float)
        for _, row in group.iterrows():
            covariance = _matrix_from_row(row, "random")
            residual = row[["residual_east_mas", "residual_north_mas"]].to_numpy(float)
            if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(covariance)):
                continue
            try:
                weight = np.linalg.inv(covariance)
            except np.linalg.LinAlgError:
                continue
            information += weight
            weighted_sum += weight @ residual
        if np.linalg.matrix_rank(information) < 2:
            continue
        residual = np.linalg.solve(information, weighted_sum)
        angle = float(uncertainty.loc[int(transit_id), "position_angle_scan_deg"])
        theta = np.deg2rad(angle)
        residual_rows.append(
            {
                "transit_id": int(transit_id),
                "residual_east_mas": residual[0],
                "residual_north_mas": residual[1],
                "residual_al_mas": residual[0] * np.sin(theta) + residual[1] * np.cos(theta),
                "residual_ac_nominal_mas": -residual[0] * np.cos(theta) + residual[1] * np.sin(theta),
            }
        )
    residual_table = pd.DataFrame(residual_rows).set_index("transit_id")
    return uncertainty.join(residual_table, how="inner").reset_index().sort_values(
        "epoch_utc_offset_jd"
    ).reset_index(drop=True)
