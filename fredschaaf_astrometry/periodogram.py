"""Covariance-aware period scan for projected two-dimensional astrometry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _least_squares_chi2(white_y: np.ndarray, white_design: np.ndarray):
    coefficients, _, rank, _ = np.linalg.lstsq(white_design, white_y, rcond=None)
    if rank < white_design.shape[1]:
        return float("nan"), coefficients
    residual = white_y - white_design @ coefficients
    return float(residual @ residual), coefficients


def scan_vector_periods(
    time_jd: np.ndarray,
    values_mas: np.ndarray,
    covariance_mas2: np.ndarray,
    projection_east_north: np.ndarray,
    periods_hours: np.ndarray,
    *,
    nuisance_design: np.ndarray | None = None,
    include_vector_offset: bool = True,
    include_vector_trend: bool = False,
) -> pd.DataFrame:
    """Scan a sinusoidal sky-plane displacement through arbitrary projections.

    Each scalar datum measures ``p_east*east + p_north*north``. Gaia AL and
    ground east/north data can therefore share one fit without assigning Gaia
    AC the precision of AL. ``power`` is a diagnostic chi-square improvement;
    its significance needs bootstrap or injection/recovery calibration.
    """
    time = np.asarray(time_jd, dtype=float)
    values = np.asarray(values_mas, dtype=float)
    covariance = np.asarray(covariance_mas2, dtype=float)
    projection = np.asarray(projection_east_north, dtype=float)
    periods = np.asarray(periods_hours, dtype=float)
    n = len(values)
    if time.shape != (n,) or projection.shape != (n, 2) or covariance.shape != (n, n):
        raise ValueError("Incompatible time, value, projection, or covariance shapes")
    if np.any(periods <= 0) or not np.all(np.isfinite(periods)):
        raise ValueError("Periods must be finite and positive")

    columns: list[np.ndarray] = []
    if include_vector_offset:
        columns.extend([projection[:, 0], projection[:, 1]])
    if include_vector_trend:
        centered_time = time - np.mean(time)
        columns.extend([projection[:, 0] * centered_time, projection[:, 1] * centered_time])
    if nuisance_design is not None:
        nuisance = np.asarray(nuisance_design, dtype=float)
        if nuisance.ndim == 1:
            nuisance = nuisance[:, None]
        if nuisance.shape[0] != n:
            raise ValueError("nuisance_design must have one row per observation")
        columns.extend(nuisance.T)
    if not columns:
        raise ValueError("The baseline model has no columns")
    baseline = np.column_stack(columns)
    cholesky = np.linalg.cholesky(covariance)
    white_values = np.linalg.solve(cholesky, values)
    white_baseline = np.linalg.solve(cholesky, baseline)
    chi2_null, _ = _least_squares_chi2(white_values, white_baseline)

    rows = []
    for period in periods:
        phase = 2.0 * np.pi * (time - np.mean(time)) / (period / 24.0)
        cosine = np.cos(phase)
        sine = np.sin(phase)
        signal = np.column_stack(
            (
                projection[:, 0] * cosine,
                projection[:, 0] * sine,
                projection[:, 1] * cosine,
                projection[:, 1] * sine,
            )
        )
        white_signal = np.linalg.solve(cholesky, signal)
        chi2, coefficients = _least_squares_chi2(
            white_values, np.column_stack((white_baseline, white_signal))
        )
        delta = chi2_null - chi2
        rows.append(
            {
                "period_hours": period,
                "chi2": chi2,
                "delta_chi2": delta,
                "power": delta / chi2_null if chi2_null > 0 else np.nan,
                "east_cos_mas": coefficients[-4],
                "east_sin_mas": coefficients[-3],
                "north_cos_mas": coefficients[-2],
                "north_sin_mas": coefficients[-1],
            }
        )
    return pd.DataFrame(rows)
