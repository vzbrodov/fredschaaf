"""Robust, source-balanced detector distortion maps."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator


def _robust_standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)) / np.sqrt(len(values)))


def stable_source_ids(
    samples: pd.DataFrame,
    *,
    min_frames: int = 20,
    max_scatter_mas: float = 500.0,
) -> np.ndarray:
    """Select sources with repeatable two-coordinate centroids."""
    rows = []
    for source_id, group in samples.groupby("source_id"):
        if group["frame"].nunique() < min_frames:
            continue
        scatters = []
        for component in ("oc_xi_mas", "oc_eta_mas"):
            values = group[component].to_numpy(float)
            median = np.nanmedian(values)
            scatters.append(1.4826 * np.nanmedian(np.abs(values - median)))
        if np.hypot(*scatters) <= max_scatter_mas:
            rows.append(source_id)
    return np.asarray(rows)


def build_distortion_grid(
    samples: pd.DataFrame,
    *,
    detector_width: float,
    detector_height: float,
    bins_x: int = 12,
    bins_y: int = 8,
    min_sources: int = 3,
    min_frames: int = 5,
    min_source_frames: int = 20,
    max_source_scatter_mas: float = 500.0,
    remove_affine_modes: bool = True,
) -> pd.DataFrame:
    """Bin star O-C vectors into a static detector correction grid.

    Every catalogue source receives equal weight: measurements are first
    median-combined per source inside a cell, then sources are combined. This
    prevents a star seen on many frames from dominating a cell. Affine modes
    are removed by default because they are degenerate with each frame's
    astrometric plate model.
    """
    required = {"frame", "source_id", "x", "y", "oc_xi_mas", "oc_eta_mas"}
    missing = sorted(required.difference(samples.columns))
    if missing:
        raise ValueError(f"Missing distortion-sample columns: {missing}")
    data = samples.copy()
    finite = np.isfinite(data[["x", "y", "oc_xi_mas", "oc_eta_mas"]]).all(axis=1)
    data = data.loc[finite].copy()
    data = data[
        data["x"].between(0, detector_width, inclusive="left")
        & data["y"].between(0, detector_height, inclusive="left")
    ]
    accepted_sources = stable_source_ids(
        data,
        min_frames=min_source_frames,
        max_scatter_mas=max_source_scatter_mas,
    )
    data = data[data["source_id"].isin(accepted_sources)]
    data["cell_x"] = np.floor(data["x"] / detector_width * bins_x).astype(int)
    data["cell_y"] = np.floor(data["y"] / detector_height * bins_y).astype(int)

    rows = []
    for (cell_y, cell_x), group in data.groupby(["cell_y", "cell_x"]):
        by_source = group.groupby("source_id")[["oc_xi_mas", "oc_eta_mas"]].median()
        n_sources = len(by_source)
        n_frames = group["frame"].nunique()
        if n_sources < min_sources or n_frames < min_frames:
            continue
        xi = by_source["oc_xi_mas"].to_numpy(float)
        eta = by_source["oc_eta_mas"].to_numpy(float)
        rows.append(
            {
                "cell_x": int(cell_x),
                "cell_y": int(cell_y),
                "center_x": (cell_x + 0.5) * detector_width / bins_x,
                "center_y": (cell_y + 0.5) * detector_height / bins_y,
                "n_measurements": len(group),
                "n_sources": n_sources,
                "n_frames": n_frames,
                "raw_xi_mas": float(np.median(xi)),
                "raw_eta_mas": float(np.median(eta)),
                "error_xi_mas": _robust_standard_error(xi),
                "error_eta_mas": _robust_standard_error(eta),
            }
        )
    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid

    grid["correction_xi_mas"] = grid["raw_xi_mas"]
    grid["correction_eta_mas"] = grid["raw_eta_mas"]
    if remove_affine_modes and len(grid) >= 3:
        normalized_x = 2.0 * grid["center_x"].to_numpy(float) / detector_width - 1.0
        normalized_y = 2.0 * grid["center_y"].to_numpy(float) / detector_height - 1.0
        design = np.column_stack((np.ones(len(grid)), normalized_x, normalized_y))
        weights = np.sqrt(grid["n_sources"].to_numpy(float))[:, None]
        for component in ("xi", "eta"):
            raw = grid[f"raw_{component}_mas"].to_numpy(float)
            coefficients = np.linalg.lstsq(design * weights, raw * weights[:, 0], rcond=None)[0]
            grid[f"correction_{component}_mas"] = raw - design @ coefficients
    return grid.sort_values(["cell_y", "cell_x"]).reset_index(drop=True)


def interpolate_distortion(
    grid: pd.DataFrame, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate the grid; return NaN outside its convex hull."""
    if len(grid) < 3:
        shape = np.broadcast_arrays(x, y)[0].shape
        return np.full(shape, np.nan), np.full(shape, np.nan)
    points = grid[["center_x", "center_y"]].to_numpy(float)
    broadcast_x, broadcast_y = np.broadcast_arrays(x, y)
    query = np.column_stack((broadcast_x.ravel(), broadcast_y.ravel()))
    results = []
    for component in ("xi", "eta"):
        interpolator = LinearNDInterpolator(
            points, grid[f"correction_{component}_mas"].to_numpy(float), fill_value=np.nan
        )
        results.append(np.asarray(interpolator(query)).reshape(broadcast_x.shape))
    return results[0], results[1]
