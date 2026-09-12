"""Reusable astrometry utilities for Gaia FPR and ground-based series."""

from .combined import ProjectedAstrometry, combine_gaia_and_ground
from .distortion import build_distortion_grid, interpolate_distortion, stable_source_ids
from .fpr import (
    add_ccd_residuals,
    aggregate_transit_residuals,
    aggregate_transits,
    prepare_ccd_observations,
)
from .periodogram import scan_vector_periods
from .series import load_ground_series_result

__all__ = [
    "add_ccd_residuals",
    "aggregate_transit_residuals",
    "aggregate_transits",
    "build_distortion_grid",
    "combine_gaia_and_ground",
    "load_ground_series_result",
    "interpolate_distortion",
    "prepare_ccd_observations",
    "ProjectedAstrometry",
    "scan_vector_periods",
    "stable_source_ids",
]
