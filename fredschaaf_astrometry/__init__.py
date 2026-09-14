"""Reusable astrometry utilities for Gaia FPR and ground-based series."""

from .appulse import AppulseFit, fit_appulse_trajectory
from .combined import ProjectedAstrometry, combine_gaia_and_ground
from .detection import FullStackDetection, build_full_stack_detection, fit_stack_psf
from .distortion import build_distortion_grid, interpolate_distortion, stable_source_ids
from .fpr import (
    add_ccd_residuals,
    aggregate_transit_residuals,
    aggregate_transits,
    prepare_ccd_observations,
)
from .ground_pipeline import ReductionProducts, reduce_series, write_reduction_outputs
from .ground import mpc_217_location, propagate_gaia_astrometry
from .ground_wcs import bootstrap_wcs_file, estimate_translation
from .periodogram import scan_vector_periods
from .series import (
    MotionFit,
    SeriesConfig,
    build_ground_series,
    build_orbit_ground_series,
    fit_series_motion,
    load_ground_series_result,
    load_series_config,
)

__all__ = [
    "add_ccd_residuals",
    "AppulseFit",
    "aggregate_transit_residuals",
    "aggregate_transits",
    "build_distortion_grid",
    "build_ground_series",
    "build_orbit_ground_series",
    "bootstrap_wcs_file",
    "combine_gaia_and_ground",
    "build_full_stack_detection",
    "estimate_translation",
    "fit_appulse_trajectory",
    "fit_stack_psf",
    "fit_series_motion",
    "load_ground_series_result",
    "load_series_config",
    "mpc_217_location",
    "interpolate_distortion",
    "prepare_ccd_observations",
    "ProjectedAstrometry",
    "propagate_gaia_astrometry",
    "reduce_series",
    "ReductionProducts",
    "MotionFit",
    "FullStackDetection",
    "SeriesConfig",
    "scan_vector_periods",
    "stable_source_ids",
    "write_reduction_outputs",
]
