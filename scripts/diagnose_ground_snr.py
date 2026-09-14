#!/usr/bin/env python3
"""Summarize the empirical image-noise budget behind ground-series S/N."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.ground import (
    exposure_middle_time,
    linear_ephemeris_coordinates,
)
from fredschaaf_astrometry.series import load_series_config


DEFAULT_CONFIGS = (
    Path("configs/20250902_R.toml"),
    Path("configs/20250903_R.toml"),
    Path("configs/20250903_H_alpha.toml"),
    Path("configs/20251010_R.toml"),
)


def robust_sigma(values: np.ndarray) -> float:
    median = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - median)))


def block_background_map(
    image: np.ndarray, rows: int = 8, columns: int = 12
) -> np.ndarray:
    """Return block medians normalized by the whole-frame median."""
    blocks = np.empty((rows, columns), float)
    for row, y_indices in enumerate(
        np.array_split(np.arange(image.shape[0]), rows)
    ):
        for column, x_indices in enumerate(
            np.array_split(np.arange(image.shape[1]), columns)
        ):
            blocks[row, column] = np.nanmedian(image[np.ix_(y_indices, x_indices)])
    return blocks / np.nanmedian(image)


def local_background_metrics(
    image: np.ndarray,
    x: float,
    y: float,
    *,
    background_span_px: float,
    inner_radius_px: float = 10.0,
    outer_radius_px: float = 18.0,
) -> dict[str, float]:
    """Compare a scalar and linear-plane background around the moving target."""
    half = int(np.ceil(outer_radius_px))
    x0, y0 = int(round(x)), int(round(y))
    if (
        x0 - half < 0
        or x0 + half >= image.shape[1]
        or y0 - half < 0
        or y0 + half >= image.shape[0]
    ):
        raise ValueError("Target annulus falls outside the detector")
    cutout = image[y0 - half : y0 + half + 1, x0 - half : x0 + half + 1]
    yy, xx = np.indices(cutout.shape, dtype=float)
    dx = xx + x0 - half - x
    dy = yy + y0 - half - y
    radius = np.hypot(dx, dy)
    annulus = (radius >= inner_radius_px) & (radius <= outer_radius_px)
    values = cutout[annulus]
    design = np.column_stack((np.ones(annulus.sum()), dx[annulus], dy[annulus]))
    keep = np.isfinite(values)
    for _ in range(4):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residuals = values - design @ coefficients
        center = np.nanmedian(residuals[keep])
        sigma = robust_sigma(residuals[keep])
        if not np.isfinite(sigma) or sigma <= 0:
            break
        updated = np.isfinite(values) & (np.abs(residuals - center) <= 3.0 * sigma)
        if np.array_equal(updated, keep):
            break
        keep = updated
    constant_noise = robust_sigma(values[keep])
    plane_noise = robust_sigma((values - design @ coefficients)[keep])
    gradient = float(np.hypot(coefficients[1], coefficients[2]))
    return {
        "local_background_adu": float(np.median(values[keep])),
        "local_constant_noise_adu": constant_noise,
        "local_plane_noise_adu": plane_noise,
        "local_plane_to_constant_noise_ratio": plane_noise / constant_noise,
        "local_gradient_across_aperture_diameter_adu": gradient
        * background_span_px,
    }


def sampled_frame_noise(
    paths: list[Path],
    maximum_pairs: int,
    *,
    ephemeris: dict[str, object],
    background_span_px: float,
    saturation_level: float,
) -> dict[str, float]:
    if len(paths) < 2:
        raise ValueError("At least two frames are required for a temporal-noise estimate")
    pair_starts = np.unique(
        np.linspace(0, len(paths) - 2, min(maximum_pairs, len(paths) - 1), dtype=int)
    )
    spatial_noise = []
    temporal_noise = []
    sky = []
    headers = []
    background_maps = []
    local_metrics = []
    saturation_fractions = []
    for index in pair_starts:
        first, header = fits.getdata(paths[index], header=True, memmap=False)
        second = fits.getdata(paths[index + 1], memmap=False)
        first = np.asarray(first, float)
        second = np.asarray(second, float)
        background_maps.append(block_background_map(first))
        saturation_fractions.append(float(np.mean(first >= saturation_level)))
        middle = exposure_middle_time(header)
        ra, dec = linear_ephemeris_coordinates(
            middle,
            reference_time=Time(str(ephemeris["reference_utc"]), scale="utc"),
            reference_ra_deg=float(ephemeris["ra_deg"]),
            reference_dec_deg=float(ephemeris["dec_deg"]),
            ra_rate_arcsec_per_hour=float(ephemeris["ra_rate_arcsec_per_hour"]),
            dec_rate_arcsec_per_hour=float(ephemeris["dec_rate_arcsec_per_hour"]),
        )
        target_x, target_y = WCS(header, fix=False).world_to_pixel_values(ra, dec)
        local_metrics.append(
            local_background_metrics(
                first,
                float(target_x),
                float(target_y),
                background_span_px=background_span_px,
            )
        )
        ny, nx = first.shape
        half = min(250, ny // 2, nx // 2)
        first = first[
            ny // 2 - half : ny // 2 + half,
            nx // 2 - half : nx // 2 + half,
        ]
        second = second[
            ny // 2 - half : ny // 2 + half,
            nx // 2 - half : nx // 2 + half,
        ]
        sky.append(float(np.nanmedian(first)))
        spatial_noise.append(robust_sigma(first))
        temporal_noise.append(robust_sigma(second - first) / np.sqrt(2.0))
        headers.append(header)
    maps = np.stack(background_maps)
    stable_map = np.median(maps, axis=0)
    local = pd.DataFrame(local_metrics)
    return {
        "sampled_pairs": len(pair_starts),
        "median_sky_adu": float(np.median(sky)),
        "spatial_pixel_noise_adu": float(np.median(spatial_noise)),
        "temporal_pixel_noise_adu": float(np.median(temporal_noise)),
        "exposure_seconds": float(headers[0].get("EXPTIME", np.nan)),
        "camera_gain_setting": float(headers[0].get("GAIN", np.nan)),
        "stable_background_pattern_rms_fraction": float(
            np.sqrt(np.mean((stable_map - 1.0) ** 2))
        ),
        "stable_background_pattern_min_fraction": float(np.min(stable_map)),
        "stable_background_pattern_max_fraction": float(np.max(stable_map)),
        "frame_to_frame_block_scatter_fraction": float(
            np.median(
                [
                    robust_sigma(maps[:, row, column])
                    for row in range(8)
                    for column in range(12)
                ]
            )
        ),
        "saturation_fraction": float(np.median(saturation_fractions)),
        "saturation_threshold_adu": saturation_level,
        "local_background_range_adu": float(
            local.local_background_adu.max() - local.local_background_adu.min()
        ),
        "local_noise_range_adu": float(
            local.local_constant_noise_adu.max()
            - local.local_constant_noise_adu.min()
        ),
        **{
            f"median_{column}": float(local[column].median())
            for column in local.columns
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--maximum-frame-pairs", type=int, default=20)
    args = parser.parse_args()
    if args.maximum_frame_pairs < 1:
        raise SystemExit("maximum-frame-pairs must be positive")

    rows = []
    for config_path in args.config or DEFAULT_CONFIGS:
        config = load_series_config(config_path)
        series_id = str(config.series["id"])
        pattern = config.path.parent / str(config.reduction["frame_glob"])
        paths = sorted(pattern.parent.glob(pattern.name))
        detection = pd.read_csv(
            args.output_dir / f"full_stack_detection_{series_id}.csv"
        ).iloc[0]
        systematics_path = args.output_dir / f"appulse_systematics_{series_id}.csv"
        fwhm = np.nan
        if systematics_path.exists():
            systematics = pd.read_csv(systematics_path).iloc[0]
            fwhm = float(systematics.median_psf_fwhm_px)
        noise = sampled_frame_noise(
            paths,
            args.maximum_frame_pairs,
            ephemeris=config.ephemeris,
            background_span_px=2.0
            * float(config.reduction["psf_aperture_radius_px"]),
            saturation_level=float(config.reduction["psf_saturation_level"]),
        )
        has_forced_measurement = (
            "forced_flux_snr" in detection.index
            and np.isfinite(float(detection.forced_flux_snr))
        )
        analysis_flux = float(
            detection.forced_flux if has_forced_measurement else detection.flux
        )
        analysis_snr = float(
            detection.forced_flux_snr
            if has_forced_measurement
            else detection.fitted_flux_snr
        )
        row = {
            "series_id": series_id,
            "filter": str(config.series["filter"]),
            "bias_calibrated": config.calibration["bias"],
            "dark_calibrated": config.calibration["dark"],
            "flat_calibrated": config.calibration["flat"],
            "frames_in_detection": int(detection.frames),
            **noise,
            "full_stack_flux_adu": float(detection.flux),
            "full_stack_fitted_snr": float(detection.fitted_flux_snr),
            "snr_measurement": "forced" if has_forced_measurement else "blind-fit",
            "analysis_flux_adu": analysis_flux,
            "analysis_snr": analysis_snr,
            "equivalent_single_frame_snr": float(
                analysis_snr / np.sqrt(detection.frames)
            ),
        }
        if systematics_path.exists():
            noise_area = 4.0 * np.pi * (fwhm / 2.35482) ** 2
            row["median_psf_fwhm_px"] = fwhm
            row["noise_equivalent_area_px2_gaussian"] = noise_area
            row["simple_matched_filter_snr"] = float(
                analysis_flux
                / (noise["temporal_pixel_noise_adu"] * np.sqrt(noise_area))
                * np.sqrt(detection.frames)
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "ground_snr_budget.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
