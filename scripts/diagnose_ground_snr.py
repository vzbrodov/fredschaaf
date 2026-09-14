#!/usr/bin/env python3
"""Summarize the empirical image-noise budget behind ground-series S/N."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def sampled_frame_noise(paths: list[Path], maximum_pairs: int) -> dict[str, float]:
    if len(paths) < 2:
        raise ValueError("At least two frames are required for a temporal-noise estimate")
    pair_starts = np.unique(
        np.linspace(0, len(paths) - 2, min(maximum_pairs, len(paths) - 1), dtype=int)
    )
    spatial_noise = []
    temporal_noise = []
    sky = []
    headers = []
    for index in pair_starts:
        first, header = fits.getdata(paths[index], header=True, memmap=False)
        second = fits.getdata(paths[index + 1], memmap=False)
        first = np.asarray(first, float)
        second = np.asarray(second, float)
        ny, nx = first.shape
        half = min(250, ny // 2, nx // 2)
        first = first[ny // 2 - half : ny // 2 + half, nx // 2 - half : nx // 2 + half]
        second = second[ny // 2 - half : ny // 2 + half, nx // 2 - half : nx // 2 + half]
        sky.append(float(np.nanmedian(first)))
        spatial_noise.append(robust_sigma(first))
        temporal_noise.append(robust_sigma(second - first) / np.sqrt(2.0))
        headers.append(header)
    return {
        "sampled_pairs": len(pair_starts),
        "median_sky_adu": float(np.median(sky)),
        "spatial_pixel_noise_adu": float(np.median(spatial_noise)),
        "temporal_pixel_noise_adu": float(np.median(temporal_noise)),
        "exposure_seconds": float(headers[0].get("EXPTIME", np.nan)),
        "camera_gain_setting": float(headers[0].get("GAIN", np.nan)),
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
        noise = sampled_frame_noise(paths, args.maximum_frame_pairs)
        detection = pd.read_csv(
            args.output_dir / f"full_stack_detection_{series_id}.csv"
        ).iloc[0]
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
        systematics_path = args.output_dir / f"appulse_systematics_{series_id}.csv"
        if systematics_path.exists():
            systematics = pd.read_csv(systematics_path).iloc[0]
            fwhm = float(systematics.median_psf_fwhm_px)
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
