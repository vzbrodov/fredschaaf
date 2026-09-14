#!/usr/bin/env python3
"""Detect a faint asteroid in one motion-compensated full-series stack."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import FITSFixedWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.detection import build_full_stack_detection
from fredschaaf_astrometry.ground_pipeline import reduce_series
from fredschaaf_astrometry.series import load_series_config


def main() -> None:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    warnings.filterwarnings("ignore", message="The fit may not have converged.*")
    warnings.filterwarnings("ignore", message="Input data contains invalid values.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--maximum-offset-px", type=float, default=8.0)
    parser.add_argument("--stack-half-size-px", type=int)
    args = parser.parse_args()
    config = load_series_config(args.config)
    products = reduce_series(config)
    detection = build_full_stack_detection(
        products,
        config,
        maximum_offset_px=args.maximum_offset_px,
        stack_half_size_px=args.stack_half_size_px,
    )
    series_id = str(config.series["id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / f"full_stack_detection_{series_id}.csv"
    fits_path = args.output_dir / f"full_stack_detection_{series_id}.fits"
    plot_path = args.output_dir / f"full_stack_detection_{series_id}.png"
    split_path = args.output_dir / f"full_stack_split_check_{series_id}.csv"
    sigma = np.sqrt(np.diag(detection.covariance_mas2))
    table = pd.DataFrame([{
        "series_id": series_id,
        "central_utc": Time(detection.reference_jd, format="jd", scale="utc").isot,
        "frames": len(detection.frame_indices),
        "span_minutes": detection.span_minutes,
        "ra_deg": detection.ra_deg,
        "dec_deg": detection.dec_deg,
        "offset_xi_mas": detection.offset_mas[0],
        "offset_eta_mas": detection.offset_mas[1],
        "formal_error_xi_mas": sigma[0],
        "formal_error_eta_mas": sigma[1],
        "cov_xi_eta_mas2": detection.covariance_mas2[0, 1],
        "flux": detection.flux,
        "flux_error": detection.flux_error,
        "fitted_flux_snr": detection.fitted_snr,
        "peak_pixel_snr": detection.peak_snr,
        "dx_px": detection.offset_px[0],
        "dy_px": detection.offset_px[1],
        "at_search_boundary": detection.at_search_boundary,
        "detection_ok": bool(
            detection.fitted_snr >= 8.0 and not detection.at_search_boundary
        ),
    }])
    table.to_csv(table_path, index=False)
    split_rows = []
    for label, indices in (
        ("even", detection.frame_indices[::2]),
        ("odd", detection.frame_indices[1::2]),
    ):
        if len(indices) < 3:
            continue
        split = build_full_stack_detection(
            products,
            config,
            maximum_offset_px=args.maximum_offset_px,
            stack_half_size_px=args.stack_half_size_px,
            frame_indices=indices,
        )
        split_rows.append({
            "split": label,
            "frames": len(indices),
            "central_utc": Time(split.reference_jd, format="jd", scale="utc").isot,
            "offset_xi_mas": split.offset_mas[0],
            "offset_eta_mas": split.offset_mas[1],
            "fitted_flux_snr": split.fitted_snr,
            "at_search_boundary": split.at_search_boundary,
        })
    split_table = pd.DataFrame(split_rows)
    if len(split_table) == 2:
        difference = np.hypot(
            np.diff(split_table.offset_xi_mas)[0],
            np.diff(split_table.offset_eta_mas)[0],
        )
        split_table["split_difference_2d_mas"] = difference
    split_table.to_csv(split_path, index=False)
    fits.PrimaryHDU(detection.image).writeto(fits_path, overwrite=True)
    center = (np.array(detection.image.shape[::-1]) - 1) / 2
    fitted = center + detection.offset_px
    fig, axis = plt.subplots(figsize=(6, 5))
    median = np.nanmedian(detection.image)
    scatter = 1.4826 * np.nanmedian(np.abs(detection.image - median))
    axis.imshow(
        detection.image,
        origin="lower",
        cmap="gray_r",
        vmin=median - scatter,
        vmax=median + 6 * scatter,
    )
    axis.plot(center[0], center[1], "+", color="tab:blue", ms=14, mew=2, label="ephemeris")
    axis.plot(fitted[0], fitted[1], "o", mfc="none", mec="tab:red", ms=14, mew=2, label="PSF fit")
    axis.set(title=f"{series_id}: full motion-compensated stack", xlabel="x, px", ylabel="y, px")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(table.to_string(index=False))
    print(split_table.to_string(index=False))
    print(f"Saved {table_path}, {split_path}, {fits_path}, and {plot_path}")


if __name__ == "__main__":
    main()
