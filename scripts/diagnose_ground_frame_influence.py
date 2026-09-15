#!/usr/bin/env python3
"""Measure leave-one-frame-out sensitivity of a full motion stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.detection import build_full_stack_detection
from fredschaaf_astrometry.ground_pipeline import reduce_series
from fredschaaf_astrometry.series import load_series_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--maximum-offset-px", type=float, default=8.0)
    args = parser.parse_args()

    config = load_series_config(args.config)
    products = reduce_series(config)
    full = build_full_stack_detection(
        products, config, maximum_offset_px=args.maximum_offset_px
    )
    rows = []
    for omitted in full.frame_indices:
        retained = tuple(index for index in full.frame_indices if index != omitted)
        fit = build_full_stack_detection(
            products,
            config,
            maximum_offset_px=args.maximum_offset_px,
            frame_indices=retained,
        )
        delta = fit.offset_mas - full.offset_mas
        quality = products.frame_quality.iloc[omitted]
        rows.append(
            {
                "omitted_frame": omitted,
                "omitted_filename": quality.filename,
                "outer_rms_mas": quality.outer_rms_mas,
                "inner_rms_after_mas": quality.inner_rms_after_mas,
                "correction_xi_mas": quality.correction_xi_mas,
                "correction_eta_mas": quality.correction_eta_mas,
                "retained_frames": len(retained),
                "fitted_flux_snr": fit.fitted_snr,
                "offset_xi_mas": fit.offset_mas[0],
                "offset_eta_mas": fit.offset_mas[1],
                "change_xi_mas": delta[0],
                "change_eta_mas": delta[1],
                "change_2d_mas": float(np.hypot(*delta)),
            }
        )
    result = pd.DataFrame(rows).sort_values("change_2d_mas", ascending=False)
    series_id = str(config.series["id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"frame_influence_{series_id}.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
