#!/usr/bin/env python3
"""Build and validate a detector distortion grid from exported star O-C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.distortion import (
    build_distortion_grid,
    interpolate_distortion,
    stable_source_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/two_zone_distortion_samples_20250903_R.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--width", type=float, default=2072)
    parser.add_argument("--height", type=float, default=1410)
    # Only 36 distinct reference stars are present in the current field, so a
    # deliberately coarse default is statistically supportable.
    parser.add_argument("--bins-x", type=int, default=4)
    parser.add_argument("--bins-y", type=int, default=3)
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--max-source-scatter-mas", type=float, default=500)
    args = parser.parse_args()

    samples = pd.read_csv(args.input)
    stable_ids = stable_source_ids(
        samples, min_frames=20, max_scatter_mas=args.max_source_scatter_mas
    )
    # Hold out whole catalogue sources. This prevents a catalogue error of one
    # repeatedly measured star from masquerading as detector distortion.
    validation_ids = set(stable_ids[::5])
    training = samples[
        samples["source_id"].isin(stable_ids) & ~samples["source_id"].isin(validation_ids)
    ]
    validation = samples[samples["source_id"].isin(validation_ids)]
    validation_grid = build_distortion_grid(
        training,
        detector_width=args.width,
        detector_height=args.height,
        bins_x=args.bins_x,
        bins_y=args.bins_y,
        min_sources=args.min_sources,
        min_frames=args.min_frames,
        max_source_scatter_mas=args.max_source_scatter_mas,
    )
    validation_points = validation.groupby("source_id").agg(
        x=("x", "median"),
        y=("y", "median"),
        oc_xi_mas=("oc_xi_mas", "median"),
        oc_eta_mas=("oc_eta_mas", "median"),
    )
    map_xi, map_eta = interpolate_distortion(
        validation_grid,
        validation_points["x"].to_numpy(),
        validation_points["y"].to_numpy(),
    )
    valid = np.isfinite(map_xi + map_eta)
    before = np.hypot(
        validation_points.loc[valid, "oc_xi_mas"],
        validation_points.loc[valid, "oc_eta_mas"],
    )
    after = np.hypot(
        validation_points.loc[valid, "oc_xi_mas"].to_numpy() - map_xi[valid],
        validation_points.loc[valid, "oc_eta_mas"].to_numpy() - map_eta[valid],
    )

    grid = build_distortion_grid(
        samples,
        detector_width=args.width,
        detector_height=args.height,
        bins_x=args.bins_x,
        bins_y=args.bins_y,
        min_sources=args.min_sources,
        min_frames=args.min_frames,
        max_source_scatter_mas=args.max_source_scatter_mas,
    )
    if grid.empty:
        raise RuntimeError(
            "No populated grid cells; use fewer bins or lower the explicit "
            "minimum-source requirement"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "distortion_grid_20250903_R.csv"
    figure_path = args.output_dir / "distortion_grid_20250903_R.png"
    diagnostics_path = args.output_dir / "distortion_grid_20250903_R_diagnostics.json"
    grid.to_csv(table_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    magnitude = np.hypot(grid["correction_xi_mas"], grid["correction_eta_mas"])
    quiver = ax.quiver(
        grid["center_x"], grid["center_y"], grid["correction_xi_mas"],
        grid["correction_eta_mas"], magnitude, angles="xy", scale_units="xy", scale=0.15
    )
    ax.quiverkey(quiver, 0.88, 1.03, 20, "20 mas", labelpos="E")
    fig.colorbar(quiver, ax=ax, label="|поправка|, mas")
    ax.set(xlabel="x, pixel", ylabel="y, pixel", title="Остаточная дисторсия после аффинной модели")
    ax.set_xlim(0, args.width)
    ax.set_ylim(0, args.height)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    diagnostics = {
        "grid_cells": len(grid),
        "stable_sources": int(len(stable_ids)),
        "validation_sources": int(valid.sum()),
        "validation_median_before_mas": float(np.median(before)) if valid.any() else None,
        "validation_median_after_mas": float(np.median(after)) if valid.any() else None,
        # Demand a material cross-validation gain, not a sub-percent numerical
        # fluctuation in a noisy validation sample.
        "recommended_for_correction": bool(
            valid.any() and np.median(after) < 0.99 * np.median(before)
        ),
        "warning": (
            "Same-pointing data cannot separate catalogue errors from detector distortion."
        ),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"Stable sources: {len(stable_ids)}; grid cells: {len(grid)}; "
        f"validation sources: {valid.sum()}"
    )
    if valid.any():
        print(f"Validation median |O-C|: {np.median(before):.2f} -> {np.median(after):.2f} mas")
        if not diagnostics["recommended_for_correction"]:
            print("Map rejected as a correction: held-out median did not improve.")
    print(table_path)
    print(figure_path)
    print(diagnostics_path)


if __name__ == "__main__":
    main()
