#!/usr/bin/env python3
"""Estimate whether a colour-dependent refraction term is identifiable."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


def fit_dcr_fixed_effects(samples: pd.DataFrame) -> dict[str, float | int | bool]:
    """Fit one DCR coefficient with per-star and per-frame 2D offsets.

    The fixed effects prevent static distortion and frame translation from being
    mistaken for DCR. This deliberately conservative model needs a useful range
    of zenith distances to identify the colour term.
    """
    data = samples.copy()
    data["tan_z"] = np.sqrt(np.maximum(data["airmass"] ** 2 - 1.0, 0.0))
    stars = {value: index for index, value in enumerate(sorted(data.source_id.unique()))}
    frames = {value: index for index, value in enumerate(sorted(data.frame.unique()))}
    n_stars, n_frames, n_samples = len(stars), len(frames), len(data)
    colour_pivot = float(data.bp_rp.median())
    dcr_column = 2 * n_stars + 2 * n_frames
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    observed: list[float] = []
    for sample_index, row in enumerate(data.itertuples()):
        star_index = stars[row.source_id]
        frame_index = frames[row.frame]
        for component, (measurement, zenith_component) in enumerate(
            (
                (row.oc_xi_mas, row.zenith_xi_unit),
                (row.oc_eta_mas, row.zenith_eta_unit),
            )
        ):
            equation = 2 * sample_index + component
            rows.extend((equation, equation, equation))
            columns.extend(
                (
                    2 * star_index + component,
                    2 * n_stars + 2 * frame_index + component,
                    dcr_column,
                )
            )
            values.extend(
                (
                    1.0,
                    1.0,
                    (row.bp_rp - colour_pivot) * row.tan_z * zenith_component,
                )
            )
            observed.append(measurement)
    design = sparse.csr_matrix(
        (values, (rows, columns)), shape=(2 * n_samples, dcr_column + 1)
    )
    observed_array = np.asarray(observed)
    keep = np.ones(2 * n_samples, dtype=bool)
    for _ in range(5):
        solution = lsqr(
            design[keep], observed_array[keep], atol=1e-10, btol=1e-10, iter_lim=2000
        )[0]
        residual = observed_array - design @ solution
        radial = np.hypot(residual[0::2], residual[1::2])
        median = np.median(radial)
        robust_sigma = 1.4826 * np.median(np.abs(radial - median))
        good_samples = radial < median + 4.0 * robust_sigma
        keep = np.repeat(good_samples, 2)

    nuisance = design[:, :-1]
    dcr_predictor = design[:, -1].toarray().ravel()
    nuisance_fit = lsqr(
        nuisance[keep], dcr_predictor[keep], atol=1e-10, btol=1e-10, iter_lim=2000
    )[0]
    independent_predictor = dcr_predictor - nuisance @ nuisance_fit
    degrees_of_freedom = max(int(keep.sum() - design.shape[1] + 2), 1)
    residual_sigma = float(np.sqrt(np.sum(residual[keep] ** 2) / degrees_of_freedom))
    predictor_norm = float(np.sqrt(np.sum(independent_predictor[keep] ** 2)))
    formal_error = residual_sigma / predictor_norm
    coefficient = float(solution[-1])
    significance = abs(coefficient) / formal_error
    tan_z_min = float(data.tan_z.min())
    tan_z_max = float(data.tan_z.max())
    return {
        "samples": n_samples,
        "retained_samples": int(good_samples.sum()),
        "stars": n_stars,
        "frames": n_frames,
        "colour_pivot_bp_rp": colour_pivot,
        "tan_z_min": tan_z_min,
        "tan_z_max": tan_z_max,
        "dcr_mas_per_bp_rp_per_tan_z": coefficient,
        "formal_error_mas_per_bp_rp_per_tan_z": formal_error,
        "formal_significance_sigma": significance,
        "residual_sigma_mas": residual_sigma,
        "identifiable": bool(significance >= 3.0 and tan_z_max - tan_z_min >= 0.2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=Path, default=Path("outputs/two_zone_distortion_samples_20250903_R.csv")
    )
    parser.add_argument(
        "--frames", type=Path, default=Path("outputs/appulse_frames_20250903_R.csv")
    )
    parser.add_argument(
        "--catalog", type=Path, default=Path("Fredschaaf/20250903/gaia_dr3_field_g19_5.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/ground_dcr_diagnostic.csv"))
    args = parser.parse_args()

    samples = pd.read_csv(args.samples)
    frames = pd.read_csv(args.frames).drop_duplicates("frame")
    catalog = pd.read_csv(args.catalog, usecols=["source_id", "bp_rp"])
    data = (
        samples.merge(
            frames[["frame", "airmass", "zenith_xi_unit", "zenith_eta_unit"]],
            on="frame",
        )
        .merge(catalog, on="source_id")
    )
    data = data[
        (~data.outer_fit_used)
        & data.bp_rp.notna()
        & data.airmass.notna()
        & data.zenith_xi_unit.notna()
        & data.zenith_eta_unit.notna()
    ].copy()
    result = pd.DataFrame([fit_dcr_fixed_effects(data)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
