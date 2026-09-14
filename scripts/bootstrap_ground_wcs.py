#!/usr/bin/env python3
"""Bootstrap WCS headers for an overlapping unsolved ground series."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from astropy.wcs import FITSFixedWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.ground_wcs import (
    bootstrap_wcs_file,
    bootstrap_wcs_from_catalog_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob", default="Fredschaaf/20250902/*.fits"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("Fredschaaf/20250903/7065_R_4_0001_wcs.fits"),
    )
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--quantile", type=float, default=99.5)
    parser.add_argument("--maximum-shift-px", type=float, default=900.0)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="seed the first frame of a new field from this Gaia CSV",
    )
    parser.add_argument("--catalog-max-gmag", type=float, default=16.5)
    parser.add_argument(
        "--chain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match each frame to the previous solved frame after the first",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pattern = Path(args.input_glob)
    paths = sorted(
        path for path in pattern.parent.glob(pattern.name)
        if not path.stem.endswith("_wcs")
    )
    if not paths:
        raise SystemExit(f"No input files match {args.input_glob}")
    warnings.simplefilter("ignore", FITSFixedWarning)
    reference = args.reference
    for frame_number, path in enumerate(paths):
        output = path.with_name(f"{path.stem}_wcs{path.suffix}")
        if frame_number == 0 and args.catalog is not None:
            dx, dy, contrast = bootstrap_wcs_from_catalog_file(
                path,
                reference,
                args.catalog,
                output,
                maximum_magnitude=args.catalog_max_gmag,
                downsample=args.downsample,
                quantile=args.quantile,
                maximum_shift_px=args.maximum_shift_px,
                overwrite=args.overwrite,
            )
        else:
            dx, dy, contrast = bootstrap_wcs_file(
                path,
                reference,
                output,
                downsample=args.downsample,
                quantile=args.quantile,
                maximum_shift_px=args.maximum_shift_px,
                overwrite=args.overwrite,
            )
        print(
            f"{path.name}: dx={dx:+.2f} px, dy={dy:+.2f} px, "
            f"contrast={contrast:.0f} -> {output.name}"
        )
        if args.chain:
            reference = output


if __name__ == "__main__":
    main()
