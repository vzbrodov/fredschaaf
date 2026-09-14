#!/usr/bin/env python3
"""Run the complete two-zone reduction for one configured CCD series."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from astropy.wcs import FITSFixedWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.ground_pipeline import (
    reduce_series,
    write_reduction_outputs,
)
from fredschaaf_astrometry.series import load_series_config


def main() -> None:
    warnings.simplefilter("ignore", FITSFixedWarning)
    warnings.filterwarnings("ignore", message="The fit may not have converged")
    warnings.filterwarnings("ignore", message="Input data contains invalid values")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/20250903_R.toml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    config = load_series_config(args.config)
    products = reduce_series(config)
    paths = write_reduction_outputs(products, config, args.output_dir)
    print(products.summary.to_string(index=False))
    print(
        f"Accepted {products.frame_quality.accepted.sum()} / "
        f"{len(products.frame_quality)} frames; wrote {len(paths)} tables"
    )


if __name__ == "__main__":
    main()
