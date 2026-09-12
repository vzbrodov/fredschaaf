#!/usr/bin/env python3
"""Download an expanded Gaia DR3 catalogue around the Pulkovo field."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import requests
from astropy.io.votable import parse_single_table


DEFAULT_TAP_URL = "https://gaia.aip.de/tap/sync"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ra", type=float, default=43.4879)
    parser.add_argument("--dec", type=float, default=15.4405)
    parser.add_argument("--radius-deg", type=float, default=0.32)
    parser.add_argument("--max-gmag", type=float, default=19.5)
    parser.add_argument("--tap-url", default=DEFAULT_TAP_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Fredschaaf/20250903/gaia_dr3_field_g19_5.csv"),
    )
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    query = f"""
SELECT source_id, ra, dec, pmra, pmdec, ref_epoch, phot_g_mean_mag,
       bp_rp, ruwe, duplicated_source
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {args.ra}, {args.dec}, {args.radius_deg})
)
AND phot_g_mean_mag <= {args.max_gmag}
AND pmra IS NOT NULL AND pmdec IS NOT NULL
""".strip()
    response = requests.get(
        args.tap_url,
        params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query},
        timeout=args.timeout,
    )
    response.raise_for_status()
    table = parse_single_table(io.BytesIO(response.content)).to_table(use_names_over_ids=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_pandas().sort_values("phot_g_mean_mag").to_csv(args.output, index=False)
    print(f"Saved {len(table)} Gaia DR3 sources to {args.output}")


if __name__ == "__main__":
    main()
