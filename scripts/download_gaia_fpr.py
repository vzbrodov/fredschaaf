#!/usr/bin/env python3
"""Download Gaia FPR astrometry and create CCD/transit uncertainty tables."""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests
from astropy.io.votable import parse_single_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fredschaaf_astrometry.fpr import aggregate_transits, prepare_ccd_observations


DEFAULT_TAP_URL = "https://gaia.aip.de/tap/sync"


def run_query(tap_url: str, query: str, timeout: float):
    response = requests.get(
        tap_url,
        params={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "QUERY": query,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_single_table(io.BytesIO(response.content)).to_table(
        use_names_over_ids=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=7065)
    parser.add_argument("--tap-url", default=DEFAULT_TAP_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_query = (
        "SELECT * FROM gaiafpr.sso_source "
        f"WHERE number_mp={args.number}"
    )
    observation_query = (
        "SELECT * FROM gaiafpr.sso_observation "
        f"WHERE number_mp={args.number} ORDER BY epoch_utc"
    )
    source = run_query(args.tap_url, source_query, args.timeout)
    observations = run_query(args.tap_url, observation_query, args.timeout)
    if len(source) != 1:
        raise RuntimeError(
            f"Expected one FPR source for {args.number}, received {len(source)}"
        )

    stem = f"gaia_fpr_{args.number}"
    source.write(args.output_dir / f"{stem}_source.ecsv", overwrite=True)
    observations.write(
        args.output_dir / f"{stem}_observations.ecsv", overwrite=True
    )

    prepared = prepare_ccd_observations(observations.to_pandas())
    transits = aggregate_transits(prepared)
    prepared.to_csv(args.output_dir / f"{stem}_ccd.csv", index=False)
    transits.to_csv(args.output_dir / f"{stem}_transits.csv", index=False)

    good = prepared["good_for_analysis"]
    metadata = {
        "minor_planet_number": args.number,
        "queried_at_utc": datetime.now(UTC).isoformat(),
        "tap_url": args.tap_url,
        "source_adql": source_query,
        "observation_adql": observation_query,
        "ccd_rows": int(len(prepared)),
        "transits": int(prepared["transit_id"].nunique()),
        "good_ccd_rows": int(good.sum()),
        "good_transits": int(prepared.loc[good, "transit_id"].nunique()),
        "note": (
            "Transit table contains propagated uncertainties only. "
            "Astrometric residuals require an orbit model before CCD averaging."
        ),
    }
    (args.output_dir / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(
        "Median good-CCD AL/AC random uncertainty: "
        f"{prepared.loc[good, 'sigma_al_random_mas'].median():.3f} / "
        f"{prepared.loc[good, 'sigma_ac_random_mas'].median():.3f} mas"
    )
    print(
        "Median transit AL/AC total uncertainty: "
        f"{transits['sigma_al_total_mas'].median():.3f} / "
        f"{transits['sigma_ac_total_mas'].median():.3f} mas"
    )


if __name__ == "__main__":
    main()
