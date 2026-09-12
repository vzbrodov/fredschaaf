"""Common table representation for ground-based series measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time


REQUIRED_RESULT_KEYS = {
    "central_utc",
    "RA_deg",
    "Dec_deg",
    "O-C_xi_mas",
    "O-C_eta_mas",
    "error_xi_mas",
    "error_eta_mas",
}


def load_ground_series_result(
    path: str | Path,
    *,
    series_id: str,
    filter_name: str,
    observatory_code: str = "217",
) -> pd.DataFrame:
    """Convert one key/value notebook result into the common series schema."""
    source_path = Path(path)
    table = pd.read_csv(source_path, dtype={"quantity": str, "value": str})
    if set(table.columns) != {"quantity", "value"}:
        raise ValueError(f"Unexpected result schema in {source_path}")
    values = dict(zip(table["quantity"], table["value"], strict=True))
    missing = sorted(REQUIRED_RESULT_KEYS.difference(values))
    if missing:
        raise ValueError(f"Missing series result keys: {missing}")

    central_time = Time(values["central_utc"], format="isot", scale="utc")
    row = {
        "source": "pulkovo",
        "series_id": series_id,
        "observatory_code": observatory_code,
        "filter": filter_name,
        "central_utc": central_time.isot,
        "jd_utc": float(central_time.jd),
        "ra_deg": float(values["RA_deg"]),
        "dec_deg": float(values["Dec_deg"]),
        "oc_xi_mas": float(values["O-C_xi_mas"]),
        "oc_eta_mas": float(values["O-C_eta_mas"]),
        "cov_xi_xi_mas2": float(values["error_xi_mas"]) ** 2,
        # The legacy notebook fitted xi and eta independently and did not save
        # their covariance.  NaN is intentional: zero would be an unsupported
        # precision claim and must not silently enter a 2D period fit.
        "cov_xi_eta_mas2": np.nan,
        "cov_eta_eta_mas2": float(values["error_eta_mas"]) ** 2,
        "ephemeris_id": "legacy-linear-horizons",
        "input_result": str(source_path),
    }
    for optional in ("median_stack_SNR", "RMS_2D_adjusted_mas"):
        row[optional.lower().replace("-", "_")] = (
            float(values[optional]) if optional in values else np.nan
        )
    return pd.DataFrame([row])
