#!/usr/bin/env python3
"""Plot adaptive substack positions and detection significance."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from pandas.errors import EmptyDataError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("series_id", nargs="+")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    for series_id in args.series_id:
        table = pd.read_csv(args.input_dir / f"adaptive_substacks_{series_id}.csv")
        trajectory_path = args.input_dir / f"adaptive_trajectory_{series_id}.csv"
        try:
            trajectory = pd.read_csv(trajectory_path)
        except EmptyDataError:
            trajectory = pd.DataFrame()
        reference = Time(table.central_utc.iloc[0], scale="utc")
        if len(trajectory):
            reference = Time(trajectory.reference_utc.iloc[0], scale="utc")
        hours = np.array([(Time(value, scale="utc") - reference).to_value("hour") for value in table.central_utc])
        good = table.detection_ok.astype(bool).to_numpy()
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        for axis, coordinate, error, label in (
            (axes[0], "offset_xi_mas", "formal_error_xi_mas", r"$\Delta\xi$, mas"),
            (axes[1], "offset_eta_mas", "formal_error_eta_mas", r"$\Delta\eta$, mas"),
        ):
            for mask, color, name in ((good, "tab:blue", "used"), (~good, "tab:red", "failed QC")):
                if np.any(mask):
                    axis.errorbar(
                        hours[mask], table.loc[mask, coordinate],
                        yerr=table.loc[mask, error], fmt="o", color=color, label=name,
                    )
            if len(trajectory):
                row = trajectory.iloc[0]
                intercept = row[coordinate]
                velocity = row[f"velocity_correction_{'xi' if coordinate == 'offset_xi_mas' else 'eta'}_mas_per_hour"]
                grid = np.linspace(hours.min(), hours.max(), 100)
                axis.plot(grid, intercept + velocity * grid, color="black", lw=1)
            axis.axhline(0, color="0.7", lw=0.8)
            axis.set_ylabel(label)
        axes[0].legend(loc="best")
        axes[2].plot(hours, table.fitted_flux_snr, "o-", label="free-centroid fit")
        axes[2].plot(hours, table.forced_flux_snr_at_seed, "s--", label="forced at seed")
        axes[2].axhline(8, color="tab:red", ls=":", label="detection threshold")
        axes[2].set(ylabel="fitted flux S/N", xlabel="hours from series reference")
        axes[2].legend(loc="best")
        fig.suptitle(series_id)
        fig.tight_layout()
        output = args.input_dir / f"adaptive_substacks_{series_id}.png"
        fig.savefig(output, dpi=170)
        plt.close(fig)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
