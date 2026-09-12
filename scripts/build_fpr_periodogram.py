#!/usr/bin/env python3
"""Reproduce the short-window scalar AL period search used in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle


def robust_sigma(values) -> float:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - median)))


def densest_window(table: pd.DataFrame, width_days: float) -> pd.DataFrame:
    table = table.sort_values("epoch_utc_offset_jd").reset_index(drop=True)
    times = table["epoch_utc_offset_jd"].to_numpy(float)
    best = (0, 0, 0)
    for start in range(len(table)):
        stop = int(np.searchsorted(times, times[start] + width_days, side="right"))
        if stop - start > best[0]:
            best = (stop - start, start, stop)
    return table.iloc[best[1] : best[2]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=7065)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--window-days", type=float, default=10)
    parser.add_argument("--min-period-hours", type=float, default=3)
    parser.add_argument("--target-period-hours", type=float, default=94.864)
    parser.add_argument("--frequencies", type=int, default=100_000)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    stem = f"gaia_fpr_{args.number}"
    ccd = pd.read_csv(args.output_dir / f"{stem}_ccd_residuals.csv")
    good = ccd[ccd["good_for_analysis"]].copy()
    transit = good.groupby("transit_id").agg(
        epoch_utc_offset_jd=("epoch_utc", "mean"),
        residual_al_mas=("residual_al_mas", "mean"),
        sigma_al_empirical_mas=("residual_al_mas", "std"),
        n_ccd=("residual_al_mas", "size"),
    ).dropna().reset_index()
    window = densest_window(transit, args.window_days)
    span_days = float(window["epoch_utc_offset_jd"].max() - window["epoch_utc_offset_jd"].min())

    residual_median = window["residual_al_mas"].median()
    residual_scale = robust_sigma(window["residual_al_mas"])
    error_median = window["sigma_al_empirical_mas"].median()
    error_scale = robust_sigma(window["sigma_al_empirical_mas"])
    keep = (
        (np.abs(window["residual_al_mas"] - residual_median) <= 3 * residual_scale)
        & (window["sigma_al_empirical_mas"] <= error_median + 3 * error_scale)
    )
    window = window[keep].copy()
    time = window["epoch_utc_offset_jd"].to_numpy(float)
    values = window["residual_al_mas"].to_numpy(float)
    errors = window["sigma_al_empirical_mas"].to_numpy(float)
    maximum_period_hours = 3.0 * span_days * 24.0
    frequency = np.linspace(
        24.0 / maximum_period_hours,
        24.0 / args.min_period_hours,
        args.frequencies,
    )
    model = LombScargle(time, values, errors, fit_mean=True, center_data=True)
    power = model.power(frequency, normalization="standard")
    periods = 24.0 / frequency
    weights = errors ** -2
    sampling_phase = np.exp(
        -2j * np.pi * (time - time.mean())[:, None] * frequency[None, :]
    )
    window_power = np.abs(
        np.sum(weights[:, None] * sampling_phase, axis=0) / np.sum(weights)
    ) ** 2
    best_index = int(np.argmax(power))
    target_power = float(model.power(24.0 / args.target_period_hours))
    best_parameters = model.model_parameters(frequency[best_index])
    target_parameters = model.model_parameters(24.0 / args.target_period_hours)
    bootstrap_fap = float(
        model.false_alarm_probability(
            float(power[best_index]),
            minimum_frequency=float(frequency.min()),
            maximum_frequency=float(frequency.max()),
            method="bootstrap",
            method_kwds={"n_bootstraps": args.bootstrap, "random_seed": args.number},
        )
    )
    periodogram = pd.DataFrame(
        {"period_hours": periods, "power": power, "spectral_window": window_power}
    )
    periodogram = periodogram.sort_values("period_hours").reset_index(drop=True)

    window_path = args.output_dir / f"{stem}_period_window.csv"
    table_path = args.output_dir / f"{stem}_periodogram.csv"
    figure_path = args.output_dir / f"{stem}_periodogram.png"
    summary_path = args.output_dir / f"{stem}_periodogram_summary.json"
    window.to_csv(window_path, index=False)
    periodogram.to_csv(table_path, index=False)

    fig, (axis_data, axis_power) = plt.subplots(2, 1, figsize=(9, 7))
    relative_hours = (time - time.min()) * 24.0
    axis_data.errorbar(relative_hours, values, errors, fmt="o", capsize=2)
    axis_data.set(xlabel="Время от начала окна, ч", ylabel="AL O−C, mas")
    axis_power.plot(periods, power, lw=1)
    axis_power.plot(periods, window_power, color="0.6", lw=0.8, label="спектральное окно")
    axis_power.axvline(args.target_period_hours, color="tab:red", ls="--", label="94.864 ч")
    axis_power.axvline(periods[best_index], color="black", ls=":", label="максимум")
    axis_power.set(xlabel="Период, ч", ylabel="GLSP power", xlim=(args.min_period_hours, maximum_period_hours))
    axis_power.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "transits_in_window": len(window),
        "window_span_days": span_days,
        "minimum_period_hours": args.min_period_hours,
        "maximum_period_hours": maximum_period_hours,
        "best_period_hours": float(periods[best_index]),
        "best_power": float(power[best_index]),
        "best_amplitude_mas": float(np.hypot(best_parameters[-2], best_parameters[-1])),
        "global_bootstrap_false_alarm_probability": bootstrap_fap,
        "bootstrap_trials": args.bootstrap,
        "target_period_hours": args.target_period_hours,
        "target_power": target_power,
        "target_amplitude_mas": float(
            np.hypot(target_parameters[-2], target_parameters[-1])
        ),
        "false_alarm_probability_calibrated": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
