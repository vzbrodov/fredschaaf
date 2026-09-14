"""Configuration and common table representation for ground-based series."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib

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


@dataclass(frozen=True)
class MotionFit:
    """Joint linear fit of paired tangent-plane positions."""

    reference_jd: float
    intercept_mas: np.ndarray
    velocity_mas_per_minute: np.ndarray
    intercept_covariance_mas2: np.ndarray
    residual_covariance_mas2: np.ndarray
    residuals_mas: np.ndarray
    rms_mas: np.ndarray
    rms_2d_mas: float
    rms_2d_adjusted_mas: float
    degrees_of_freedom: int


@dataclass(frozen=True)
class SeriesConfig:
    """Validated, path-resolved configuration for one observing series."""

    path: Path
    series: dict[str, object]
    calibration: dict[str, bool]
    inputs: dict[str, Path]
    reduction: dict[str, object]
    ephemeris: dict[str, object]
    quality: dict[str, float | int]


def _required(mapping: dict[str, object], keys: set[str], section: str) -> None:
    missing = sorted(keys.difference(mapping))
    if missing:
        raise ValueError(f"Missing [{section}] keys: {missing}")


def load_series_config(path: str | Path) -> SeriesConfig:
    """Read and validate a TOML configuration for one observing series.

    Input paths are resolved relative to the configuration file, making a run
    independent of the current working directory.
    """
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    for section in (
        "series",
        "calibration",
        "inputs",
        "reduction",
        "ephemeris",
        "quality",
    ):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"Missing or invalid [{section}] section")

    series = dict(raw["series"])
    calibration = dict(raw["calibration"])
    inputs_raw = dict(raw["inputs"])
    reduction = dict(raw["reduction"])
    ephemeris = dict(raw["ephemeris"])
    quality = dict(raw["quality"])
    _required(
        series,
        {"id", "source", "observatory_code", "filter", "expected_frames"},
        "series",
    )
    _required(calibration, {"bias", "dark", "flat"}, "calibration")
    if not all(
        isinstance(calibration[key], bool) for key in ("bias", "dark", "flat")
    ):
        raise ValueError("Calibration flags must be booleans")
    _required(inputs_raw, {"summary", "stacks", "frame_quality"}, "inputs")
    _required(
        reduction,
        {
            "frame_glob",
            "catalog",
            "inner_radius_arcmin",
            "outer_radius_arcmin",
            "group_size",
            "detector_margin_px",
            "astrometric_mag_min",
            "astrometric_mag_max",
            "distortion_mag_max",
            "star_half_box_px",
            "stack_half_size_px",
            "psf_half_size_px",
            "psf_radius_arcmin",
            "psf_mag_min",
            "psf_mag_max",
            "affine_clip_iterations",
            "affine_clip_sigma",
            "affine_min_stars",
            "affine_min_clip_radius_mas",
            "frame_rejection_sigma",
            "psf_aperture_radius_px",
            "psf_min_flux",
            "psf_saturation_level",
            "psf_fit_max_offset_px",
            "psf_clip_sigma",
            "stack_clip_sigma",
            "background_clip_sigma",
        },
        "reduction",
    )
    _required(
        ephemeris,
        {
            "id",
            "reference_utc",
            "ra_deg",
            "dec_deg",
            "ra_rate_arcsec_per_hour",
            "dec_rate_arcsec_per_hour",
        },
        "ephemeris",
    )
    _required(
        quality,
        {
            "min_stacks",
            "min_frames_per_stack",
            "min_median_stack_snr",
            "max_rejected_fraction",
            "max_adjusted_rms_mas",
        },
        "quality",
    )

    inner = float(reduction["inner_radius_arcmin"])
    outer = float(reduction["outer_radius_arcmin"])
    if not 0 < inner < outer:
        raise ValueError("Reduction radii must satisfy 0 < inner < outer")
    if int(reduction["group_size"]) < 3:
        raise ValueError("reduction.group_size must be at least 3")
    if (
        "max_inter_frame_gap_minutes" in reduction
        and float(reduction["max_inter_frame_gap_minutes"]) <= 0
    ):
        raise ValueError("reduction.max_inter_frame_gap_minutes must be positive")
    if int(series["expected_frames"]) < 1:
        raise ValueError("series.expected_frames must be positive")
    for key in (
        "star_half_box_px",
        "stack_half_size_px",
        "psf_half_size_px",
        "affine_clip_iterations",
        "affine_min_stars",
    ):
        if int(reduction[key]) < 1:
            raise ValueError(f"reduction.{key} must be positive")
    if float(reduction["psf_mag_min"]) >= float(reduction["psf_mag_max"]):
        raise ValueError("reduction PSF magnitude limits are reversed")
    if not 0 <= float(quality["max_rejected_fraction"]) <= 1:
        raise ValueError("quality.max_rejected_fraction must be between 0 and 1")

    inputs = {
        name: (config_path.parent / str(value)).resolve()
        for name, value in inputs_raw.items()
    }
    return SeriesConfig(
        path=config_path,
        series=series,
        calibration=calibration,
        inputs=inputs,
        reduction=reduction,
        ephemeris=ephemeris,
        quality=quality,
    )


def fit_series_motion(stacks: pd.DataFrame, *, prefix: str = "psf") -> MotionFit:
    """Fit both coordinates jointly and retain their empirical covariance.

    The same design matrix is used for xi and eta. The 2x2 covariance of the
    paired residuals supplies the cross term for the mean-epoch position.
    """
    required = {"middle_jd", f"{prefix}_xi_mas", f"{prefix}_eta_mas"}
    missing = sorted(required.difference(stacks.columns))
    if missing:
        raise ValueError(f"Missing stack columns: {missing}")
    values = stacks[[f"{prefix}_xi_mas", f"{prefix}_eta_mas"]].to_numpy(float)
    times = stacks["middle_jd"].to_numpy(float)
    valid = np.isfinite(times) & np.all(np.isfinite(values), axis=1)
    times = times[valid]
    values = values[valid]
    if len(times) < 3 or np.unique(times).size < 3:
        raise ValueError("At least three finite stacks at distinct epochs are required")

    reference_jd = float(np.mean(times))
    minutes = (times - reference_jd) * 1440.0
    design = np.column_stack((np.ones(len(times)), minutes))
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("Stack epochs do not support a linear motion fit")
    line_parameters = []
    scalar_covariances = []
    residual_columns = []
    for column in range(2):
        line, covariance = np.polyfit(minutes, values[:, column], 1, cov=True)
        line_parameters.append(line)
        scalar_covariances.append(covariance)
        residual_columns.append(values[:, column] - np.polyval(line, minutes))
    residuals = np.column_stack(residual_columns)
    degrees_of_freedom = len(times) - 2
    residual_covariance = residuals.T @ residuals / degrees_of_freedom
    design_covariance = np.linalg.inv(design.T @ design)
    intercept_covariance = residual_covariance * design_covariance[0, 0]
    intercept_covariance[0, 0] = scalar_covariances[0][1, 1]
    intercept_covariance[1, 1] = scalar_covariances[1][1, 1]
    rms = np.sqrt(np.mean(residuals**2, axis=0))
    line_parameters = np.asarray(line_parameters)

    return MotionFit(
        reference_jd=reference_jd,
        intercept_mas=line_parameters[:, 1],
        velocity_mas_per_minute=line_parameters[:, 0],
        intercept_covariance_mas2=intercept_covariance,
        residual_covariance_mas2=residual_covariance,
        residuals_mas=residuals,
        rms_mas=rms,
        rms_2d_mas=float(np.hypot(*rms)),
        rms_2d_adjusted_mas=float(np.sqrt(np.sum(residuals**2) / degrees_of_freedom)),
        degrees_of_freedom=degrees_of_freedom,
    )


def load_ground_series_result(
    path: str | Path,
    *,
    series_id: str,
    filter_name: str,
    observatory_code: str = "217",
    source: str = "ground",
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
        "source": source,
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
        # A legacy summary fitted xi and eta independently. NaN makes the
        # missing covariance explicit until paired stack residuals are loaded.
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


def build_ground_series(config: SeriesConfig) -> pd.DataFrame:
    """Build a covariance-complete, quality-tagged row from reduction outputs."""
    stacks = pd.read_csv(config.inputs["stacks"])
    frames = pd.read_csv(config.inputs["frame_quality"])
    if frames.empty:
        raise ValueError("Frame quality table is empty")
    result = load_ground_series_result(
        config.inputs["summary"],
        series_id=str(config.series["id"]),
        filter_name=str(config.series["filter"]),
        observatory_code=str(config.series["observatory_code"]),
        source=str(config.series["source"]),
    )
    fit = fit_series_motion(stacks)

    epoch_difference_ms = (
        abs(float(result.loc[0, "jd_utc"]) - fit.reference_jd) * 86_400_000
    )
    if epoch_difference_ms > 2.0:
        raise ValueError(
            "Summary central epoch differs from the mean stack epoch by "
            f"{epoch_difference_ms:.3f} ms"
        )
    summary_intercept = result.loc[0, ["oc_xi_mas", "oc_eta_mas"]].to_numpy(float)
    if not np.allclose(summary_intercept, fit.intercept_mas, atol=1e-3, rtol=0):
        difference = float(np.max(np.abs(summary_intercept - fit.intercept_mas)))
        raise ValueError(
            "Summary O-C is inconsistent with the paired stack fit: "
            f"maximum difference is {difference:.6g} mas"
        )

    covariance = fit.intercept_covariance_mas2
    result.loc[0, "cov_xi_xi_mas2"] = covariance[0, 0]
    result.loc[0, "cov_xi_eta_mas2"] = covariance[0, 1]
    result.loc[0, "cov_eta_eta_mas2"] = covariance[1, 1]
    result.loc[0, "ephemeris_id"] = str(config.ephemeris["id"])
    result.loc[0, "covariance_method"] = "joint-linear-stack-residuals"
    result.loc[0, "velocity_xi_mas_per_minute"] = fit.velocity_mas_per_minute[0]
    result.loc[0, "velocity_eta_mas_per_minute"] = fit.velocity_mas_per_minute[1]
    result.loc[0, "rms_xi_mas"] = fit.rms_mas[0]
    result.loc[0, "rms_eta_mas"] = fit.rms_mas[1]
    result.loc[0, "rms_2d_mas"] = fit.rms_2d_mas
    result.loc[0, "rms_2d_adjusted_mas"] = fit.rms_2d_adjusted_mas
    result["n_stacks"] = len(stacks)
    result["n_frames_total"] = len(frames)
    if "accepted" not in frames:
        raise ValueError("Frame quality table has no accepted column")
    accepted = (
        frames["accepted"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    if accepted.isna().any():
        raise ValueError("Frame accepted column must contain booleans")
    n_accepted = int(accepted.sum())
    result["n_frames_accepted"] = n_accepted
    result.loc[0, "rejected_fraction"] = 1.0 - n_accepted / len(frames)
    result.loc[0, "median_stack_snr"] = float(stacks["snr"].median())
    result["min_frames_per_stack"] = int(stacks["frames"].min())

    limits = config.quality
    checks = (
        (len(stacks) >= int(limits["min_stacks"]), "too_few_stacks"),
        (
            int(stacks["frames"].min()) >= int(limits["min_frames_per_stack"]),
            "too_few_frames_in_stack",
        ),
        (
            float(stacks["snr"].median()) >= float(limits["min_median_stack_snr"]),
            "low_stack_snr",
        ),
        (
            float(result.loc[0, "rejected_fraction"])
            <= float(limits["max_rejected_fraction"]),
            "too_many_rejected_frames",
        ),
        (
            fit.rms_2d_adjusted_mas <= float(limits["max_adjusted_rms_mas"]),
            "large_astrometric_scatter",
        ),
    )
    failures = [label for passed, label in checks if not passed]
    result["quality_ok"] = not failures
    result.loc[0, "quality_flags"] = ";".join(failures) if failures else "ok"
    result.loc[0, "input_result"] = os.path.relpath(
        config.inputs["summary"], start=config.path.parent
    )
    result.loc[0, "config"] = config.path.name
    return result


def build_orbit_ground_series(
    full_stack_series: pd.DataFrame,
    replacements: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Select exactly one ground measurement per observing series.

    A specialized measurement (for example an appulse forward fit) replaces
    the full-stack measurement from the same frames.  It is never appended,
    because treating the two strongly correlated measurements as independent
    would count one observing series twice in an orbit fit.
    """
    required = {
        "series_id",
        "central_utc",
        "cov_xi_xi_mas2",
        "cov_xi_eta_mas2",
        "cov_eta_eta_mas2",
    }
    missing = sorted(required.difference(full_stack_series.columns))
    if missing:
        raise ValueError(f"Missing full-stack columns: {missing}")
    result = full_stack_series.copy()
    if result.series_id.astype(str).duplicated().any():
        raise ValueError("Full-stack table contains duplicate series_id values")

    for series_id, replacement in (replacements or {}).items():
        if len(replacement) != 1:
            raise ValueError(f"Replacement for {series_id} must contain exactly one row")
        if series_id not in set(result.series_id.astype(str)):
            raise ValueError(f"Replacement target is absent from full-stack table: {series_id}")
        missing = sorted(required.difference(replacement.columns))
        if missing:
            raise ValueError(f"Missing replacement columns for {series_id}: {missing}")
        row = replacement.copy()
        row.loc[row.index[0], "series_id"] = series_id
        for column in (name for name in row.columns if name.endswith("source_id")):
            row[column] = pd.array(row[column], dtype="Int64")
            if column not in result:
                result[column] = pd.array([pd.NA] * len(result), dtype="Int64")
            else:
                result[column] = pd.array(result[column], dtype="Int64")
        result = result.loc[result.series_id.astype(str) != series_id]
        result = pd.concat([result, row], ignore_index=True, sort=False)

    if "quality_ok" in result:
        quality = (
            result.quality_ok.astype(str).str.strip().str.lower()
            .map({"true": True, "false": False})
        )
        if quality.isna().any() or not quality.all():
            failed = result.loc[quality != True, "series_id"].astype(str).tolist()
            raise ValueError(f"Ground series failed quality checks: {failed}")
    if "jd_utc" not in result or result.jd_utc.isna().any():
        result["jd_utc"] = Time(result.central_utc.to_list(), scale="utc").jd
    # A union with rows that lack this field would otherwise convert 64-bit
    # Gaia identifiers to float and silently discard their low-order digits.
    for column in (name for name in result.columns if name.endswith("source_id")):
        result[column] = pd.array(result[column], dtype="Int64")
    for row in result.itertuples():
        covariance = np.array(
            [
                [row.cov_xi_xi_mas2, row.cov_xi_eta_mas2],
                [row.cov_xi_eta_mas2, row.cov_eta_eta_mas2],
            ],
            dtype=float,
        )
        if not np.all(np.linalg.eigvalsh(covariance) > 0):
            raise ValueError(f"Ground covariance is not positive definite: {row.series_id}")
    return result.sort_values("jd_utc").reset_index(drop=True)
