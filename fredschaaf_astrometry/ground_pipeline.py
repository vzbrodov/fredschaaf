"""Batch orchestration for the two-zone ground-astrometry reduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS
from scipy.ndimage import shift as image_shift

from .ground import (
    MAS_PER_DEGREE,
    centroid_star,
    combine_psf_stamps,
    exposure_middle_time,
    fit_affine_model,
    fit_empirical_psf_centroid,
    fit_gaussian_centroid,
    inverse_tangent_plane,
    linear_ephemeris_coordinates,
    normalized_star_stamp,
    pixel_offset_to_mas,
    propagate_gaia_linear_motion,
    read_fits_frame,
    robust_upper_limit,
    tangent_plane,
)
from .series import MotionFit, SeriesConfig, fit_series_motion


@dataclass
class FrameModel:
    """In-memory astrometric solution and measurements for one CCD frame."""

    path: Path
    time: Time
    image: np.ndarray
    coefficients: np.ndarray
    correction_deg: np.ndarray
    asteroid_ra_deg: float
    asteroid_dec_deg: float
    stars: pd.DataFrame


@dataclass
class StackGroup:
    """One independent shift-and-add target stack and its empirical PSF."""

    frame_indices: list[int]
    image: np.ndarray
    middle_jd: float
    span_minutes: float
    psf: np.ndarray | None = None
    psf_star_images: int = 0


@dataclass
class ReductionProducts:
    """Tables and retained images produced by :func:`reduce_series`."""

    frame_models: list[FrameModel]
    stack_groups: list[StackGroup]
    frame_quality: pd.DataFrame
    control_oc: pd.DataFrame
    distortion_samples: pd.DataFrame
    stacks: pd.DataFrame
    comparison: pd.DataFrame
    summary: pd.DataFrame
    gaussian_fit: MotionFit
    psf_fit: MotionFit


def _config_path(config: SeriesConfig, key: str) -> Path:
    return (config.path.parent / str(config.reduction[key])).resolve()


def _frame_paths(config: SeriesConfig) -> list[Path]:
    pattern = config.path.parent / str(config.reduction["frame_glob"])
    paths = sorted(pattern.parent.glob(pattern.name))
    expected = int(config.series["expected_frames"])
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} frames, found {len(paths)}")
    return paths


def _ephemeris(config: SeriesConfig, time: Time) -> tuple[float, float]:
    values = config.ephemeris
    ra, dec = linear_ephemeris_coordinates(
        time,
        reference_time=Time(str(values["reference_utc"]), scale="utc"),
        reference_ra_deg=float(values["ra_deg"]),
        reference_dec_deg=float(values["dec_deg"]),
        ra_rate_arcsec_per_hour=float(values["ra_rate_arcsec_per_hour"]),
        dec_rate_arcsec_per_hour=float(values["dec_rate_arcsec_per_hour"]),
    )
    return float(ra), float(dec)


def _tangent(config: SeriesConfig, ra, dec) -> np.ndarray:
    return tangent_plane(
        ra,
        dec,
        origin_ra_deg=float(config.ephemeris["ra_deg"]),
        origin_dec_deg=float(config.ephemeris["dec_deg"]),
    )


def _inside_detector(predicted: np.ndarray, shape, margin: int) -> np.ndarray:
    return (
        (predicted[:, 0] > margin)
        & (predicted[:, 0] < shape[1] - margin)
        & (predicted[:, 1] > margin)
        & (predicted[:, 1] < shape[0] - margin)
    )


def _measure_centroids(
    image: np.ndarray,
    predicted: np.ndarray,
    indices: np.ndarray,
    *,
    half_box: int,
) -> np.ndarray:
    """Measure a possibly empty star selection with a stable ``(n, 2)`` shape."""
    return np.asarray(
        [
            centroid_star(image, *predicted[index], half_box=half_box)
            for index in indices
        ],
        dtype=float,
    ).reshape(-1, 2)


def _reduce_frames(
    config: SeriesConfig,
    paths: list[Path],
    catalog: pd.DataFrame,
    extra_catalog: pd.DataFrame,
) -> tuple[list[FrameModel], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reduction = config.reduction
    inner_radius = float(reduction["inner_radius_arcmin"])
    outer_radius = float(reduction["outer_radius_arcmin"])
    margin = int(reduction["detector_margin_px"])
    frame_models: list[FrameModel] = []
    quality_rows = []
    control_rows = []
    distortion_rows = []

    for frame_number, path in enumerate(paths):
        image, header = read_fits_frame(path)
        time = exposure_middle_time(header)
        wcs = WCS(header)
        epoch_catalog = propagate_gaia_linear_motion(catalog, time)
        asteroid_ra, asteroid_dec = _ephemeris(config, time)
        predicted = wcs.all_world2pix(
            epoch_catalog[["ra_epoch", "dec_epoch"]].to_numpy(), 0
        )
        separation = SkyCoord(
            epoch_catalog.ra_epoch, epoch_catalog.dec_epoch, unit="deg"
        ).separation(SkyCoord(asteroid_ra, asteroid_dec, unit="deg")).arcmin
        inside = _inside_detector(predicted, image.shape, margin)
        inner = inside & (separation < inner_radius)
        outer = inside & (separation >= inner_radius) & (separation < outer_radius)

        catalog_indices = np.where(inner | outer)[0]
        measured_xy = _measure_centroids(
            image,
            predicted,
            catalog_indices,
            half_box=int(reduction["star_half_box_px"]),
        )
        measured_ok = np.isfinite(measured_xy).all(axis=1)
        catalog_indices = catalog_indices[measured_ok]
        measured_xy = measured_xy[measured_ok]
        is_inner = inner[catalog_indices]
        is_outer = outer[catalog_indices]
        if not np.any(is_inner):
            raise ValueError(f"No inner-zone control stars measured in {path.name}")

        outer_true = _tangent(
            config,
            epoch_catalog.ra_epoch.to_numpy()[catalog_indices[is_outer]],
            epoch_catalog.dec_epoch.to_numpy()[catalog_indices[is_outer]],
        )
        coefficients, outer_residual, fit_good = fit_affine_model(
            measured_xy[is_outer],
            outer_true,
            iterations=int(reduction["affine_clip_iterations"]),
            clip_sigma=float(reduction["affine_clip_sigma"]),
            minimum_stars=int(reduction["affine_min_stars"]),
            minimum_clip_radius_mas=float(reduction["affine_min_clip_radius_mas"]),
        )

        inner_design = np.column_stack((np.ones(is_inner.sum()), measured_xy[is_inner]))
        inner_observed = inner_design @ coefficients
        inner_catalog = _tangent(
            config,
            epoch_catalog.ra_epoch.to_numpy()[catalog_indices[is_inner]],
            epoch_catalog.dec_epoch.to_numpy()[catalog_indices[is_inner]],
        )
        inner_oc_mas = (inner_observed - inner_catalog) * MAS_PER_DEGREE
        local_correction_deg = inner_oc_mas.mean(axis=0) / MAS_PER_DEGREE
        all_design = np.column_stack((np.ones(len(measured_xy)), measured_xy))
        all_catalog = _tangent(
            config,
            epoch_catalog.ra_epoch.to_numpy()[catalog_indices],
            epoch_catalog.dec_epoch.to_numpy()[catalog_indices],
        )
        all_oc_mas = (all_design @ coefficients - all_catalog) * MAS_PER_DEGREE
        outer_rms = np.sqrt(np.mean(np.sum(outer_residual[fit_good] ** 2, axis=1)))
        inner_rms_before = np.sqrt(np.mean(np.sum(inner_oc_mas**2, axis=1)))
        centered_inner_oc = inner_oc_mas - inner_oc_mas.mean(axis=0)
        inner_rms_after = np.sqrt(np.mean(np.sum(centered_inner_oc**2, axis=1)))
        measured_stars = pd.DataFrame(
            {
                "source_id": epoch_catalog.source_id.to_numpy()[catalog_indices],
                "x": measured_xy[:, 0],
                "y": measured_xy[:, 1],
                "gmag": epoch_catalog.phot_g_mean_mag.to_numpy()[catalog_indices],
                "separation_arcmin": separation[catalog_indices],
            }
        )
        frame_models.append(
            FrameModel(
                path=path,
                time=time,
                image=image,
                coefficients=coefficients,
                correction_deg=local_correction_deg,
                asteroid_ra_deg=asteroid_ra,
                asteroid_dec_deg=asteroid_dec,
                stars=measured_stars,
            )
        )
        quality_rows.append(
            (
                frame_number,
                path.name,
                is_outer.sum(),
                fit_good.sum(),
                is_inner.sum(),
                outer_rms,
                inner_rms_before,
                inner_rms_after,
                *inner_oc_mas.mean(axis=0),
            )
        )
        for source_id, oc in zip(
            epoch_catalog.source_id.to_numpy()[catalog_indices[is_inner]],
            inner_oc_mas,
        ):
            control_rows.append((frame_number, int(source_id), oc[0], oc[1]))
        outer_fit_used = np.zeros(len(measured_xy), dtype=bool)
        outer_fit_used[np.where(is_outer)[0][fit_good]] = True
        for star_index, oc in enumerate(all_oc_mas):
            distortion_rows.append(
                (
                    frame_number,
                    path.name,
                    int(epoch_catalog.source_id.to_numpy()[catalog_indices[star_index]]),
                    measured_xy[star_index, 0],
                    measured_xy[star_index, 1],
                    oc[0],
                    oc[1],
                    bool(is_inner[star_index]),
                    bool(is_outer[star_index]),
                    bool(outer_fit_used[star_index]),
                )
            )

        faint = propagate_gaia_linear_motion(extra_catalog, time)
        faint_predicted = wcs.all_world2pix(faint[["ra_epoch", "dec_epoch"]].to_numpy(), 0)
        faint_indices = np.where(_inside_detector(faint_predicted, image.shape, margin))[0]
        faint_measured = _measure_centroids(
            image,
            faint_predicted,
            faint_indices,
            half_box=int(reduction["star_half_box_px"]),
        )
        faint_ok = np.isfinite(faint_measured).all(axis=1)
        faint_indices = faint_indices[faint_ok]
        faint_measured = faint_measured[faint_ok]
        faint_design = np.column_stack((np.ones(len(faint_measured)), faint_measured))
        faint_catalog_xy = _tangent(
            config,
            faint.ra_epoch.to_numpy()[faint_indices],
            faint.dec_epoch.to_numpy()[faint_indices],
        )
        faint_oc_mas = (faint_design @ coefficients - faint_catalog_xy) * MAS_PER_DEGREE
        faint_separation = SkyCoord(
            faint.ra_epoch.to_numpy()[faint_indices],
            faint.dec_epoch.to_numpy()[faint_indices],
            unit="deg",
        ).separation(SkyCoord(asteroid_ra, asteroid_dec, unit="deg")).arcmin
        for star_index, oc in enumerate(faint_oc_mas):
            separation_value = faint_separation[star_index]
            distortion_rows.append(
                (
                    frame_number,
                    path.name,
                    int(faint.source_id.to_numpy()[faint_indices[star_index]]),
                    faint_measured[star_index, 0],
                    faint_measured[star_index, 1],
                    oc[0],
                    oc[1],
                    bool(separation_value < inner_radius),
                    bool(inner_radius <= separation_value < outer_radius),
                    False,
                )
            )

    quality = pd.DataFrame(
        quality_rows,
        columns=[
            "frame",
            "filename",
            "outer_measured",
            "outer_used",
            "inner",
            "outer_rms_mas",
            "inner_rms_before_mas",
            "inner_rms_after_mas",
            "correction_xi_mas",
            "correction_eta_mas",
        ],
    )
    quality["accepted"] = quality.outer_rms_mas <= robust_upper_limit(
        quality.outer_rms_mas,
        sigma=float(reduction["frame_rejection_sigma"]),
    )
    control = pd.DataFrame(
        control_rows, columns=["frame", "source_id", "oc_xi_mas", "oc_eta_mas"]
    )
    distortion = pd.DataFrame(
        distortion_rows,
        columns=[
            "frame",
            "filename",
            "source_id",
            "x",
            "y",
            "oc_xi_mas",
            "oc_eta_mas",
            "is_inner",
            "is_outer",
            "outer_fit_used",
        ],
    )
    return frame_models, quality, control, distortion


def _build_stacks(
    config: SeriesConfig,
    frame_models: list[FrameModel],
    quality: pd.DataFrame,
) -> list[StackGroup]:
    reduction = config.reduction
    half = int(reduction["stack_half_size_px"])
    cutouts: list[np.ndarray | None] = [None] * len(frame_models)
    for frame_number, model in enumerate(frame_models):
        if not quality.accepted.iloc[frame_number]:
            continue
        expected_tangent = _tangent(
            config, [model.asteroid_ra_deg], [model.asteroid_dec_deg]
        )[0]
        expected_tangent += model.correction_deg
        asteroid_xy = np.linalg.solve(
            model.coefficients[1:].T,
            expected_tangent - model.coefficients[0],
        )
        ix, iy = np.round(asteroid_xy).astype(int)
        cutout = model.image[iy - half : iy + half + 1, ix - half : ix + half + 1]
        _, background, _ = sigma_clipped_stats(
            cutout, sigma=float(reduction["background_clip_sigma"])
        )
        cutouts[frame_number] = image_shift(
            cutout - background,
            (-(asteroid_xy[1] - iy), -(asteroid_xy[0] - ix)),
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )

    groups = []
    group_size = int(reduction["group_size"])
    minimum_frames = int(config.quality["min_frames_per_stack"])
    if "max_inter_frame_gap_minutes" in reduction:
        accepted = [
            index for index, value in enumerate(cutouts) if value is not None
        ]
        maximum_gap = float(reduction["max_inter_frame_gap_minutes"])
        segments: list[list[int]] = []
        for index in accepted:
            if not segments:
                segments.append([index])
                continue
            gap = (
                frame_models[index].time
                - frame_models[segments[-1][-1]].time
            ).to_value("min")
            if gap > maximum_gap:
                segments.append([index])
            else:
                segments[-1].append(index)
        index_groups = []
        for segment in segments:
            if len(segment) < minimum_frames:
                raise ValueError(
                    f"Contiguous segment has only {len(segment)} accepted frames"
                )
            number = int(np.ceil(len(segment) / group_size))
            index_groups.extend(
                [list(map(int, part)) for part in np.array_split(segment, number)]
            )
    else:
        index_groups = [
            [index for index in range(start, min(start + group_size, len(cutouts)))
             if cutouts[index] is not None]
            for start in range(0, len(cutouts), group_size)
        ]
    for group_number, indices in enumerate(index_groups, start=1):
        if len(indices) < minimum_frames:
            raise ValueError(f"Stack {group_number} has only {len(indices)} frames")
        cube = np.stack([cutouts[index] for index in indices])
        stack = np.ma.mean(
            sigma_clip(cube, sigma=float(reduction["stack_clip_sigma"]), axis=0),
            axis=0,
        ).filled(np.nan)
        groups.append(
            StackGroup(
                frame_indices=indices,
                image=stack,
                middle_jd=float(np.mean([frame_models[index].time.jd for index in indices])),
                span_minutes=float(
                    (frame_models[indices[-1]].time - frame_models[indices[0]].time).to_value(
                        "min"
                    )
                ),
            )
        )
    return groups


def _measure_stacks(
    config: SeriesConfig,
    frame_models: list[FrameModel],
    groups: list[StackGroup],
) -> pd.DataFrame:
    reduction = config.reduction
    half = int(reduction["psf_half_size_px"])
    rows = []
    for group_number, group in enumerate(groups, start=1):
        stamps = []
        for frame_number in group.frame_indices:
            model = frame_models[frame_number]
            selected = model.stars[
                model.stars.gmag.between(
                    float(reduction["psf_mag_min"]), float(reduction["psf_mag_max"])
                )
                & (model.stars.separation_arcmin < float(reduction["psf_radius_arcmin"]))
            ]
            for star in selected.itertuples():
                stamp = normalized_star_stamp(
                    model.image,
                    star.x,
                    star.y,
                    half_size=half,
                    aperture_radius_px=float(reduction["psf_aperture_radius_px"]),
                    minimum_flux=float(reduction["psf_min_flux"]),
                    saturation_level=float(reduction["psf_saturation_level"]),
                )
                if stamp is not None:
                    stamps.append(stamp)
        group.psf = combine_psf_stamps(
            stamps, sigma=float(reduction["psf_clip_sigma"])
        )
        group.psf_star_images = len(stamps)
        gaussian_dx, gaussian_dy, _, _ = fit_gaussian_centroid(
            group.image,
            stack_center=int(reduction["stack_half_size_px"]),
            fit_half_size=half,
            sky_radius_px=float(reduction["psf_aperture_radius_px"]),
        )
        psf_dx, psf_dy, _, _ = fit_empirical_psf_centroid(
            group.image,
            group.psf,
            stack_center=int(reduction["stack_half_size_px"]),
            fit_half_size=half,
            sky_radius_px=float(reduction["psf_aperture_radius_px"]),
            maximum_offset_px=float(reduction["psf_fit_max_offset_px"]),
        )
        coefficients = [frame_models[index].coefficients for index in group.frame_indices]
        gaussian_oc = pixel_offset_to_mas(coefficients, gaussian_dx, gaussian_dy)
        psf_oc = pixel_offset_to_mas(coefficients, psf_dx, psf_dy)
        center = int(reduction["stack_half_size_px"])
        y, x = np.indices(group.image.shape)
        background_mask = np.hypot(x - center, y - center) > 10
        _, background, noise = sigma_clipped_stats(
            group.image[background_mask], sigma=float(reduction["background_clip_sigma"])
        )
        peak_snr = (np.nanmax(group.image) - background) / noise
        rows.append(
            (
                group_number,
                len(group.frame_indices),
                group.middle_jd,
                group.span_minutes,
                *gaussian_oc,
                *psf_oc,
                peak_snr,
                group.psf_star_images,
            )
        )
    stacks = pd.DataFrame(
        rows,
        columns=[
            "group",
            "frames",
            "middle_jd",
            "span_minutes",
            "gaussian_xi_mas",
            "gaussian_eta_mas",
            "psf_xi_mas",
            "psf_eta_mas",
            "snr",
            "psf_star_images",
        ],
    )
    stacks["middle_utc"] = Time(stacks.middle_jd, format="jd").isot
    return stacks


def _summary_tables(
    config: SeriesConfig, stacks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, MotionFit, MotionFit]:
    gaussian = fit_series_motion(stacks, prefix="gaussian")
    psf = fit_series_motion(stacks, prefix="psf")
    comparison = pd.DataFrame(
        [
            (
                "Свободная гауссиана",
                gaussian.rms_mas[0],
                gaussian.rms_mas[1],
                gaussian.rms_2d_mas,
                gaussian.rms_2d_adjusted_mas,
            ),
            (
                "Звёздная PSF",
                psf.rms_mas[0],
                psf.rms_mas[1],
                psf.rms_2d_mas,
                psf.rms_2d_adjusted_mas,
            ),
        ],
        columns=[
            "model",
            "RMS_xi_mas",
            "RMS_eta_mas",
            "RMS_2D_mas",
            "RMS_2D_n_minus_2_mas",
        ],
    )
    central_time = Time(psf.reference_jd, format="jd", scale="utc")
    ephemeris_ra, ephemeris_dec = _ephemeris(config, central_time)
    ephemeris_tangent = _tangent(config, [ephemeris_ra], [ephemeris_dec])[0]
    measured_tangent = ephemeris_tangent + psf.intercept_mas / MAS_PER_DEGREE
    measured_ra, measured_dec = inverse_tangent_plane(
        *measured_tangent,
        origin_ra_deg=float(config.ephemeris["ra_deg"]),
        origin_dec_deg=float(config.ephemeris["dec_deg"]),
    )
    summary = pd.DataFrame(
        {
            "quantity": [
                "central_utc",
                "RA_deg",
                "Dec_deg",
                "O-C_xi_mas",
                "O-C_eta_mas",
                "error_xi_mas",
                "error_eta_mas",
                "RMS_xi_mas",
                "RMS_eta_mas",
                "RMS_2D_mas",
                "RMS_2D_adjusted_mas",
                "median_stack_SNR",
            ],
            "value": [
                central_time.isot,
                measured_ra,
                measured_dec,
                psf.intercept_mas[0],
                psf.intercept_mas[1],
                np.sqrt(psf.intercept_covariance_mas2[0, 0]),
                np.sqrt(psf.intercept_covariance_mas2[1, 1]),
                psf.rms_mas[0],
                psf.rms_mas[1],
                psf.rms_2d_mas,
                psf.rms_2d_adjusted_mas,
                stacks.snr.median(),
            ],
        }
    )
    return comparison, summary, gaussian, psf


def reduce_series(config: SeriesConfig) -> ReductionProducts:
    """Run the complete two-zone shift-and-add reduction for one series."""
    reduction = config.reduction
    catalog_all = pd.read_csv(_config_path(config, "catalog")).dropna(
        subset=["ra", "dec", "pmra", "pmdec", "ref_epoch"]
    )
    bright = catalog_all[
        catalog_all.phot_g_mean_mag.between(
            float(reduction["astrometric_mag_min"]),
            float(reduction["astrometric_mag_max"]),
        )
    ].copy()
    faint = catalog_all[
        catalog_all.phot_g_mean_mag.gt(float(reduction["astrometric_mag_max"]))
        & catalog_all.phot_g_mean_mag.le(float(reduction["distortion_mag_max"]))
    ].copy()
    frame_models, quality, control, distortion = _reduce_frames(
        config, _frame_paths(config), bright, faint
    )
    groups = _build_stacks(config, frame_models, quality)
    stacks = _measure_stacks(config, frame_models, groups)
    comparison, summary, gaussian, psf = _summary_tables(config, stacks)
    return ReductionProducts(
        frame_models=frame_models,
        stack_groups=groups,
        frame_quality=quality,
        control_oc=control,
        distortion_samples=distortion,
        stacks=stacks,
        comparison=comparison,
        summary=summary,
        gaussian_fit=gaussian,
        psf_fit=psf,
    )


def write_reduction_outputs(
    products: ReductionProducts, config: SeriesConfig, output_dir: str | Path
) -> dict[str, Path]:
    """Write the six stable tabular products of a series reduction."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    series_id = str(config.series["id"])
    tables = {
        "frame_quality": products.frame_quality,
        "control_oc": products.control_oc,
        "distortion_samples": products.distortion_samples,
        "asteroid_stacks": products.stacks,
        "model_comparison": products.comparison,
        "result": products.summary,
    }
    paths = {}
    for name, table in tables.items():
        path = destination / f"two_zone_{name}_{series_id}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths
