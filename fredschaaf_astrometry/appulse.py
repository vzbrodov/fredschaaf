"""Direct differential astrometry of an asteroid and a nearby Gaia star."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from scipy.ndimage import shift as image_shift
from scipy.optimize import least_squares

from .ground import (
    MAS_PER_DEGREE,
    combine_psf_stamps,
    fit_empirical_psf_centroid,
    linear_ephemeris_coordinates,
    mpc_217_location,
    normalized_star_stamp,
    propagate_gaia_astrometry,
    tangent_plane,
)
from .ground_pipeline import ReductionProducts
from .series import SeriesConfig, fit_series_motion


@dataclass(frozen=True)
class AppulseFit:
    """Joint trajectory fit with per-frame background and flux."""

    reference_jd: float
    position_mas: np.ndarray
    velocity_mas_per_minute: np.ndarray
    formal_covariance_mas2: np.ndarray
    frame_diagnostics: pd.DataFrame
    pseudo_stars: pd.DataFrame
    jackknife: pd.DataFrame
    systematics: pd.DataFrame
    psf_mode: str
    event_source_id: int
    event_separation_arcsec: float


def _linear_flux_and_background(data, template, noise, background_order=0):
    valid = np.isfinite(data) & np.isfinite(template)
    columns = [np.ones(valid.sum()), template[valid]]
    if background_order == 1:
        yy, xx = np.indices(data.shape, dtype=float)
        xx -= np.mean(xx)
        yy -= np.mean(yy)
        columns.extend((xx[valid], yy[valid]))
    elif background_order != 0:
        raise ValueError("background_order must be 0 or 1")
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, data[valid], rcond=None)[0]
    residual = (design @ coefficients - data[valid]) / noise
    covariance = noise**2 * np.linalg.inv(design.T @ design)
    return coefficients, covariance, residual


def _psf_shape(psf: np.ndarray) -> tuple[float, float]:
    """Return moment-based circularized FWHM and ellipticity."""
    positive = np.maximum(np.asarray(psf, dtype=float), 0)
    positive /= positive.sum()
    yy, xx = np.indices(positive.shape)
    x0 = np.sum(xx * positive)
    y0 = np.sum(yy * positive)
    covariance = np.array(
        [
            [np.sum((xx - x0) ** 2 * positive), np.sum((xx - x0) * (yy - y0) * positive)],
            [np.sum((xx - x0) * (yy - y0) * positive), np.sum((yy - y0) ** 2 * positive)],
        ]
    )
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    fwhm = 2.35482 * np.sqrt(np.mean(eigenvalues))
    ellipticity = 1.0 - np.sqrt(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] else 0.0
    return float(fwhm), float(ellipticity)


def _frame_psf(model, config: SeriesConfig, fallback: np.ndarray, event_source_id: int):
    """Build an independent stellar PSF for one frame or use the group PSF."""
    reduction = config.reduction
    selected = model.stars[
        model.stars.gmag.between(
            float(reduction["psf_mag_min"]), float(reduction["psf_mag_max"])
        )
        & (model.stars.separation_arcmin < float(reduction["psf_radius_arcmin"]))
        & (model.stars.source_id.astype("int64") != int(event_source_id))
    ]
    stamps = []
    for star in selected.itertuples():
        stamp = normalized_star_stamp(
            model.image,
            star.x,
            star.y,
            half_size=int(reduction["psf_half_size_px"]),
            aperture_radius_px=float(reduction["psf_aperture_radius_px"]),
            minimum_flux=float(reduction["psf_min_flux"]),
            saturation_level=float(reduction["psf_saturation_level"]),
        )
        if stamp is not None:
            stamps.append(stamp)
    minimum = int(reduction.get("appulse_min_frame_psf_stars", 3))
    if len(stamps) < minimum:
        return fallback, "group", len(stamps)
    return combine_psf_stamps(
        stamps, sigma=float(reduction["psf_clip_sigma"])
    ), "frame", len(stamps)


def _atmosphere_geometry(time: Time, ra_deg: float, dec_deg: float):
    """Return altitude, airmass and the tangent-plane direction to zenith."""
    frame = AltAz(obstime=time, location=mpc_217_location())
    target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    horizontal = target.transform_to(frame)
    zenith = SkyCoord(
        az=horizontal.az, alt=90 * u.deg, frame=frame
    ).transform_to("icrs")
    position_angle = target.position_angle(zenith).rad
    altitude = float(horizontal.alt.deg)
    airmass = float(1.0 / np.sin(horizontal.alt.rad)) if altitude > 0 else np.nan
    return altitude, airmass, float(np.sin(position_angle)), float(np.cos(position_angle))


def summarize_appulse_systematics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize seeing and zenith-directed residual diagnostics without correcting them."""
    valid = diagnostics[
        np.isfinite(diagnostics.local_residual_xi_mas)
        & np.isfinite(diagnostics.local_residual_eta_mas)
        & ~diagnostics.local_at_offset_boundary
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    along_zenith = (
        valid.local_residual_xi_mas * valid.zenith_xi_unit
        + valid.local_residual_eta_mas * valid.zenith_eta_unit
    )
    dcr_basis = np.sqrt(np.maximum(valid.airmass.to_numpy(float) ** 2 - 1.0, 0))
    minutes = (valid.jd.to_numpy(float) - valid.jd.mean()) * 1440.0
    time_dcr_correlation = float(np.corrcoef(minutes, dcr_basis)[0, 1])
    dcr_identifiable = bool(
        np.ptp(dcr_basis) >= 0.1 and abs(time_dcr_correlation) <= 0.95
    )
    if dcr_identifiable:
        design = np.column_stack((np.ones(len(valid)), minutes, dcr_basis))
        dcr_coefficient = float(
            np.linalg.lstsq(design, along_zenith, rcond=None)[0][2]
        )
    else:
        dcr_coefficient = np.nan
    return pd.DataFrame([{
        "usable_frames": len(valid),
        "frame_psf_frames": int((valid.psf_scope == "frame").sum()),
        "group_psf_fallback_frames": int((valid.psf_scope == "group").sum()),
        "median_psf_stars": float(valid.psf_star_images.median()),
        "median_psf_fwhm_px": float(valid.psf_fwhm_px.median()),
        "psf_fwhm_range_px": float(valid.psf_fwhm_px.max() - valid.psf_fwhm_px.min()),
        "airmass_min": float(valid.airmass.min()),
        "airmass_max": float(valid.airmass.max()),
        "dcr_time_correlation": time_dcr_correlation,
        "dcr_identifiable": dcr_identifiable,
        "local_residual_rms_xi_mas": float(np.sqrt(np.mean(valid.local_residual_xi_mas**2))),
        "local_residual_rms_eta_mas": float(np.sqrt(np.mean(valid.local_residual_eta_mas**2))),
        "dcr_coefficient_mas_per_tan_z": dcr_coefficient,
        "corr_zenith_residual_fwhm": float(along_zenith.corr(valid.psf_fwhm_px)),
        "corr_local_xi_outer_rms": float(
            valid.local_residual_xi_mas.corr(valid.outer_rms_mas)
        ),
        "corr_local_eta_outer_rms": float(
            valid.local_residual_eta_mas.corr(valid.outer_rms_mas)
        ),
    }])


def _pseudo_star_table(
    products: ReductionProducts,
    catalog: pd.DataFrame,
    event_source_id: int,
) -> pd.DataFrame:
    control = products.control_oc
    event = control.loc[
        control.source_id == event_source_id,
        ["frame", "oc_xi_mas", "oc_eta_mas"],
    ].rename(columns={"oc_xi_mas": "event_xi", "oc_eta_mas": "event_eta"})
    catalog_by_id = catalog.set_index("source_id")
    rows = []
    for source_id, values in control.groupby("source_id"):
        if int(source_id) == event_source_id:
            continue
        paired = values.merge(event, on="frame")
        if len(paired) < 3:
            continue
        stacks = pd.DataFrame(
            {
                "middle_jd": [products.frame_models[int(i)].time.jd for i in paired.frame],
                "psf_xi_mas": paired.oc_xi_mas - paired.event_xi,
                "psf_eta_mas": paired.oc_eta_mas - paired.event_eta,
            }
        )
        fit = fit_series_motion(stacks)
        star = catalog_by_id.loc[int(source_id)]
        rows.append(
            {
                "source_id": int(source_id),
                "frames": len(paired),
                "gmag": float(star.phot_g_mean_mag),
                "rms_xi_mas": fit.rms_mas[0],
                "rms_eta_mas": fit.rms_mas[1],
                "rms_2d_mas": fit.rms_2d_mas,
                "error_xi_mas": np.sqrt(fit.intercept_covariance_mas2[0, 0]),
                "error_eta_mas": np.sqrt(fit.intercept_covariance_mas2[1, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("rms_2d_mas").reset_index(drop=True)


def fit_appulse_trajectory(
    products: ReductionProducts,
    config: SeriesConfig,
    catalog: pd.DataFrame,
    *,
    event_source_id: int,
    initial_parameters: np.ndarray | None = None,
    psf_mode: str = "frame",
    compute_jackknife: bool = True,
) -> AppulseFit:
    """Fit all asteroid pixels relative to one Gaia event star.

    The global parameters are the two-coordinate position correction at the
    mean epoch and its velocity. Background and flux are solved independently
    and linearly in every frame for each trial trajectory.
    """
    if psf_mode not in {"frame", "group"}:
        raise ValueError("psf_mode must be 'frame' or 'group'")
    catalog_row = catalog.loc[catalog.source_id == event_source_id]
    if len(catalog_row) != 1:
        raise ValueError("event_source_id must identify exactly one catalogue row")
    half = int(config.reduction["psf_half_size_px"])
    radius = float(config.reduction["psf_aperture_radius_px"])
    maximum_offset = float(config.reduction.get("appulse_maximum_offset_px", 3.0))
    background_order = int(config.reduction.get("appulse_background_order", 0))
    fit_loss = str(config.reduction.get("appulse_fit_loss", "soft_l1"))
    if fit_loss not in {"linear", "soft_l1"}:
        raise ValueError("appulse_fit_loss must be 'linear' or 'soft_l1'")
    observer_location = mpc_217_location()
    frame_to_psf = {}
    for group in products.stack_groups:
        for frame_index in group.frame_indices:
            frame_to_psf[frame_index] = group.psf

    provisional_reference_jd = float(
        np.mean([model.time.jd for model in products.frame_models])
    )
    records = []
    for frame_index, model in enumerate(products.frame_models):
        if frame_index not in frame_to_psf:
            continue
        measured = model.stars.loc[model.stars.source_id == event_source_id]
        if len(measured) != 1:
            continue
        group_psf = frame_to_psf[frame_index]
        frame_psf, frame_psf_scope, psf_star_images = _frame_psf(
            model, config, group_psf, event_source_id
        )
        if psf_mode == "frame":
            psf, psf_scope = frame_psf, frame_psf_scope
        else:
            psf, psf_scope = group_psf, "group"
        star_x = float(measured.iloc[0].x)
        star_y = float(measured.iloc[0].y)
        star_ix, star_iy = np.round([star_x, star_y]).astype(int)
        star_stamp = model.image[
            star_iy - half : star_iy + half + 1,
            star_ix - half : star_ix + half + 1,
        ]
        if star_stamp.shape != psf.shape:
            continue
        star_dx, star_dy, _, _ = fit_empirical_psf_centroid(
            star_stamp,
            psf,
            stack_center=half,
            fit_half_size=half,
            sky_radius_px=radius,
            maximum_offset_px=maximum_offset,
        )
        star_xy = np.array([star_ix + star_dx, star_iy + star_dy])
        epoch_star = propagate_gaia_astrometry(
            catalog_row, model.time, observer_location=observer_location
        ).iloc[0]
        star_tangent = tangent_plane(
            [epoch_star.ra_epoch],
            [epoch_star.dec_epoch],
            origin_ra_deg=float(config.ephemeris["ra_deg"]),
            origin_dec_deg=float(config.ephemeris["dec_deg"]),
        )[0]
        asteroid_tangent = tangent_plane(
            [model.asteroid_ra_deg],
            [model.asteroid_dec_deg],
            origin_ra_deg=float(config.ephemeris["ra_deg"]),
            origin_dec_deg=float(config.ephemeris["dec_deg"]),
        )[0]
        jacobian = model.coefficients[1:].T
        base_xy = star_xy + np.linalg.solve(jacobian, asteroid_tangent - star_tangent)
        target_ix, target_iy = np.round(base_xy).astype(int)
        data = model.image[
            target_iy - half : target_iy + half + 1,
            target_ix - half : target_ix + half + 1,
        ].astype(float)
        if data.shape != psf.shape:
            continue
        yy, xx = np.indices(data.shape)
        _, _, noise = sigma_clipped_stats(
            data[np.hypot(xx - half, yy - half) > radius], sigma=3.0
        )
        if not np.isfinite(noise) or noise <= 0:
            continue
        psf_fwhm, psf_ellipticity = _psf_shape(psf)
        altitude, airmass, zenith_xi, zenith_eta = _atmosphere_geometry(
            model.time, model.asteroid_ra_deg, model.asteroid_dec_deg
        )
        records.append(
            {
                "frame": frame_index,
                "filename": model.path.name,
                "jd": model.time.jd,
                "minutes": (model.time.jd - provisional_reference_jd) * 1440.0,
                "data": data,
                "psf": psf,
                "noise": float(noise),
                "base_xy": base_xy,
                "integer_xy": np.array([target_ix, target_iy]),
                "inverse_jacobian": np.linalg.inv(jacobian),
                "psf_scope": psf_scope,
                "psf_star_images": psf_star_images,
                "psf_fwhm_px": psf_fwhm,
                "psf_ellipticity": psf_ellipticity,
                "altitude_deg": altitude,
                "airmass": airmass,
                "zenith_xi_unit": zenith_xi,
                "zenith_eta_unit": zenith_eta,
            }
        )
    if len(records) < 3:
        raise ValueError("Fewer than three usable differential frames")
    reference_jd = float(np.mean([record["jd"] for record in records]))
    for record in records:
        record["minutes"] = (record["jd"] - reference_jd) * 1440.0

    def evaluate(parameters, *, diagnostics=False, selected_records=None):
        residuals = []
        rows = []
        chosen_records = records if selected_records is None else selected_records
        for record in chosen_records:
            tangent_offset_mas = parameters[:2] + parameters[2:] * record["minutes"]
            pixel_offset = record["inverse_jacobian"] @ (
                tangent_offset_mas / MAS_PER_DEGREE
            )
            center_xy = record["base_xy"] + pixel_offset
            dx, dy = center_xy - record["integer_xy"]
            template = image_shift(
                record["psf"],
                (dy, dx),
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
            coefficients, covariance, residual = _linear_flux_and_background(
                record["data"], template, record["noise"], background_order
            )
            residuals.append(residual)
            if diagnostics:
                flux_error = np.sqrt(covariance[1, 1])
                try:
                    local_dx, local_dy, _, _ = fit_empirical_psf_centroid(
                        record["data"],
                        record["psf"],
                        stack_center=half,
                        fit_half_size=half,
                        sky_radius_px=radius,
                        maximum_offset_px=maximum_offset,
                    )
                    local_residual = np.linalg.inv(record["inverse_jacobian"]) @ (
                        np.array([local_dx - dx, local_dy - dy])
                    ) * MAS_PER_DEGREE
                    local_boundary = max(abs(local_dx), abs(local_dy)) >= maximum_offset - 0.05
                except (ValueError, RuntimeError):
                    local_dx = local_dy = np.nan
                    local_residual = np.full(2, np.nan)
                    local_boundary = True
                rows.append(
                    {
                        "frame": record["frame"],
                        "filename": record["filename"],
                        "jd": record["jd"],
                        "background": coefficients[0],
                        "flux": coefficients[1],
                        "flux_error": flux_error,
                        "fitted_flux_snr": coefficients[1] / flux_error,
                        "dx_px": dx,
                        "dy_px": dy,
                        "at_offset_boundary": bool(
                            max(abs(dx), abs(dy)) >= maximum_offset - 0.05
                        ),
                        "psf_scope": record["psf_scope"],
                        "psf_star_images": record["psf_star_images"],
                        "psf_fwhm_px": record["psf_fwhm_px"],
                        "psf_ellipticity": record["psf_ellipticity"],
                        "altitude_deg": record["altitude_deg"],
                        "airmass": record["airmass"],
                        "zenith_xi_unit": record["zenith_xi_unit"],
                        "zenith_eta_unit": record["zenith_eta_unit"],
                        "local_dx_px": local_dx,
                        "local_dy_px": local_dy,
                        "local_residual_xi_mas": local_residual[0],
                        "local_residual_eta_mas": local_residual[1],
                        "local_at_offset_boundary": bool(local_boundary),
                    }
                )
        if diagnostics:
            return pd.DataFrame(rows)
        return np.concatenate(residuals)

    if initial_parameters is None:
        initial_parameters = np.r_[
            products.psf_fit.intercept_mas,
            products.psf_fit.velocity_mas_per_minute,
        ]
    pixel_scale_mas = np.median(
        [np.sqrt(abs(np.linalg.det(np.linalg.inv(r["inverse_jacobian"])))) for r in records]
    ) * MAS_PER_DEGREE
    position_bound = maximum_offset * pixel_scale_mas
    result = least_squares(
        evaluate,
        np.asarray(initial_parameters, dtype=float),
        bounds=(
            [-position_bound, -position_bound, -200.0, -200.0],
            [position_bound, position_bound, 200.0, 200.0],
        ),
        x_scale=[100.0, 100.0, 10.0, 10.0],
        loss=fit_loss,
        f_scale=1.0,
        max_nfev=300,
    )
    if not result.success:
        raise RuntimeError(f"Appulse trajectory fit failed: {result.message}")
    residual = evaluate(result.x)
    dof = max(1, residual.size - 4 - 2 * len(records))
    covariance = np.linalg.inv(result.jac.T @ result.jac) * np.sum(residual**2) / dof
    diagnostics = evaluate(result.x, diagnostics=True).merge(
        products.frame_quality[
            [
                "frame",
                "outer_rms_mas",
                "inner_rms_after_mas",
                "correction_xi_mas",
                "correction_eta_mas",
            ]
        ],
        on="frame",
        how="left",
        validate="one_to_one",
    )
    jackknife_rows = []
    groups_for_jackknife = products.stack_groups if compute_jackknife else []
    for group_number, group in enumerate(groups_for_jackknife, start=1):
        omitted = set(group.frame_indices)
        retained = [record for record in records if record["frame"] not in omitted]
        if len(retained) < 3:
            continue
        check = least_squares(
            lambda parameters: evaluate(
                parameters, selected_records=retained
            ),
            result.x,
            bounds=(
                [-position_bound, -position_bound, -200.0, -200.0],
                [position_bound, position_bound, 200.0, 200.0],
            ),
            x_scale=[100.0, 100.0, 10.0, 10.0],
            loss=fit_loss,
            f_scale=1.0,
            max_nfev=150,
        )
        jackknife_rows.append(
            {
                "omitted_group": group_number,
                "omitted_frames": len(group.frame_indices),
                "position_xi_mas": check.x[0],
                "position_eta_mas": check.x[1],
                "velocity_xi_mas_per_minute": check.x[2],
                "velocity_eta_mas_per_minute": check.x[3],
            }
        )
    event = catalog_row.iloc[0]
    central_time = Time(reference_jd, format="jd", scale="utc")
    central_ra, central_dec = linear_ephemeris_coordinates(
        central_time,
        reference_time=Time(str(config.ephemeris["reference_utc"]), scale="utc"),
        reference_ra_deg=float(config.ephemeris["ra_deg"]),
        reference_dec_deg=float(config.ephemeris["dec_deg"]),
        ra_rate_arcsec_per_hour=float(config.ephemeris["ra_rate_arcsec_per_hour"]),
        dec_rate_arcsec_per_hour=float(config.ephemeris["dec_rate_arcsec_per_hour"]),
    )
    central_event = propagate_gaia_astrometry(
        catalog_row, central_time, observer_location=observer_location
    ).iloc[0]
    central_asteroid = tangent_plane(
        [central_ra],
        [central_dec],
        origin_ra_deg=float(config.ephemeris["ra_deg"]),
        origin_dec_deg=float(config.ephemeris["dec_deg"]),
    )[0]
    central_star = tangent_plane(
        [central_event.ra_epoch], [central_event.dec_epoch],
        origin_ra_deg=float(config.ephemeris["ra_deg"]),
        origin_dec_deg=float(config.ephemeris["dec_deg"]),
    )[0]
    separation = np.hypot(*(central_asteroid - central_star)) * 3600.0
    return AppulseFit(
        reference_jd=reference_jd,
        position_mas=result.x[:2],
        velocity_mas_per_minute=result.x[2:],
        formal_covariance_mas2=covariance[:2, :2],
        frame_diagnostics=diagnostics,
        pseudo_stars=_pseudo_star_table(products, catalog, event_source_id),
        jackknife=pd.DataFrame(jackknife_rows),
        systematics=summarize_appulse_systematics(diagnostics),
        event_source_id=event_source_id,
        event_separation_arcsec=float(separation),
        psf_mode=psf_mode,
    )
