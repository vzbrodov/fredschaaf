"""Forced and blind matched-PSF detection on a motion-compensated CCD series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.time import Time
from scipy.ndimage import shift as image_shift
from scipy.optimize import least_squares

from .ground import (
    MAS_PER_DEGREE,
    inverse_tangent_plane,
    linear_ephemeris_coordinates,
    tangent_plane,
)
from .ground_pipeline import ReductionProducts
from .series import SeriesConfig


@dataclass(frozen=True)
class FullStackDetection:
    """One series-level measurement from all accepted frames."""

    reference_jd: float
    frame_indices: tuple[int, ...]
    span_minutes: float
    image: np.ndarray
    psf: np.ndarray
    offset_px: np.ndarray
    offset_mas: np.ndarray
    covariance_mas2: np.ndarray
    flux: float
    flux_error: float
    fitted_snr: float
    peak_snr: float
    noise: float
    at_search_boundary: bool
    ra_deg: float
    dec_deg: float


def fit_stack_flux_at_offset(
    image: np.ndarray,
    psf: np.ndarray,
    offset_px: np.ndarray | tuple[float, float],
) -> tuple[float, float, float]:
    """Fit background and signed flux at a fixed, externally supplied position.

    This is the appropriate diagnostic for a non-detection: unlike a blind
    positive-flux search, it does not select the largest favourable noise peak.
    Returns flux, flux uncertainty and background noise per stack pixel.
    """
    data = np.asarray(image, dtype=float)
    kernel = np.asarray(psf, dtype=float)
    embedded = np.zeros_like(data)
    y0 = (data.shape[0] - kernel.shape[0]) // 2
    x0 = (data.shape[1] - kernel.shape[1]) // 2
    embedded[y0 : y0 + kernel.shape[0], x0 : x0 + kernel.shape[1]] = kernel
    dx, dy = np.asarray(offset_px, float)
    template = image_shift(
        embedded, (dy, dx), order=3, mode="constant", cval=0.0, prefilter=True
    )
    yy, xx = np.indices(data.shape)
    center_x = (data.shape[1] - 1) / 2 + dx
    center_y = (data.shape[0] - 1) / 2 + dy
    fit_radius = max(kernel.shape) / 2 + 2.0
    valid = (
        np.isfinite(data)
        & np.isfinite(template)
        & (np.hypot(xx - center_x, yy - center_y) <= fit_radius)
    )
    design = np.column_stack((np.ones(valid.sum()), template[valid]))
    coefficients = np.linalg.lstsq(design, data[valid], rcond=None)[0]
    residual = data[valid] - design @ coefficients
    dof = max(1, valid.sum() - 2)
    noise = float(np.sqrt(residual @ residual / dof))
    covariance = np.linalg.pinv(design.T @ design) * noise**2
    return float(coefficients[1]), float(np.sqrt(covariance[1, 1])), noise


def fit_stack_psf(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    maximum_offset_px: float,
    grid_step_px: float = 0.5,
    search_center_px: np.ndarray | tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Fit background, flux and PSF translation after a coarse matched search.

    Returns parameters ``[background, flux, dx, dy]``, their covariance, the
    background noise, peak-pixel S/N, reduced chi-square and fit cost.
    """
    data = np.asarray(image, dtype=float)
    kernel = np.asarray(psf, dtype=float)
    if data.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and psf must be two-dimensional")
    if any(k > d for k, d in zip(kernel.shape, data.shape)):
        raise ValueError("psf must fit inside image")
    embedded = np.zeros_like(data)
    y0 = (data.shape[0] - kernel.shape[0]) // 2
    x0 = (data.shape[1] - kernel.shape[1]) // 2
    embedded[y0 : y0 + kernel.shape[0], x0 : x0 + kernel.shape[1]] = kernel
    yy, xx = np.indices(data.shape)
    center_x = (data.shape[1] - 1) / 2
    center_y = (data.shape[0] - 1) / 2
    center = np.zeros(2) if search_center_px is None else np.asarray(search_center_px, float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("search_center_px must contain finite dx, dy")
    fit_radius = max(kernel.shape) / 2 + maximum_offset_px + 2.0
    distance = np.hypot(xx - (center_x + center[0]), yy - (center_y + center[1]))
    sky = distance > (
        min(
            fit_radius + 1,
            0.45 * min(data.shape),
        )
    )
    _, background, noise = sigma_clipped_stats(data[sky], sigma=3.0)
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Could not estimate positive stack noise")
    valid = np.isfinite(data) & (distance <= fit_radius)

    def linear_solution(template):
        design = np.column_stack((np.ones(valid.sum()), template[valid]))
        coefficients = np.linalg.lstsq(design, data[valid], rcond=None)[0]
        residual = (design @ coefficients - data[valid]) / noise
        return coefficients, float(residual @ residual)

    best = None
    grid = np.arange(-maximum_offset_px, maximum_offset_px + grid_step_px / 2, grid_step_px)
    for delta_y in grid:
        for delta_x in grid:
            dx, dy = center + (delta_x, delta_y)
            template = image_shift(
                embedded, (dy, dx), order=3, mode="constant", cval=0.0, prefilter=True
            )
            coefficients, chi2 = linear_solution(template)
            if coefficients[1] <= 0:
                continue
            candidate = (chi2, coefficients[0], coefficients[1], dx, dy)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("No positive-flux matched-PSF solution")

    def residual(parameters):
        fitted_background, flux, dx, dy = parameters
        template = image_shift(
            embedded, (dy, dx), order=3, mode="constant", cval=0.0, prefilter=True
        )
        return (
            fitted_background + flux * template[valid] - data[valid]
        ) / noise

    result = least_squares(
        residual,
        np.array(best[1:]),
        bounds=(
            [-np.inf, 0.0, center[0] - maximum_offset_px, center[1] - maximum_offset_px],
            [np.inf, np.inf, center[0] + maximum_offset_px, center[1] + maximum_offset_px],
        ),
        x_scale=[noise, max(best[2], noise), 1.0, 1.0],
        max_nfev=300,
    )
    dof = max(1, valid.sum() - 4)
    reduced_chi2 = float(2 * result.cost / dof)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * reduced_chi2
    peak_snr = float((np.nanmax(data) - background) / noise)
    return result.x, covariance, float(noise), peak_snr, reduced_chi2, float(result.cost)


def build_full_stack_detection(
    products: ReductionProducts,
    config: SeriesConfig,
    *,
    maximum_offset_px: float = 8.0,
    stack_half_size_px: int | None = None,
    frame_indices: tuple[int, ...] | list[int] | None = None,
    search_center_offset_mas: np.ndarray | tuple[float, float] | None = None,
    injection_offset_mas: np.ndarray | tuple[float, float] | None = None,
    injection_flux: float | None = None,
) -> FullStackDetection:
    """Shift-and-add accepted frames and fit one series-level position.

    ``injection_offset_mas`` and ``injection_flux`` add the empirical stellar
    PSF to every real frame before interpolation, clipping and stacking.  The
    offset is fixed in the tangent plane, so the artificial source follows the
    same ephemeris motion as the target.  This hook is intended for recovery
    tests and never mutates the stored frame images.
    """
    all_accepted = tuple(
        int(index)
        for index, value in enumerate(products.frame_quality.accepted)
        if bool(value)
    )
    if frame_indices is None:
        accepted = all_accepted
    else:
        requested = set(map(int, frame_indices))
        accepted = tuple(index for index in all_accepted if index in requested)
    if not accepted:
        raise ValueError("No accepted frames")
    if (injection_offset_mas is None) != (injection_flux is None):
        raise ValueError("injection offset and flux must be provided together")
    if injection_flux is not None and (not np.isfinite(injection_flux) or injection_flux <= 0):
        raise ValueError("injection_flux must be finite and positive")
    injection_mas = (
        None if injection_offset_mas is None else np.asarray(injection_offset_mas, float)
    )
    if injection_mas is not None and (
        injection_mas.shape != (2,) or not np.all(np.isfinite(injection_mas))
    ):
        raise ValueError("injection_offset_mas must contain finite xi, eta")
    half = (
        int(config.reduction["stack_half_size_px"])
        if stack_half_size_px is None
        else int(stack_half_size_px)
    )
    if half <= maximum_offset_px + max(
        int(config.reduction["psf_half_size_px"]), 2
    ):
        raise ValueError("stack half-size must contain the shifted PSF search region")
    frame_to_psf = {}
    for group in products.stack_groups:
        if group.psf is None:
            continue
        for frame_index in group.frame_indices:
            frame_to_psf[int(frame_index)] = group.psf
    cutouts = []
    for index in accepted:
        model = products.frame_models[index]
        expected = tangent_plane(
            [model.asteroid_ra_deg],
            [model.asteroid_dec_deg],
            origin_ra_deg=float(config.ephemeris["ra_deg"]),
            origin_dec_deg=float(config.ephemeris["dec_deg"]),
        )[0] + model.correction_deg
        asteroid_xy = np.linalg.solve(
            model.coefficients[1:].T, expected - model.coefficients[0]
        )
        ix, iy = np.round(asteroid_xy).astype(int)
        cutout = model.image[
            iy - half : iy + half + 1,
            ix - half : ix + half + 1,
        ].astype(float)
        if cutout.shape != (2 * half + 1, 2 * half + 1):
            continue
        _, background, _ = sigma_clipped_stats(
            cutout, sigma=float(config.reduction["background_clip_sigma"])
        )
        prepared = cutout - background
        if injection_mas is not None:
            if index not in frame_to_psf:
                raise ValueError(f"No empirical PSF for injected frame {index}")
            frame_psf = np.asarray(frame_to_psf[index], float)
            embedded = np.zeros_like(prepared)
            y0 = (embedded.shape[0] - frame_psf.shape[0]) // 2
            x0 = (embedded.shape[1] - frame_psf.shape[1]) // 2
            embedded[y0 : y0 + frame_psf.shape[0], x0 : x0 + frame_psf.shape[1]] = frame_psf
            injection_px = np.linalg.solve(
                model.coefficients[1:].T, injection_mas / MAS_PER_DEGREE
            )
            fractional = asteroid_xy - np.array([ix, iy], float)
            prepared = prepared + float(injection_flux) * image_shift(
                embedded,
                (fractional[1] + injection_px[1], fractional[0] + injection_px[0]),
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
        cutouts.append(
            image_shift(
                prepared,
                (-(asteroid_xy[1] - iy), -(asteroid_xy[0] - ix)),
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
        )
    if len(cutouts) != len(accepted):
        raise ValueError("One or more accepted target cutouts cross the detector edge")
    cube = np.stack(cutouts)
    stack = np.ma.mean(
        sigma_clip(
            cube,
            sigma=float(config.reduction["stack_clip_sigma"]),
            axis=0,
        ),
        axis=0,
    ).filled(np.nan)
    group_and_weight = [
        (group, len(set(group.frame_indices).intersection(accepted)))
        for group in products.stack_groups
    ]
    group_and_weight = [
        (group, weight)
        for group, weight in group_and_weight
        if weight and group.psf is not None
    ]
    if not group_and_weight:
        raise ValueError("No empirical PSF is available for the accepted frames")
    weights = np.array([weight for _, weight in group_and_weight], float)
    psfs = np.stack([group.psf for group, _ in group_and_weight])
    psf = np.average(psfs, axis=0, weights=weights)
    psf /= psf.sum()
    jacobian = np.mean(
        [products.frame_models[index].coefficients[1:].T for index in accepted],
        axis=0,
    )
    search_center_px = None
    if search_center_offset_mas is not None:
        search_center_px = np.linalg.solve(
            jacobian,
            np.asarray(search_center_offset_mas, float) / MAS_PER_DEGREE,
        )
    parameters, covariance, noise, peak_snr, _, _ = fit_stack_psf(
        stack,
        psf,
        maximum_offset_px=maximum_offset_px,
        search_center_px=search_center_px,
    )
    offset_px = parameters[2:4]
    offset_mas = jacobian @ offset_px * MAS_PER_DEGREE
    covariance_mas2 = (
        jacobian @ covariance[2:4, 2:4] @ jacobian.T * MAS_PER_DEGREE**2
    )
    reference_jd = float(
        np.mean([products.frame_models[index].time.jd for index in accepted])
    )
    central_time = Time(reference_jd, format="jd", scale="utc")
    central_ra, central_dec = linear_ephemeris_coordinates(
        central_time,
        reference_time=Time(str(config.ephemeris["reference_utc"]), scale="utc"),
        reference_ra_deg=float(config.ephemeris["ra_deg"]),
        reference_dec_deg=float(config.ephemeris["dec_deg"]),
        ra_rate_arcsec_per_hour=float(config.ephemeris["ra_rate_arcsec_per_hour"]),
        dec_rate_arcsec_per_hour=float(config.ephemeris["dec_rate_arcsec_per_hour"]),
    )
    base = tangent_plane(
        [central_ra],
        [central_dec],
        origin_ra_deg=float(config.ephemeris["ra_deg"]),
        origin_dec_deg=float(config.ephemeris["dec_deg"]),
    )[0]
    measured = base + offset_mas / MAS_PER_DEGREE
    ra, dec = inverse_tangent_plane(
        *measured,
        origin_ra_deg=float(config.ephemeris["ra_deg"]),
        origin_dec_deg=float(config.ephemeris["dec_deg"]),
    )
    flux_error = float(np.sqrt(covariance[1, 1]))
    return FullStackDetection(
        reference_jd=reference_jd,
        frame_indices=accepted,
        span_minutes=float(
            (
                products.frame_models[accepted[-1]].time
                - products.frame_models[accepted[0]].time
            ).to_value("min")
        ),
        image=stack,
        psf=psf,
        offset_px=offset_px,
        offset_mas=offset_mas,
        covariance_mas2=covariance_mas2,
        flux=float(parameters[1]),
        flux_error=flux_error,
        fitted_snr=float(parameters[1] / flux_error),
        peak_snr=peak_snr,
        noise=noise,
        at_search_boundary=bool(
            np.max(np.abs(offset_px - (
                np.zeros(2) if search_center_px is None else search_center_px
            ))) >= maximum_offset_px - 0.05
        ),
        ra_deg=float(ra),
        dec_deg=float(dec),
    )
