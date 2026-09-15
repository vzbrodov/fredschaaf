"""Reusable measurement primitives for ground-based CCD astrometry."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import Distance, EarthLocation, GCRS, SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time, TimeDelta
from photutils.centroids import centroid_2dg
from scipy.ndimage import shift as image_shift
from scipy.optimize import curve_fit, least_squares


MAS_PER_DEGREE = 3_600_000.0


def read_fits_frame(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Read the primary image and detach its header from the FITS file."""
    with fits.open(path, memmap=False) as hdul:
        image = hdul[0].data.astype(float)
        header = hdul[0].header.copy()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D primary image in {path}")
    return image, header


def exposure_middle_time(header: Mapping[str, object]) -> Time:
    """Return the UTC midpoint using DATE-OBS as exposure start."""
    start = Time(str(header["DATE-OBS"]), format="isot", scale="utc")
    exposure_seconds = float(header["EXPTIME"])
    if not np.isfinite(exposure_seconds) or exposure_seconds <= 0:
        raise ValueError("EXPTIME must be finite and positive")
    return start + TimeDelta(exposure_seconds / 2, format="sec")


def linear_ephemeris_coordinates(
    time: Time,
    *,
    reference_time: Time,
    reference_ra_deg: float,
    reference_dec_deg: float,
    ra_rate_arcsec_per_hour: float,
    dec_rate_arcsec_per_hour: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Evaluate a local constant-rate ephemeris used for search and stacking."""
    hours = (time - reference_time).to_value("hour")
    ra = reference_ra_deg + ra_rate_arcsec_per_hour * hours / (
        3600.0 * np.cos(np.deg2rad(reference_dec_deg))
    )
    dec = reference_dec_deg + dec_rate_arcsec_per_hour * hours / 3600.0
    return ra, dec


def mpc_217_location() -> EarthLocation:
    """Return the MPC 217 (Assy-Turgen) location from its parallax constants."""
    longitude = np.deg2rad(77.87114)
    radius = 6378.137 * u.km
    return EarthLocation.from_geocentric(
        radius * 0.730114 * np.cos(longitude),
        radius * 0.730114 * np.sin(longitude),
        radius * 0.681643,
    )


def propagate_gaia_astrometry(
    catalog: pd.DataFrame,
    time: Time,
    *,
    observer_location: EarthLocation | None = None,
) -> pd.DataFrame:
    """Propagate Gaia astrometry, with a safe proper-motion-only fallback.

    Rows with a finite positive parallax and a physically plausible implied
    tangential speed use Astropy 3D space motion. A finite radial velocity adds
    perspective acceleration; otherwise zero radial velocity is used. When
    ``observer_location`` is supplied, the differential finite-distance GCRS
    displacement is added as annual/topocentric parallax. Rows without a usable
    parallax retain the explicit linear Gaia model. The generous 3000 km/s
    ceiling prevents noisy near-zero parallaxes from producing an unphysical
    distance/velocity pair inside ERFA; their parallax is negligible here.
    """
    required = {"ra", "dec", "pmra", "pmdec", "ref_epoch"}
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise ValueError(f"Missing Gaia columns: {missing}")
    result = catalog.copy()
    years = time.jyear - result["ref_epoch"].to_numpy(float)
    result["ra_epoch"] = result["ra"] + result["pmra"] * years / (
        MAS_PER_DEGREE * np.cos(np.deg2rad(result["dec"]))
    )
    result["dec_epoch"] = result["dec"] + result["pmdec"] * years / MAS_PER_DEGREE
    result["gaia_propagation"] = "linear_proper_motion"
    if "parallax" not in result:
        return result
    parallax = pd.to_numeric(result["parallax"], errors="coerce").to_numpy(float)
    total_proper_motion = np.hypot(
        pd.to_numeric(result["pmra"], errors="coerce").to_numpy(float),
        pd.to_numeric(result["pmdec"], errors="coerce").to_numpy(float),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        tangential_speed_kms = 4.74047 * total_proper_motion / parallax
    full = (
        np.isfinite(parallax)
        & (parallax > 0)
        & np.isfinite(tangential_speed_kms)
        & (tangential_speed_kms < 3000.0)
    )
    if not np.any(full):
        return result
    radial_velocity = np.zeros(len(result), dtype=float)
    has_radial_velocity = np.zeros(len(result), dtype=bool)
    if "radial_velocity" in result:
        values = pd.to_numeric(result["radial_velocity"], errors="coerce").to_numpy(float)
        has_radial_velocity = np.isfinite(values)
        radial_velocity[has_radial_velocity] = values[has_radial_velocity]
    selected = result.loc[full]
    coordinates = SkyCoord(
        ra=selected["ra"].to_numpy(float) * u.deg,
        dec=selected["dec"].to_numpy(float) * u.deg,
        distance=Distance(parallax=parallax[full] * u.mas),
        pm_ra_cosdec=selected["pmra"].to_numpy(float) * u.mas / u.yr,
        pm_dec=selected["pmdec"].to_numpy(float) * u.mas / u.yr,
        radial_velocity=radial_velocity[full] * u.km / u.s,
        obstime=Time(selected["ref_epoch"].to_numpy(float), format="jyear"),
        frame="icrs",
    )
    moved = coordinates.apply_space_motion(new_obstime=time)
    propagated = moved
    model = np.where(
        has_radial_velocity[full], "space_motion_with_rv", "space_motion_no_rv"
    ).astype(object)
    if observer_location is not None:
        obsgeoloc, obsgeovel = observer_location.get_gcrs_posvel(time)
        frame = GCRS(obstime=time, obsgeoloc=obsgeoloc, obsgeovel=obsgeovel)
        finite = moved.transform_to(frame)
        far = SkyCoord(
            ra=moved.ra,
            dec=moved.dec,
            distance=np.full(len(moved), 1e9) * u.pc,
            obstime=time,
            frame="icrs",
        ).transform_to(frame)
        delta_lon, delta_lat = far.spherical_offsets_to(finite)
        propagated = moved.spherical_offsets_by(delta_lon, delta_lat)
        model = np.char.add(model.astype(str), "_parallax")
    result.loc[full, "ra_epoch"] = propagated.ra.deg
    result.loc[full, "dec_epoch"] = propagated.dec.deg
    result.loc[full, "gaia_propagation"] = model
    return result


def propagate_gaia_linear_motion(catalog: pd.DataFrame, time: Time) -> pd.DataFrame:
    """Backward-compatible proper-motion-only Gaia propagation."""
    linear_catalog = catalog.drop(
        columns=["parallax", "radial_velocity"], errors="ignore"
    )
    return propagate_gaia_astrometry(linear_catalog, time)


def tangent_plane(
    ra_deg: np.ndarray | list[float] | float,
    dec_deg: np.ndarray | list[float] | float,
    *,
    origin_ra_deg: float,
    origin_dec_deg: float,
) -> np.ndarray:
    """Project sky coordinates onto a gnomonic plane, in degrees."""
    ra = np.deg2rad(np.atleast_1d(ra_deg).astype(float))
    dec = np.deg2rad(np.atleast_1d(dec_deg).astype(float))
    ra0 = np.deg2rad(origin_ra_deg)
    dec0 = np.deg2rad(origin_dec_deg)
    denominator = (
        np.sin(dec) * np.sin(dec0)
        + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
    )
    if np.any(denominator <= 0):
        raise ValueError("Coordinates are outside the valid gnomonic hemisphere")
    xi = np.cos(dec) * np.sin(ra - ra0) / denominator
    eta = (
        np.sin(dec) * np.cos(dec0)
        - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
    ) / denominator
    return np.column_stack((np.rad2deg(xi), np.rad2deg(eta)))


def inverse_tangent_plane(
    xi_deg: np.ndarray | float,
    eta_deg: np.ndarray | float,
    *,
    origin_ra_deg: float,
    origin_dec_deg: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Invert :func:`tangent_plane`."""
    xi = np.deg2rad(np.asarray(xi_deg, dtype=float))
    eta = np.deg2rad(np.asarray(eta_deg, dtype=float))
    ra0 = np.deg2rad(origin_ra_deg)
    dec0 = np.deg2rad(origin_dec_deg)
    denominator = np.cos(dec0) - eta * np.sin(dec0)
    ra = ra0 + np.arctan2(xi, denominator)
    dec = np.arctan2(
        np.sin(dec0) + eta * np.cos(dec0),
        np.hypot(denominator, xi),
    )
    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def centroid_star(
    image: np.ndarray, x: float, y: float, *, half_box: int
) -> tuple[float, float]:
    """Measure a star with a 2D Gaussian centroid in a bounded cutout."""
    if half_box < 3:
        raise ValueError("half_box must be at least 3 pixels")
    ix = int(round(x))
    iy = int(round(y))
    cutout = image[
        iy - half_box : iy + half_box + 1,
        ix - half_box : ix + half_box + 1,
    ]
    expected_shape = (2 * half_box + 1, 2 * half_box + 1)
    if cutout.shape != expected_shape:
        return np.nan, np.nan
    try:
        cx, cy = centroid_2dg(cutout - np.nanmedian(cutout))
    except Exception:
        return np.nan, np.nan
    safely_inside = 2 < cx < 2 * half_box - 2 and 2 < cy < 2 * half_box - 2
    if not (np.isfinite(cx + cy) and safely_inside):
        return np.nan, np.nan
    return ix - half_box + float(cx), iy - half_box + float(cy)


def fit_affine_model(
    pixel_xy: np.ndarray,
    tangent_xy_deg: np.ndarray,
    *,
    iterations: int = 4,
    clip_sigma: float = 3.0,
    minimum_stars: int = 6,
    minimum_clip_radius_mas: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit pixel-to-sky affine coefficients with radial robust clipping."""
    pixel = np.asarray(pixel_xy, dtype=float)
    tangent = np.asarray(tangent_xy_deg, dtype=float)
    if pixel.shape != tangent.shape or pixel.ndim != 2 or pixel.shape[1] != 2:
        raise ValueError("pixel_xy and tangent_xy_deg must both have shape (N, 2)")
    if iterations < 1 or clip_sigma <= 0:
        raise ValueError("iterations and clip_sigma must be positive")
    if minimum_stars < 3:
        raise ValueError("minimum_stars must be at least 3")
    if len(pixel) < minimum_stars:
        raise ValueError(f"At least {minimum_stars} stars are required")
    good = np.ones(len(pixel), dtype=bool)
    residual_mas = np.empty_like(tangent)
    for _ in range(iterations):
        design = np.column_stack((np.ones(good.sum()), pixel[good]))
        coefficients = np.linalg.lstsq(design, tangent[good], rcond=None)[0]
        all_design = np.column_stack((np.ones(len(pixel)), pixel))
        residual_mas = (tangent - all_design @ coefficients) * MAS_PER_DEGREE
        radial = np.hypot(residual_mas[:, 0], residual_mas[:, 1])
        median = np.median(radial[good])
        robust_sigma = 1.4826 * np.median(np.abs(radial[good] - median))
        limit = max(median + clip_sigma * robust_sigma, minimum_clip_radius_mas)
        new_good = radial < limit
        if new_good.sum() < minimum_stars or np.array_equal(new_good, good):
            break
        good = new_good
    return coefficients, residual_mas, good


def robust_upper_limit(values: np.ndarray, *, sigma: float = 3.0) -> float:
    """Return median + ``sigma`` times the MAD-based robust scatter."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or array.size == 0:
        raise ValueError("values must be a non-empty finite array")
    median = np.median(array)
    robust_scatter = 1.4826 * np.median(np.abs(array - median))
    return float(median + sigma * robust_scatter)


def normalized_star_stamp(
    image: np.ndarray,
    x: float,
    y: float,
    *,
    half_size: int,
    aperture_radius_px: float,
    minimum_flux: float,
    saturation_level: float,
) -> np.ndarray | None:
    """Return a subpixel-aligned, background-subtracted unit-flux star."""
    if half_size < 1 or not 0 < aperture_radius_px < np.sqrt(2) * half_size:
        raise ValueError("Invalid half_size or aperture_radius_px")
    size = 2 * half_size + 1
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - half_size, yy - half_size)
    ix = int(round(x))
    iy = int(round(y))
    stamp = image[
        iy - half_size : iy + half_size + 1,
        ix - half_size : ix + half_size + 1,
    ]
    if stamp.shape != (size, size):
        return None
    background = np.median(stamp[radius > aperture_radius_px])
    signal = stamp - background
    flux = np.sum(signal[radius < aperture_radius_px])
    if flux <= minimum_flux or np.max(stamp) >= saturation_level:
        return None
    center_in_stamp = (x - (ix - half_size), y - (iy - half_size))
    aligned = image_shift(
        signal,
        (half_size - center_in_stamp[1], half_size - center_in_stamp[0]),
        order=3,
        mode="constant",
        cval=0,
        prefilter=True,
    )
    normalization = np.sum(aligned[radius < aperture_radius_px])
    if normalization <= 0:
        return None
    return aligned / normalization


def combine_psf_stamps(stamps: list[np.ndarray], *, sigma: float = 3.0) -> np.ndarray:
    """Sigma-clip aligned star stamps into a normalized empirical PSF."""
    if not stamps:
        raise ValueError("No usable star stamps were supplied")
    cube = np.stack(stamps)
    psf = np.ma.mean(sigma_clip(cube, sigma=sigma, axis=0), axis=0).filled(0)
    total = psf.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("The empirical PSF has non-positive flux")
    return psf / total


def _gaussian_2d(coordinates, background, amplitude, x0, y0, sigma_x, sigma_y):
    x, y = coordinates
    exponent = ((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2
    return (background + amplitude * np.exp(-0.5 * exponent)).ravel()


def fit_gaussian_centroid(
    stack: np.ndarray, *, stack_center: int, fit_half_size: int, sky_radius_px: float
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit a free elliptical Gaussian around a stacked target."""
    half = fit_half_size
    center = stack_center
    data = stack[center - half : center + half + 1, center - half : center + half + 1]
    y, x = np.indices(data.shape)
    radius = np.hypot(x - half, y - half)
    peak_y, peak_x = np.unravel_index(np.nanargmax(data), data.shape)
    background = np.nanmedian(data[radius > sky_radius_px])
    lower_center = 2.0
    upper_x = float(data.shape[1] - 3)
    upper_y = float(data.shape[0] - 3)
    initial = [
        background,
        np.nanmax(data) - background,
        float(np.clip(peak_x, lower_center, upper_x)),
        float(np.clip(peak_y, lower_center, upper_y)),
        2.0,
        2.0,
    ]
    parameters, covariance = curve_fit(
        _gaussian_2d,
        (x, y),
        data.ravel(),
        p0=initial,
        bounds=(
            [-np.inf, 0, lower_center, lower_center, 0.7, 0.7],
            [np.inf, np.inf, upper_x, upper_y, 6, 6],
        ),
        maxfev=20_000,
    )
    return parameters[2] - half, parameters[3] - half, parameters, covariance


def fit_empirical_psf_centroid(
    stack: np.ndarray,
    psf: np.ndarray,
    *,
    stack_center: int,
    fit_half_size: int,
    sky_radius_px: float,
    maximum_offset_px: float = 5.0,
) -> tuple[float, float, np.ndarray, float]:
    """Fit background, flux and translation of a fixed empirical PSF."""
    half = fit_half_size
    center = stack_center
    data = stack[center - half : center + half + 1, center - half : center + half + 1]
    if data.shape != psf.shape:
        raise ValueError("PSF shape must match the target fit window")
    y, x = np.indices(data.shape)
    radius = np.hypot(x - half, y - half)
    peak_y, peak_x = np.unravel_index(np.nanargmax(data), data.shape)
    background = np.nanmedian(data[radius > sky_radius_px])

    def residual(parameters):
        fitted_background, flux, dx, dy = parameters
        shifted_psf = image_shift(
            psf, (dy, dx), order=3, mode="constant", cval=0, prefilter=True
        )
        return (fitted_background + flux * shifted_psf - data).ravel()

    initial = [
        background,
        max(1, np.sum(data - background)),
        np.clip(peak_x - half, -maximum_offset_px, maximum_offset_px),
        np.clip(peak_y - half, -maximum_offset_px, maximum_offset_px),
    ]
    fit = least_squares(
        residual,
        initial,
        bounds=(
            [-np.inf, 0, -maximum_offset_px, -maximum_offset_px],
            [np.inf, np.inf, maximum_offset_px, maximum_offset_px],
        ),
    )
    return fit.x[2], fit.x[3], fit.x, fit.cost


def pixel_offset_to_mas(
    affine_coefficients: list[np.ndarray], dx: float, dy: float
) -> np.ndarray:
    """Transform a target pixel offset through the mean frame Jacobian."""
    if not affine_coefficients:
        raise ValueError("At least one affine coefficient matrix is required")
    jacobians = [np.asarray(item, dtype=float)[1:].T for item in affine_coefficients]
    return np.mean(jacobians, axis=0) @ np.array([dx, dy]) * MAS_PER_DEGREE
