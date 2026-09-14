"""Bootstrap WCS solutions from an overlapping solved reference image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


def _geometry_header(wcs: WCS) -> fits.Header:
    """Serialize WCS geometry without copying observation-time metadata."""
    header = wcs.to_header(relax=True)
    for temporal_key in ("DATE-OBS", "MJD-OBS", "DATE-END", "MJD-END"):
        header.remove(temporal_key, ignore_missing=True)
    return header


def stellar_high_pass(
    image: np.ndarray,
    *,
    downsample: int = 2,
    quantile: float = 99.5,
) -> np.ndarray:
    """Return a sparse star-dominated image for translation matching."""
    if downsample < 1:
        raise ValueError("downsample must be positive")
    if not 50.0 < quantile < 100.0:
        raise ValueError("quantile must be between 50 and 100")
    sampled = np.asarray(image, dtype=np.float32)[::downsample, ::downsample]
    high_pass = gaussian_filter(sampled, 1.0) - gaussian_filter(sampled, 4.0)
    threshold = np.nanpercentile(high_pass, quantile)
    result = np.where(high_pass > threshold, high_pass - threshold, 0.0)
    border = max(8, 40 // downsample)
    result[:border] = 0
    result[-border:] = 0
    result[:, :border] = 0
    result[:, -border:] = 0
    return result.astype(np.float32, copy=False)


def _parabolic_peak(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(np.clip(offset, -1.0, 1.0))


def estimate_translation(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    downsample: int = 2,
    quantile: float = 99.5,
    maximum_shift_px: float | None = None,
) -> tuple[float, float, float]:
    """Estimate ``target_pixel - reference_pixel`` by FFT correlation.

    Returns ``(dx, dy, peak_contrast)`` in full-resolution pixels. The
    contrast is the correlation peak divided by the median positive value and
    is intended only as an operational diagnostic.
    """
    if np.shape(reference) != np.shape(target):
        raise ValueError("reference and target images must have the same shape")
    reference_stars = stellar_high_pass(
        reference, downsample=downsample, quantile=quantile
    )
    target_stars = stellar_high_pass(
        target, downsample=downsample, quantile=quantile
    )
    correlation = fftconvolve(
        target_stars, reference_stars[::-1, ::-1], mode="same"
    )
    center_y, center_x = np.asarray(correlation.shape) // 2
    if maximum_shift_px is not None:
        radius = int(np.ceil(maximum_shift_px / downsample))
        mask = np.ones(correlation.shape, dtype=bool)
        mask[
            max(0, center_y - radius) : center_y + radius + 1,
            max(0, center_x - radius) : center_x + radius + 1,
        ] = False
        correlation[mask] = -np.inf
    peak_y, peak_x = np.unravel_index(np.nanargmax(correlation), correlation.shape)
    if not (0 < peak_x < correlation.shape[1] - 1):
        subpixel_x = 0.0
    else:
        subpixel_x = _parabolic_peak(
            correlation[peak_y, peak_x - 1],
            correlation[peak_y, peak_x],
            correlation[peak_y, peak_x + 1],
        )
    if not (0 < peak_y < correlation.shape[0] - 1):
        subpixel_y = 0.0
    else:
        subpixel_y = _parabolic_peak(
            correlation[peak_y - 1, peak_x],
            correlation[peak_y, peak_x],
            correlation[peak_y + 1, peak_x],
        )
    positive = correlation[np.isfinite(correlation) & (correlation > 0)]
    contrast = (
        float(correlation[peak_y, peak_x] / np.median(positive))
        if positive.size
        else np.inf
    )
    dx = (peak_x + subpixel_x - center_x) * downsample
    dy = (peak_y + subpixel_y - center_y) * downsample
    return float(dx), float(dy), contrast


def transferred_wcs_header(
    target_header: fits.Header,
    reference_header: fits.Header,
    *,
    dx: float,
    dy: float,
) -> fits.Header:
    """Copy reference camera WCS and translate its reference pixel."""
    result = target_header.copy()
    wcs_header = _geometry_header(WCS(reference_header))
    wcs_header["CRPIX1"] = float(wcs_header["CRPIX1"]) + dx
    wcs_header["CRPIX2"] = float(wcs_header["CRPIX2"]) + dy
    result.update(wcs_header)
    result["WCSBOOT"] = (True, "WCS bootstrapped from overlapping solved image")
    result["WCSDX"] = (dx, "target-reference image translation, pixel")
    result["WCSDY"] = (dy, "target-reference image translation, pixel")
    return result


def bootstrap_wcs_file(
    input_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    downsample: int = 2,
    quantile: float = 99.5,
    maximum_shift_px: float | None = None,
    overwrite: bool = False,
) -> tuple[float, float, float]:
    """Write one translated WCS FITS and return its correlation diagnostics."""
    input_path = Path(input_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    with fits.open(reference_path, memmap=False) as hdul:
        reference = hdul[0].data.astype(np.float32)
        reference_header = hdul[0].header.copy()
    with fits.open(input_path, memmap=False) as hdul:
        target = hdul[0].data
        target_header = hdul[0].header.copy()
    dx, dy, contrast = estimate_translation(
        reference,
        target,
        downsample=downsample,
        quantile=quantile,
        maximum_shift_px=maximum_shift_px,
    )
    header = transferred_wcs_header(
        target_header, reference_header, dx=dx, dy=dy
    )
    header["WCSREF"] = str(reference_path)
    header["WCSCORR"] = (contrast, "translation correlation peak contrast")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=target, header=header).writeto(
        output_path, overwrite=overwrite
    )
    return dx, dy, contrast


def bootstrap_wcs_from_catalog_file(
    input_path: str | Path,
    camera_reference_path: str | Path,
    catalog_path: str | Path,
    output_path: str | Path,
    *,
    maximum_magnitude: float = 16.5,
    downsample: int = 2,
    quantile: float = 99.5,
    maximum_shift_px: float | None = None,
    overwrite: bool = False,
) -> tuple[float, float, float]:
    """Seed a new-field WCS from telescope pointing and a Gaia catalogue.

    The solved reference supplies only the camera scale, rotation, and the
    stable offset between the FITS ``RA``/``DEC`` pointing and image centre.
    A catalogue-to-image correlation then determines the residual translation.
    This is intended for a new field observed with unchanged camera geometry.
    """
    input_path = Path(input_path)
    camera_reference_path = Path(camera_reference_path)
    catalog_path = Path(catalog_path)
    output_path = Path(output_path)
    with fits.open(camera_reference_path, memmap=False) as hdul:
        reference_header = hdul[0].header.copy()
    with fits.open(input_path, memmap=False) as hdul:
        target = hdul[0].data
        target_header = hdul[0].header.copy()

    reference_pointing = SkyCoord(
        reference_header["RA"], reference_header["DEC"], unit=("hourangle", "deg")
    )
    target_pointing = SkyCoord(
        target_header["RA"], target_header["DEC"], unit=("hourangle", "deg")
    )
    reference_wcs = WCS(reference_header).celestial
    reference_center = SkyCoord(*reference_wcs.wcs.crval, unit="deg")
    offset_lon, offset_lat = reference_pointing.spherical_offsets_to(reference_center)
    target_center = target_pointing.spherical_offsets_by(offset_lon, offset_lat)
    seeded_wcs = reference_wcs.deepcopy()
    seeded_wcs.wcs.crval = [target_center.ra.deg, target_center.dec.deg]

    catalog = pd.read_csv(catalog_path)
    required = {"ra", "dec", "phot_g_mean_mag"}
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise ValueError(f"Catalogue is missing columns: {missing}")
    xy = seeded_wcs.all_world2pix(catalog[["ra", "dec"]].to_numpy(float), 0)
    margin = 20
    selected = (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= margin)
        & (xy[:, 0] < target.shape[1] - margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 1] < target.shape[0] - margin)
        & (catalog["phot_g_mean_mag"].to_numpy(float) <= maximum_magnitude)
    )
    if selected.sum() < 6:
        raise ValueError("Fewer than six bright catalogue stars fall on the detector")
    synthetic = np.zeros(target.shape, dtype=np.float32)
    magnitudes = catalog.loc[selected, "phot_g_mean_mag"].to_numpy(float)
    weights = np.minimum(10 ** (-0.4 * (magnitudes - maximum_magnitude)), 1000.0)
    for (x, y), weight in zip(xy[selected], weights):
        synthetic[int(round(y)), int(round(x))] += weight
    synthetic = gaussian_filter(synthetic, 2.0)
    observed_stars = stellar_high_pass(
        target, downsample=downsample, quantile=quantile
    )
    synthetic = synthetic[::downsample, ::downsample]
    correlation = fftconvolve(
        observed_stars, synthetic[::-1, ::-1], mode="same"
    )
    center_y, center_x = np.asarray(correlation.shape) // 2
    if maximum_shift_px is not None:
        radius = int(np.ceil(maximum_shift_px / downsample))
        outside = np.ones(correlation.shape, dtype=bool)
        outside[
            max(0, center_y - radius) : center_y + radius + 1,
            max(0, center_x - radius) : center_x + radius + 1,
        ] = False
        correlation[outside] = -np.inf
    peak_y, peak_x = np.unravel_index(np.nanargmax(correlation), correlation.shape)
    dx = float((peak_x - center_x) * downsample)
    dy = float((peak_y - center_y) * downsample)
    positive = correlation[np.isfinite(correlation) & (correlation > 0)]
    contrast = (
        float(correlation[peak_y, peak_x] / np.median(positive))
        if positive.size
        else np.inf
    )

    result = target_header.copy()
    wcs_header = _geometry_header(seeded_wcs)
    wcs_header["CRPIX1"] = float(wcs_header["CRPIX1"]) + dx
    wcs_header["CRPIX2"] = float(wcs_header["CRPIX2"]) + dy
    result.update(wcs_header)
    result["WCSBOOT"] = (True, "WCS seeded from pointing and Gaia catalogue")
    result["WCSREF"] = str(camera_reference_path)
    result["WCSCAT"] = str(catalog_path)
    result["WCSDX"] = (dx, "image-catalog translation, pixel")
    result["WCSDY"] = (dy, "image-catalog translation, pixel")
    result["WCSCORR"] = (contrast, "catalog correlation peak contrast")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=target, header=result).writeto(output_path, overwrite=overwrite)
    return dx, dy, contrast
