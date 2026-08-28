#!/usr/bin/env python3
"""Basic per-frame Gaia DR3 astrometric calibration for the Fredschaaf data.

The fitted plate model is affine in a common gnomonic tangent plane:

    xi_deg  = xi_c0  + xi_cx*x + xi_cy*y
    eta_deg = eta_c0 + eta_cx*x + eta_cy*y

The output also contains the inverse coefficients (tangent plane to pixels),
fit/leave-one-out residuals, and enough provenance to reproduce the result.
Original FITS files are never modified.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyUserWarning
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import FITSFixedWarning, WCS
from astroquery.gaia import Gaia
from photutils.centroids import centroid_2dg
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)
warnings.filterwarnings("ignore", message="The fit may not have converged.*", category=AstropyUserWarning)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "Fredschaaf"
OUTPUT_ROOT = ROOT / "outputs" / "astrometry"

# One Gaia cone covers each observing field.  The centres deliberately exceed
# the camera footprint so small pointing changes do not require another query.
FIELDS = {
    "field_sep": (43.49, 15.45, 0.30),
    "field_oct": (42.08, 13.06, 0.30),
}


def field_for(path: Path) -> str:
    return "field_oct" if "20251010" in path.parts else "field_sep"


def query_gaia(name: str, output_dir: Path) -> pd.DataFrame:
    cache = output_dir / f"gaia_dr3_{name}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    ra, dec, radius = FIELDS[name]
    query = f"""
    SELECT source_id, ra, dec, pmra, pmdec, ref_epoch,
           phot_g_mean_mag, bp_rp, ruwe, duplicated_source
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius:.5f})) = 1
      AND phot_g_mean_mag BETWEEN 8.0 AND 17.5
      AND pmra IS NOT NULL AND pmdec IS NOT NULL
      AND astrometric_params_solved >= 31
      AND duplicated_source = 'false'
      AND ruwe < 1.4
    """
    table = Gaia.launch_job_async(query).get_results().to_pandas()
    table.columns = [str(c).lower() for c in table.columns]
    table.to_csv(cache, index=False)
    return table


def mid_epoch(header: fits.Header) -> Time:
    start = Time(header["DATE-OBS"], format="isot", scale="utc")
    if header.get("DATE-END"):
        end = Time(header["DATE-END"], format="isot", scale="utc")
        return start + (end - start) / 2
    return start + float(header["EXPTIME"]) / 2 / 86400


def propagated_catalog(catalog: pd.DataFrame, epoch: Time) -> pd.DataFrame:
    """Propagate Gaia positions with pmra = d(alpha)/dt*cos(delta)."""
    result = catalog.copy()
    dt = epoch.jyear - result.ref_epoch.to_numpy(float)
    dec = result.dec.to_numpy(float)
    result["ra_epoch"] = (
        result.ra.to_numpy(float)
        + result.pmra.to_numpy(float) * dt / (3_600_000 * np.cos(np.deg2rad(dec)))
    )
    result["dec_epoch"] = result.dec.to_numpy(float) + result.pmdec.to_numpy(float) * dt / 3_600_000
    return result


def initial_wcs(path: Path, header: fits.Header, templates: dict[str, fits.Header]) -> tuple[WCS, str]:
    own = WCS(header)
    if own.has_celestial:
        return own, "fits_header"
    if "20251010" in path.parts:
        return WCS(templates["oct"]), "exam_same_exposure"

    # The camera geometry and orientation are taken from the following night;
    # the telescope RA/DEC supplies the approximate centre for blind matching.
    template = templates["sep"].copy()
    pointing = SkyCoord(header["RA"], header["DEC"], unit=("hourangle", "deg"))
    template["CRVAL1"] = pointing.ra.deg
    template["CRVAL2"] = pointing.dec.deg
    return WCS(template), "next_night_geometry+telescope_pointing"


def detect_sources(image: np.ndarray) -> np.ndarray:
    _, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    for threshold in (8.0, 6.0, 5.0, 4.0, 3.0):
        found = DAOStarFinder(fwhm=4.0, threshold=threshold * std, exclude_border=True)(image - median)
        if found is not None and len(found) >= 20:
            break
    if found is None:
        return np.empty((0, 2))
    found.sort("flux", reverse=True)
    return np.column_stack((found["xcentroid"], found["ycentroid"]))[:400]


def translation_match(predicted: np.ndarray, detected: np.ndarray, max_shift: float = 140.0) -> tuple[float, float, int]:
    """Find a global WCS-to-image translation by consensus matching."""
    if len(predicted) == 0 or len(detected) == 0:
        return 0.0, 0.0, 0
    deltas = (detected[:, None, :] - predicted[None, :, :]).reshape(-1, 2)
    deltas = deltas[np.all(np.abs(deltas) <= max_shift, axis=1)]
    if not len(deltas):
        return 0.0, 0.0, 0
    bins = np.round(deltas / 5.0).astype(int)
    _, index, counts = np.unique(bins, axis=0, return_index=True, return_counts=True)
    candidates = deltas[index[np.argsort(counts)[-100:]]]
    tree = cKDTree(detected)
    best = (0, np.inf, np.zeros(len(predicted), dtype=bool), np.array([0.0, 0.0]))
    for delta in candidates:
        distance, _ = tree.query(predicted + delta, distance_upper_bound=8.0)
        good = np.isfinite(distance)
        score = (int(good.sum()), float(np.nanmedian(distance[good])) if good.any() else np.inf)
        if score[0] > best[0] or (score[0] == best[0] and score[1] < best[1]):
            best = (score[0], score[1], good, delta)
    if best[0] >= 3:
        _, nearest = tree.query(predicted[best[2]] + best[3])
        refined = np.median(detected[nearest] - predicted[best[2]], axis=0)
        return float(refined[0]), float(refined[1]), best[0]
    return float(best[3][0]), float(best[3][1]), best[0]


def _triangle_signatures(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale-free triangle signatures and vertices ordered by opposite side."""
    signatures, ordered_vertices = [], []
    for vertices in itertools.combinations(range(len(points)), 3):
        triangle = points[list(vertices)]
        opposite = np.array([
            np.linalg.norm(triangle[1] - triangle[2]),
            np.linalg.norm(triangle[0] - triangle[2]),
            np.linalg.norm(triangle[0] - triangle[1]),
        ])
        order = np.argsort(opposite)
        sides = opposite[order]
        if sides[2] < 80 or sides[0] / sides[2] < 0.12:
            continue
        signatures.append((sides[0] / sides[2], sides[1] / sides[2]))
        ordered_vertices.append(np.asarray(vertices)[order])
    return np.asarray(signatures), np.asarray(ordered_vertices)


def blind_affine_match(predicted: np.ndarray, detected: np.ndarray) -> tuple[np.ndarray, int]:
    """Match two star patterns using scale-free triangle invariants."""
    reference = predicted[:100]
    # Bright detections are far less contaminated by hot pixels and cosmic
    # rays in the short, noisy 2025-09-02 exposures.
    observed = detected[:8]
    ref_sig, ref_vertices = _triangle_signatures(reference)
    obs_sig, obs_vertices = _triangle_signatures(observed)
    if not len(ref_sig) or not len(obs_sig):
        return np.eye(3), 0
    invariant_tree = cKDTree(ref_sig)
    distances, nearest = invariant_tree.query(obs_sig, k=min(200, len(ref_sig)))
    obs_index = np.repeat(np.arange(len(obs_sig)), distances.shape[1])
    ref_index = nearest.reshape(-1)
    distances = distances.reshape(-1)
    candidate_order = np.argsort(distances)[:10000]
    observed_tree = cKDTree(detected)
    best_count, best_median, best_affine = 0, np.inf, np.eye(3)
    for candidate in candidate_order:
        if distances[candidate] > 0.02:
            break
        oi, ri = obs_index[candidate], ref_index[candidate]
        source = np.column_stack((np.ones(3), reference[ref_vertices[ri]]))
        target = observed[obs_vertices[oi]]
        affine, *_ = np.linalg.lstsq(source, target, rcond=None)
        singular_values = np.linalg.svd(affine[1:, :], compute_uv=False)
        # Both September nights use the same binned camera geometry.  This
        # rejects chance triangle coincidences with implausible scale/shear.
        if singular_values.min() < 0.985 or singular_values.max() > 1.015:
            continue
        transformed = np.column_stack((np.ones(len(predicted)), predicted)) @ affine
        match_distance, _ = observed_tree.query(transformed, distance_upper_bound=6.0)
        good = np.isfinite(match_distance)
        count = int(good.sum())
        median = float(np.median(match_distance[good])) if count else np.inf
        if count > best_count or (count == best_count and median < best_median):
            best_count, best_median, best_affine = count, median, affine
    matrix = np.eye(3)
    matrix[:2, :] = best_affine.T
    return matrix, best_count


def centroid_star(image: np.ndarray, x: float, y: float, half_box: int = 9) -> tuple[float, float]:
    xi, yi = int(round(x)), int(round(y))
    cutout = image[yi-half_box:yi+half_box+1, xi-half_box:xi+half_box+1]
    if cutout.shape != (2 * half_box + 1, 2 * half_box + 1):
        return np.nan, np.nan
    try:
        cx, cy = centroid_2dg(cutout - np.nanmedian(cutout))
    except Exception:
        return np.nan, np.nan
    measured = np.array([xi - half_box + cx, yi - half_box + cy])
    if not np.all(np.isfinite(measured)) or np.hypot(measured[0] - x, measured[1] - y) > 6:
        return np.nan, np.nan
    return float(measured[0]), float(measured[1])


def tangent_plane(ra_deg: np.ndarray, dec_deg: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    ra0r, dec0r = np.deg2rad(ra0), np.deg2rad(dec0)
    denominator = np.sin(dec) * np.sin(dec0r) + np.cos(dec) * np.cos(dec0r) * np.cos(ra - ra0r)
    xi = np.cos(dec) * np.sin(ra - ra0r) / denominator
    eta = (np.sin(dec) * np.cos(dec0r) - np.cos(dec) * np.sin(dec0r) * np.cos(ra - ra0r)) / denominator
    return np.column_stack((np.rad2deg(xi), np.rad2deg(eta)))


def robust_plate(xy: np.ndarray, tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack((np.ones(len(xy)), xy))
    keep = np.ones(len(xy), dtype=bool)
    for _ in range(8):
        coefficients, *_ = np.linalg.lstsq(design[keep], tangent[keep], rcond=None)
        residual = (tangent - design @ coefficients) * 3_600_000
        radius = np.hypot(residual[:, 0], residual[:, 1])
        centre = np.median(radius[keep])
        scatter = 1.4826 * np.median(np.abs(radius[keep] - centre))
        limit = max(150.0, centre + 4.0 * scatter)
        new_keep = radius < limit
        if new_keep.sum() < 5 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    coefficients, *_ = np.linalg.lstsq(design[keep], tangent[keep], rcond=None)
    residual = (tangent - design @ coefficients) * 3_600_000
    return coefficients, residual, keep


def loo_rms(xy: np.ndarray, tangent: np.ndarray) -> float:
    residuals = []
    for i in range(len(xy)):
        use = np.arange(len(xy)) != i
        design = np.column_stack((np.ones(use.sum()), xy[use]))
        coef, *_ = np.linalg.lstsq(design, tangent[use], rcond=None)
        prediction = np.array([1.0, xy[i, 0], xy[i, 1]]) @ coef
        residuals.append((tangent[i] - prediction) * 3_600_000)
    return float(np.sqrt(np.mean(np.sum(np.asarray(residuals) ** 2, axis=1))))


def calibrate(path: Path, catalogs: dict[str, pd.DataFrame], templates: dict[str, fits.Header]) -> dict[str, object]:
    with fits.open(path, memmap=False) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    height, width = image.shape
    epoch = mid_epoch(header)
    catalog = propagated_catalog(catalogs[field_for(path)], epoch)
    wcs, initial_source = initial_wcs(path, header, templates)
    predicted = wcs.all_world2pix(catalog[["ra_epoch", "dec_epoch"]].to_numpy(), 0)
    order = np.argsort(catalog["phot_g_mean_mag"].to_numpy())
    catalog = catalog.iloc[order].reset_index(drop=True)
    predicted = predicted[order]
    margin = 25
    detected = detect_sources(image)
    if initial_source == "next_night_geometry+telescope_pointing":
        affine, translation_matches = blind_affine_match(predicted, detected)
        predicted = np.column_stack((np.ones(len(predicted)), predicted)) @ affine[:2, :].T
        dx, dy = affine[0, 0], affine[1, 0]
    else:
        preliminary_inside = np.all(np.isfinite(predicted), axis=1) & (predicted[:, 0] > margin) & (predicted[:, 0] < width-margin) & (predicted[:, 1] > margin) & (predicted[:, 1] < height-margin)
        dx, dy, translation_matches = translation_match(predicted[preliminary_inside], detected)
        predicted += np.array([dx, dy])
    inside = np.all(np.isfinite(predicted), axis=1) & (predicted[:, 0] > margin) & (predicted[:, 0] < width-margin) & (predicted[:, 1] > margin) & (predicted[:, 1] < height-margin)
    candidate_indices = np.flatnonzero(inside)
    detection_tree = cKDTree(detected)
    match_distance, nearest_detection = detection_tree.query(predicted[inside], distance_upper_bound=6.0)
    matched = np.isfinite(match_distance)
    candidate_indices = candidate_indices[matched]
    nearest_detection = nearest_detection[matched]
    # One image detection may be assigned to only one Gaia source.
    _, unique_position = np.unique(nearest_detection, return_index=True)
    unique_position.sort()
    candidate_indices = candidate_indices[unique_position]
    nearest_detection = nearest_detection[unique_position]
    stars = catalog.iloc[candidate_indices].reset_index(drop=True)
    predicted = detected[nearest_detection]

    rows = []
    for star, (x0, y0) in zip(stars.itertuples(), predicted):
        x, y = centroid_star(image, x0, y0)
        if np.isfinite(x):
            rows.append((x, y, star.ra_epoch, star.dec_epoch, star.phot_g_mean_mag, star.source_id))
    measured = pd.DataFrame(rows, columns=["x", "y", "ra", "dec", "gmag", "source_id"])
    base = {
        "file": str(path.relative_to(ROOT)), "date_obs": header["DATE-OBS"],
        "filter": header.get("FILTER", ""), "initial_wcs": initial_source,
        "detected_sources": len(detected), "gaia_in_frame": len(stars),
        "centroided": len(measured), "translation_dx_pix": dx,
        "translation_dy_pix": dy, "translation_matches": translation_matches,
    }
    if len(measured) < 5:
        return base | {"status": "failed_too_few_stars", "used_stars": 0}

    # A common, fixed tangent point makes coefficients comparable within a field.
    ra0, dec0, _ = FIELDS[field_for(path)]
    xy = measured[["x", "y"]].to_numpy(float)
    tangent = tangent_plane(measured.ra.to_numpy(), measured.dec.to_numpy(), ra0, dec0)
    coef, residual, keep = robust_plate(xy, tangent)
    xy, tangent, residual = xy[keep], tangent[keep], residual[keep]
    if len(xy) < 5:
        return base | {"status": "failed_after_clipping", "used_stars": len(xy)}
    # Refit after clipping and form tangent -> pixel inverse.
    design = np.column_stack((np.ones(len(xy)), xy))
    coef, *_ = np.linalg.lstsq(design, tangent, rcond=None)
    residual = (tangent - design @ coef) * 3_600_000
    linear = coef[1:3, :].T
    inverse = np.linalg.inv(linear)
    inverse_offset = -inverse @ coef[0, :]
    scale_x = 3600 * np.hypot(coef[1, 0], coef[2, 0])
    scale_y = 3600 * np.hypot(coef[1, 1], coef[2, 1])
    dof = len(xy) - 3
    rms_xi = float(np.sqrt(np.sum(residual[:, 0] ** 2) / dof))
    rms_eta = float(np.sqrt(np.sum(residual[:, 1] ** 2) / dof))
    rms_2d = float(np.hypot(rms_xi, rms_eta))
    result = base | {
        "status": "ok", "used_stars": len(xy), "rejected_stars": int(len(measured)-len(xy)),
        "tangent_ra0_deg": ra0, "tangent_dec0_deg": dec0,
        "xi_c0_deg": coef[0, 0], "xi_cx_deg_pix": coef[1, 0], "xi_cy_deg_pix": coef[2, 0],
        "eta_c0_deg": coef[0, 1], "eta_cx_deg_pix": coef[1, 1], "eta_cy_deg_pix": coef[2, 1],
        "x_c0_pix": inverse_offset[0], "x_cxi_pix_deg": inverse[0, 0], "x_ceta_pix_deg": inverse[0, 1],
        "y_c0_pix": inverse_offset[1], "y_cxi_pix_deg": inverse[1, 0], "y_ceta_pix_deg": inverse[1, 1],
        "scale_x_arcsec_pix": scale_x, "scale_y_arcsec_pix": scale_y,
        "rms_xi_mas": rms_xi, "rms_eta_mas": rms_eta, "rms_2d_mas": rms_2d,
        "rms_2d_pix": rms_2d / (500 * (scale_x + scale_y)),
        "loo_2d_rms_mas": loo_rms(xy, tangent),
    }
    if result["loo_2d_rms_mas"] > 1000 or result["rms_2d_mas"] > 750:
        result["status"] = "failed_astrometric_qc"
    return result


def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    good = results[results.status == "ok"].copy()
    good["date"] = good.date_obs.str[:10]
    rows = []
    for (date, filt), group in good.groupby(["date", "filter"], sort=True):
        rows.append({
            "date": date, "filter": filt, "frames_ok": len(group),
            "frames_total": int(((results.date_obs.str[:10] == date) & (results["filter"] == filt)).sum()),
            "median_stars": float(group.used_stars.median()),
            "median_rms_2d_mas": float(group.rms_2d_mas.median()),
            "p16_rms_2d_mas": float(group.rms_2d_mas.quantile(0.16)),
            "p84_rms_2d_mas": float(group.rms_2d_mas.quantile(0.84)),
            "median_loo_2d_rms_mas": float(group.loo_2d_rms_mas.median()),
            "p16_loo_2d_rms_mas": float(group.loo_2d_rms_mas.quantile(0.16)),
            "p84_loo_2d_rms_mas": float(group.loo_2d_rms_mas.quantile(0.84)),
            "median_rms_2d_pix": float(group.rms_2d_pix.median()),
            "median_scale_arcsec_pix": float((0.5 * (group.scale_x_arcsec_pix + group.scale_y_arcsec_pix)).median()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Only process the first N frames (diagnostics).")
    parser.add_argument("--retry-failed", action="store_true", help="Reprocess only rows not marked ok.")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_ROOT.rglob("*.fits"))
    previous = None
    frame_csv = OUTPUT_ROOT / "frame_astrometry.csv"
    if args.retry_failed and frame_csv.exists():
        previous = pd.read_csv(frame_csv)
        failed_files = set(previous.loc[previous.status != "ok", "file"])
        files = [path for path in files if str(path.relative_to(ROOT)) in failed_files]
    if args.limit:
        files = files[:args.limit]
    templates = {
        "sep": fits.getheader(DATA_ROOT / "20250903" / "7065_R_4_0001_wcs.fits"),
        "oct": fits.getheader(ROOT / "exam" / "Fredschaaf_R.fits"),
    }
    catalogs = {name: query_gaia(name, OUTPUT_ROOT) for name in FIELDS}
    results = []
    for number, path in enumerate(files, 1):
        try:
            row = calibrate(path, catalogs, templates)
        except Exception as exc:
            row = {"file": str(path.relative_to(ROOT)), "date_obs": "", "filter": "", "status": f"error:{type(exc).__name__}:{exc}"}
        results.append(row)
        print(f"[{number:3d}/{len(files)}] {path.name}: {row['status']}", flush=True)
    frame_table = pd.DataFrame(results)
    if previous is not None:
        retried = set(frame_table.file)
        frame_table = pd.concat([previous.loc[~previous.file.isin(retried)], frame_table], ignore_index=True).sort_values("file")
    frame_table.to_csv(frame_csv, index=False)
    summary = make_summary(frame_table)
    summary.to_csv(OUTPUT_ROOT / "summary_by_series.csv", index=False)
    frame_table.loc[frame_table.status != "ok"].to_csv(OUTPUT_ROOT / "failed_frames.csv", index=False)
    metadata = {
        "catalog": "Gaia DR3", "position_epoch_model": "linear proper motion from Gaia ref_epoch",
        "plate_model": "six-constant affine: detector x,y <-> common gnomonic xi,eta",
        "residual_units": "mas", "pixel_origin": 0, "original_fits_modified": False,
        "frames_requested": len(frame_table), "frames_ok": int((frame_table.status == "ok").sum()),
    }
    (OUTPUT_ROOT / "README.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print("\n", summary.to_string(index=False), sep="")


if __name__ == "__main__":
    main()
