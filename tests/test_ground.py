import unittest

import numpy as np
import pandas as pd
from astropy.time import Time

from fredschaaf_astrometry.ground import (
    centroid_star,
    combine_psf_stamps,
    exposure_middle_time,
    fit_affine_model,
    fit_empirical_psf_centroid,
    fit_gaussian_centroid,
    inverse_tangent_plane,
    linear_ephemeris_coordinates,
    mpc_217_location,
    propagate_gaia_astrometry,
    propagate_gaia_linear_motion,
    tangent_plane,
)
from scipy.ndimage import shift as image_shift


class GroundMeasurementTests(unittest.TestCase):
    def test_exposure_midpoint_and_linear_ephemeris(self):
        middle = exposure_middle_time(
            {"DATE-OBS": "2025-09-03T21:20:00", "EXPTIME": 10}
        )
        self.assertEqual(middle.isot, "2025-09-03T21:20:05.000")
        ra, dec = linear_ephemeris_coordinates(
            middle,
            reference_time=Time("2025-09-03T21:20:00", scale="utc"),
            reference_ra_deg=43.0,
            reference_dec_deg=60.0,
            ra_rate_arcsec_per_hour=3.6,
            dec_rate_arcsec_per_hour=-7.2,
        )
        self.assertAlmostEqual(ra, 43.0 + 2.0 * 5 / 3600_000)
        self.assertAlmostEqual(dec, 60.0 - 7.2 * 5 / 3_600 / 3_600)

    def test_gaia_proper_motion_and_tangent_round_trip(self):
        catalog = pd.DataFrame(
            {
                "ra": [43.1],
                "dec": [15.2],
                "pmra": [100.0],
                "pmdec": [-50.0],
                "ref_epoch": [2016.0],
            }
        )
        moved = propagate_gaia_linear_motion(catalog, Time(2026.0, format="jyear"))
        self.assertAlmostEqual(
            moved.loc[0, "ra_epoch"],
            43.1 + 1000 / (3_600_000 * np.cos(np.deg2rad(15.2))),
        )
        projected = tangent_plane(
            moved.ra_epoch,
            moved.dec_epoch,
            origin_ra_deg=43.0,
            origin_dec_deg=15.0,
        )
        ra, dec = inverse_tangent_plane(
            projected[:, 0],
            projected[:, 1],
            origin_ra_deg=43.0,
            origin_dec_deg=15.0,
        )
        np.testing.assert_allclose(ra, moved.ra_epoch, atol=1e-12)
        np.testing.assert_allclose(dec, moved.dec_epoch, atol=1e-12)

    def test_gaia_space_motion_parallax_and_linear_fallback(self):
        catalog = pd.DataFrame(
            {
                "ra": [43.1, 43.2],
                "dec": [15.2, 15.3],
                "pmra": [100.0, 20.0],
                "pmdec": [-50.0, 10.0],
                "parallax": [100.0, np.nan],
                "radial_velocity": [30.0, np.nan],
                "ref_epoch": [2016.0, 2016.0],
            }
        )
        epoch = Time("2025-09-03T21:21:00", scale="utc")
        barycentric = propagate_gaia_astrometry(catalog, epoch)
        topocentric = propagate_gaia_astrometry(
            catalog, epoch, observer_location=mpc_217_location()
        )
        self.assertEqual(
            topocentric.loc[0, "gaia_propagation"],
            "space_motion_with_rv_parallax",
        )
        self.assertEqual(
            topocentric.loc[1, "gaia_propagation"], "linear_proper_motion"
        )
        parallax_shift_mas = np.hypot(
            (topocentric.loc[0, "ra_epoch"] - barycentric.loc[0, "ra_epoch"])
            * np.cos(np.deg2rad(barycentric.loc[0, "dec_epoch"])),
            topocentric.loc[0, "dec_epoch"] - barycentric.loc[0, "dec_epoch"],
        ) * 3_600_000
        self.assertGreater(parallax_shift_mas, 10.0)
        self.assertLess(parallax_shift_mas, 150.0)
        self.assertEqual(
            topocentric.loc[1, "ra_epoch"], barycentric.loc[1, "ra_epoch"]
        )
        self.assertEqual(
            topocentric.loc[1, "dec_epoch"], barycentric.loc[1, "dec_epoch"]
        )

    def test_centroid_and_affine_outlier_rejection(self):
        y, x = np.indices((41, 41))
        image = 100 + 10_000 * np.exp(
            -0.5 * (((x - 20.3) / 1.8) ** 2 + ((y - 19.7) / 2.1) ** 2)
        )
        cx, cy = centroid_star(image, 20, 20, half_box=8)
        self.assertAlmostEqual(cx, 20.3, places=3)
        self.assertAlmostEqual(cy, 19.7, places=3)

        pixel = np.array(
            [(x_value, y_value) for x_value in range(5) for y_value in range(5)],
            dtype=float,
        )
        coefficients = np.array([[1.0, -2.0], [0.01, 0.002], [-0.003, 0.02]])
        tangent = np.column_stack((np.ones(len(pixel)), pixel)) @ coefficients
        tangent[-1] += 1 / 3600
        fitted, _, used = fit_affine_model(
            pixel,
            tangent,
            minimum_clip_radius_mas=100,
        )
        self.assertFalse(used[-1])
        np.testing.assert_allclose(fitted, coefficients, atol=1e-12)

    def test_gaussian_and_fixed_psf_centroids(self):
        y, x = np.indices((15, 15))
        psf = np.exp(-0.5 * (((x - 7) / 1.7) ** 2 + ((y - 7) / 2.0) ** 2))
        psf /= psf.sum()
        expected_dx, expected_dy = 0.37, -0.28
        stack = 20 + 8_000 * image_shift(
            psf,
            (expected_dy, expected_dx),
            order=3,
            mode="constant",
            cval=0,
            prefilter=True,
        )
        empirical_dx, empirical_dy, _, _ = fit_empirical_psf_centroid(
            stack,
            psf,
            stack_center=7,
            fit_half_size=7,
            sky_radius_px=5.5,
        )
        gaussian_dx, gaussian_dy, _, _ = fit_gaussian_centroid(
            stack,
            stack_center=7,
            fit_half_size=7,
            sky_radius_px=5.5,
        )
        self.assertAlmostEqual(empirical_dx, expected_dx, places=5)
        self.assertAlmostEqual(empirical_dy, expected_dy, places=5)
        self.assertAlmostEqual(gaussian_dx, expected_dx, places=2)
        self.assertAlmostEqual(gaussian_dy, expected_dy, places=2)
        combined = combine_psf_stamps([psf, psf])
        self.assertAlmostEqual(combined.sum(), 1.0)

    def test_centroid_fits_accept_an_edge_peak_as_a_bounded_initial_guess(self):
        y, x = np.indices((15, 15))
        psf = np.exp(-0.5 * (((x - 7) / 1.5) ** 2 + ((y - 7) / 1.5) ** 2))
        psf /= psf.sum()
        stack = 10 + 5_000 * np.exp(
            -0.5 * (((x - 13) / 1.5) ** 2 + ((y - 7) / 1.5) ** 2)
        )
        gaussian_dx, gaussian_dy, _, _ = fit_gaussian_centroid(
            stack, stack_center=7, fit_half_size=7, sky_radius_px=5.5
        )
        empirical_dx, empirical_dy, _, _ = fit_empirical_psf_centroid(
            stack,
            psf,
            stack_center=7,
            fit_half_size=7,
            sky_radius_px=5.5,
            maximum_offset_px=5,
        )
        self.assertTrue(np.isfinite(gaussian_dx + gaussian_dy))
        self.assertTrue(np.isfinite(empirical_dx + empirical_dy))


if __name__ == "__main__":
    unittest.main()
