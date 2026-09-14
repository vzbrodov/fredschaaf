import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS

from fredschaaf_astrometry.ground_wcs import (
    bootstrap_wcs_from_catalog_file,
    estimate_translation,
    transferred_wcs_header,
)


def shifted(image, dx, dy):
    result = np.zeros_like(image)
    source_y = slice(max(0, -dy), min(image.shape[0], image.shape[0] - dy))
    source_x = slice(max(0, -dx), min(image.shape[1], image.shape[1] - dx))
    target_y = slice(max(0, dy), min(image.shape[0], image.shape[0] + dy))
    target_x = slice(max(0, dx), min(image.shape[1], image.shape[1] + dx))
    result[target_y, target_x] = image[source_y, source_x]
    return result


class BootstrapWcsTests(unittest.TestCase):
    def test_recovers_integer_translation(self):
        rng = np.random.default_rng(4)
        reference = rng.normal(0, 0.05, (160, 220))
        for y, x, amplitude in [(35, 44, 20), (80, 170, 15), (125, 93, 18)]:
            reference[y, x] += amplitude
        target = shifted(reference, dx=-18, dy=12)
        dx, dy, contrast = estimate_translation(
            reference, target, downsample=1, quantile=98
        )
        self.assertAlmostEqual(dx, -18, delta=0.2)
        self.assertAlmostEqual(dy, 12, delta=0.2)
        self.assertGreater(contrast, 5)

    def test_translates_reference_pixel_and_preserves_target_metadata(self):
        reference = fits.Header(
            {
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRPIX1": 100.0,
                "CRPIX2": 80.0,
                "CRVAL1": 43.0,
                "CRVAL2": 15.0,
                "CD1_1": -1e-4,
                "CD1_2": 0.0,
                "CD2_1": 0.0,
                "CD2_2": 1e-4,
                "DATE-OBS": "2025-09-03T21:10:18",
                "DATE-END": "2025-09-03T21:10:23",
            }
        )
        target = fits.Header(
            {
                "DATE-OBS": "2025-09-02T21:24:08",
                "DATE-END": "2025-09-02T21:24:13",
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            result = transferred_wcs_header(target, reference, dx=-12.5, dy=3.0)
        self.assertEqual(result["DATE-OBS"], target["DATE-OBS"])
        self.assertEqual(result["DATE-END"], target["DATE-END"])
        self.assertNotIn("MJD-OBS", result)
        self.assertEqual(result["CRPIX1"], 87.5)
        self.assertEqual(result["CRPIX2"], 83.0)
        self.assertTrue(result["WCSBOOT"])

    def test_catalog_seed_preserves_target_time(self):
        reference = fits.Header(
            {
                "NAXIS": 2,
                "NAXIS1": 240,
                "NAXIS2": 180,
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRPIX1": 120.0,
                "CRPIX2": 90.0,
                "CRVAL1": 43.0,
                "CRVAL2": 15.0,
                "CD1_1": -1e-4,
                "CD1_2": 0.0,
                "CD2_1": 0.0,
                "CD2_2": 1e-4,
                "RA": "02:52:00",
                "DEC": "+15:00:00",
                "DATE-OBS": "2025-09-03T21:10:18",
            }
        )
        target_header = fits.Header(
            {
                "RA": "02:52:00",
                "DEC": "+15:00:00",
                "DATE-OBS": "2025-10-10T20:55:07",
                "DATE-END": "2025-10-10T20:55:12",
            }
        )
        points = np.array(
            [(55, 55), (85, 120), (115, 65), (145, 135),
             (175, 75), (195, 120), (70, 145), (160, 45)],
            dtype=float,
        )
        world = WCS(reference).all_pix2world(points, 0)
        image = np.zeros((180, 240), dtype=np.float32)
        dx, dy = 7, -5
        for x, y in points.astype(int):
            image[y + dy, x + dx] = 1000.0
        catalog = pd.DataFrame(
            {"ra": world[:, 0], "dec": world[:, 1], "phot_g_mean_mag": 15.0}
        )
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            reference_path = directory / "reference.fits"
            target_path = directory / "target.fits"
            catalog_path = directory / "catalog.csv"
            output_path = directory / "target_wcs.fits"
            fits.PrimaryHDU(np.zeros_like(image), reference).writeto(reference_path)
            fits.PrimaryHDU(image, target_header).writeto(target_path)
            catalog.to_csv(catalog_path, index=False)
            measured_dx, measured_dy, _ = bootstrap_wcs_from_catalog_file(
                target_path,
                reference_path,
                catalog_path,
                output_path,
                downsample=1,
                quantile=95,
                maximum_shift_px=20,
            )
            result = fits.getheader(output_path)
        self.assertEqual(measured_dx, dx)
        self.assertEqual(measured_dy, dy)
        self.assertEqual(result["DATE-OBS"], target_header["DATE-OBS"])
        self.assertEqual(result["DATE-END"], target_header["DATE-END"])
        self.assertNotIn("MJD-OBS", result)


if __name__ == "__main__":
    unittest.main()
