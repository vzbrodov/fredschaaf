import unittest

import numpy as np
from scipy.ndimage import shift as image_shift

from fredschaaf_astrometry.detection import fit_stack_flux_at_offset, fit_stack_psf


class FullStackDetectionTests(unittest.TestCase):
    def test_matched_psf_fit_recovers_faint_shifted_source(self):
        y, x = np.indices((15, 15))
        psf = np.exp(-0.5 * (((x - 7) / 1.7) ** 2 + ((y - 7) / 2.0) ** 2))
        psf /= psf.sum()
        image = np.full((41, 41), 30.0)
        embedded = np.zeros_like(image)
        embedded[13:28, 13:28] = psf
        expected = np.array([2.25, -1.35])
        image += 800 * image_shift(
            embedded, (expected[1], expected[0]), order=3, mode="constant", cval=0
        )
        rng = np.random.default_rng(4)
        image += rng.normal(0, 1.0, image.shape)
        parameters, covariance, noise, _, _, _ = fit_stack_psf(
            image, psf, maximum_offset_px=5.0
        )
        np.testing.assert_allclose(parameters[2:4], expected, atol=0.04)
        self.assertGreater(parameters[1] / np.sqrt(covariance[1, 1]), 30)
        self.assertAlmostEqual(noise, 1.0, delta=0.15)

    def test_external_seed_and_fixed_signed_flux(self):
        y, x = np.indices((11, 11))
        psf = np.exp(-0.5 * (((x - 5) / 1.5) ** 2 + ((y - 5) / 1.8) ** 2))
        psf /= psf.sum()
        image = np.full((31, 31), 12.0)
        embedded = np.zeros_like(image)
        embedded[10:21, 10:21] = psf
        expected = np.array([4.2, -3.4])
        image += 300 * image_shift(
            embedded, (expected[1], expected[0]), order=3, mode="constant", cval=0
        )
        rng = np.random.default_rng(9)
        image += rng.normal(0, 1.0, image.shape)
        parameters, _, _, _, _, _ = fit_stack_psf(
            image,
            psf,
            maximum_offset_px=0.8,
            search_center_px=(4.0, -3.5),
        )
        np.testing.assert_allclose(parameters[2:4], expected, atol=0.10)
        flux, flux_error, _ = fit_stack_flux_at_offset(image, psf, expected)
        self.assertAlmostEqual(flux, 300, delta=3 * flux_error)


if __name__ == "__main__":
    unittest.main()
