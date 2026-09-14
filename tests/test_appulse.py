import unittest

import numpy as np
import pandas as pd

from fredschaaf_astrometry.appulse import (
    _linear_flux_and_background,
    _psf_shape,
    summarize_appulse_systematics,
)


class AppulseTests(unittest.TestCase):
    def test_linear_nuisance_fit_recovers_flux_and_background(self):
        template = np.zeros((9, 9), dtype=float)
        template[4, 4] = 0.5
        template[4, 5] = 0.3
        template[5, 4] = 0.2
        data = 17.0 + 1200.0 * template
        coefficients, covariance, residual = _linear_flux_and_background(
            data, template, noise=3.0
        )
        np.testing.assert_allclose(coefficients, [17.0, 1200.0], atol=1e-10)
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))

    def test_linear_gradient_does_not_bias_flux(self):
        yy, xx = np.indices((11, 11), dtype=float)
        template = np.exp(-0.5 * (((xx - 5) / 1.4) ** 2 + ((yy - 5) / 1.7) ** 2))
        template /= template.sum()
        data = 20.0 + 0.7 * (xx - 5) - 0.4 * (yy - 5) + 180.0 * template
        coefficients, _, residual = _linear_flux_and_background(
            data, template, 1.0, background_order=1
        )
        self.assertAlmostEqual(coefficients[1], 180.0, places=6)
        self.assertLess(np.max(np.abs(residual)), 1e-10)

    def test_psf_shape_and_systematics_summary(self):
        y, x = np.indices((15, 15))
        psf = np.exp(-0.5 * (((x - 7) / 2.0) ** 2 + ((y - 7) / 2.0) ** 2))
        fwhm, ellipticity = _psf_shape(psf)
        self.assertAlmostEqual(fwhm, 2.35482 * 2.0, delta=0.01)
        self.assertAlmostEqual(ellipticity, 0.0, places=3)
        diagnostics = pd.DataFrame(
            {
                "local_residual_xi_mas": [1.0, 2.0, 3.0],
                "local_residual_eta_mas": [0.0, 1.0, 0.0],
                "local_at_offset_boundary": [False, False, False],
                "zenith_xi_unit": [1.0, 1.0, 1.0],
                "zenith_eta_unit": [0.0, 0.0, 0.0],
                "airmass": [1.1, 1.2, 1.3],
                "jd": [2460000.0, 2460000.01, 2460000.02],
                "psf_scope": ["frame", "frame", "group"],
                "psf_star_images": [5, 4, 2],
                "psf_fwhm_px": [4.0, 4.2, 4.4],
                "outer_rms_mas": [100.0, 110.0, 120.0],
            }
        )
        summary = summarize_appulse_systematics(diagnostics).iloc[0]
        self.assertEqual(summary.frame_psf_frames, 2)
        self.assertEqual(summary.group_psf_fallback_frames, 1)
        self.assertAlmostEqual(summary.median_psf_fwhm_px, 4.2)
        self.assertFalse(summary.dcr_identifiable)


if __name__ == "__main__":
    unittest.main()
