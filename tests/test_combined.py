import unittest

import numpy as np
import pandas as pd

from fredschaaf_astrometry.combined import combine_gaia_and_ground


class CombinedAstrometryTests(unittest.TestCase):
    def setUp(self):
        self.gaia = pd.DataFrame(
            {
                "transit_id": [123],
                "epoch_utc_offset_jd": [2000.0],
                "position_angle_scan_deg": [90.0],
                "residual_al_mas": [4.0],
                "sigma_al_total_mas": [0.5],
            }
        )
        self.ground = pd.DataFrame(
            {
                "series_id": ["night_1"],
                "jd_utc": [2_460_000.0],
                "oc_xi_mas": [10.0],
                "oc_eta_mas": [-5.0],
                "cov_xi_xi_mas2": [100.0],
                "cov_xi_eta_mas2": [20.0],
                "cov_eta_eta_mas2": [64.0],
            }
        )

    def test_builds_al_and_correlated_ground_block(self):
        result = combine_gaia_and_ground(self.gaia, self.ground)
        np.testing.assert_allclose(result.values_mas, [4.0, 10.0, -5.0])
        np.testing.assert_allclose(result.projection_east_north[0], [1.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(
            result.covariance_mas2,
            [[0.25, 0.0, 0.0], [0.0, 100.0, 20.0], [0.0, 20.0, 64.0]],
        )

    def test_missing_ground_cross_covariance_requires_explicit_assumption(self):
        ground = self.ground.copy()
        ground["cov_xi_eta_mas2"] = np.nan
        with self.assertRaisesRegex(ValueError, "no ξ/η cross covariance"):
            combine_gaia_and_ground(self.gaia, ground)
        result = combine_gaia_and_ground(
            self.gaia,
            ground,
            assume_zero_missing_ground_cross_covariance=True,
        )
        self.assertEqual(result.covariance_mas2[1, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
