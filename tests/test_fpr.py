import unittest

import numpy as np
import pandas as pd

from fredschaaf_astrometry.fpr import (
    add_ccd_residuals,
    aggregate_transit_residuals,
    aggregate_transits,
    covariance_from_errors,
    prepare_ccd_observations,
    project_covariance,
    tangent_plane_residuals,
)


class FprCovarianceTests(unittest.TestCase):
    def test_scan_projection_cardinal_angles(self):
        covariance = covariance_from_errors([2, 2], [5, 5], [0, 0])
        variance_al, variance_ac = project_covariance(covariance, [0, 90])
        np.testing.assert_allclose(variance_al, [25, 4], atol=1e-12)
        np.testing.assert_allclose(variance_ac, [4, 25], atol=1e-12)

    def test_random_reduces_but_systematic_does_not(self):
        observations = pd.DataFrame(
            {
                "transit_id": [10, 10],
                "epoch_utc": [100.0, 100.0001],
                "position_angle_scan": [0.0, 0.0],
                "astrometric_outcome_ccd": [1, 1],
                "is_rejected": [False, False],
                "ra_error_random": [20.0, 20.0],
                "dec_error_random": [4.0, 4.0],
                "ra_dec_correlation_random": [0.0, 0.0],
                "ra_error_systematic": [3.0, 3.0],
                "dec_error_systematic": [2.0, 2.0],
                "ra_dec_correlation_systematic": [0.0, 0.0],
            }
        )
        prepared = prepare_ccd_observations(observations)
        transits = aggregate_transits(prepared)
        expected_al = np.hypot(4 / np.sqrt(2), 2)
        self.assertAlmostEqual(transits.loc[0, "sigma_al_total_mas"], expected_al)

    def test_rejected_ccd_is_not_aggregated(self):
        observations = pd.DataFrame(
            {
                "transit_id": [10, 10],
                "epoch_utc": [100.0, 100.0001],
                "position_angle_scan": [90.0, 90.0],
                "astrometric_outcome_ccd": [1, 1],
                "is_rejected": [False, True],
                "ra_error_random": [4.0, 1.0],
                "dec_error_random": [20.0, 20.0],
                "ra_dec_correlation_random": [0.0, 0.0],
                "ra_error_systematic": [2.0, 2.0],
                "dec_error_systematic": [3.0, 3.0],
                "ra_dec_correlation_systematic": [0.0, 0.0],
            }
        )
        transits = aggregate_transits(prepare_ccd_observations(observations))
        self.assertEqual(transits.loc[0, "n_ccd_good"], 1)
        self.assertAlmostEqual(
            transits.loc[0, "sigma_al_total_mas"], np.hypot(4, 2)
        )

    def test_ra_residual_wraps_at_zero(self):
        east, north = tangent_plane_residuals(
            [0.000001], [0.0], [359.999999], [0.0]
        )
        self.assertAlmostEqual(east[0], 7.2, places=5)
        self.assertAlmostEqual(north[0], 0.0)

    def test_residuals_are_reduced_before_transit_average(self):
        observations = pd.DataFrame(
            {
                "transit_id": [10, 10],
                "epoch_utc": [100.0, 100.0001],
                "position_angle_scan": [90.0, 90.0],
                "astrometric_outcome_ccd": [1, 1],
                "is_rejected": [False, False],
                "ra": [20.0, 20.001],
                "dec": [0.0, 0.0],
                "ra_error_random": [2.0, 2.0],
                "dec_error_random": [5.0, 5.0],
                "ra_dec_correlation_random": [0.0, 0.0],
                "ra_error_systematic": [1.0, 1.0],
                "dec_error_systematic": [3.0, 3.0],
                "ra_dec_correlation_systematic": [0.0, 0.0],
            }
        )
        prepared = prepare_ccd_observations(observations)
        model_ra = observations["ra"].to_numpy() - 10.0 / 3_600_000.0
        reduced = add_ccd_residuals(prepared, model_ra, [0.0, 0.0])
        transit = aggregate_transit_residuals(reduced)
        self.assertAlmostEqual(transit.loc[0, "residual_east_mas"], 10.0, places=5)
        self.assertAlmostEqual(transit.loc[0, "residual_al_mas"], 10.0, places=5)


if __name__ == "__main__":
    unittest.main()
