import unittest

import numpy as np

from fredschaaf_astrometry.periodogram import scan_vector_periods


class VectorPeriodogramTests(unittest.TestCase):
    def test_recovers_projected_period(self):
        time = np.linspace(0.0, 12.0, 120)
        angles = np.linspace(0.1, 5.7, len(time))
        projection = np.column_stack((np.sin(angles), np.cos(angles)))
        true_period_hours = 96.0
        phase = 2 * np.pi * time / (true_period_hours / 24.0)
        east = 8.0 * np.cos(phase) - 2.0 * np.sin(phase)
        north = 3.0 * np.cos(phase) + 5.0 * np.sin(phase)
        values = projection[:, 0] * east + projection[:, 1] * north
        covariance = np.eye(len(time))
        periods = np.linspace(80.0, 112.0, 65)
        result = scan_vector_periods(time, values, covariance, projection, periods)
        best = result.loc[result["chi2"].idxmin()]
        self.assertAlmostEqual(best["period_hours"], true_period_hours)
        self.assertLess(best["chi2"], 1e-20)


if __name__ == "__main__":
    unittest.main()
