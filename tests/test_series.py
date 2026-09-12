import tempfile
import unittest
from pathlib import Path

import numpy as np

from fredschaaf_astrometry.series import load_ground_series_result


class GroundSeriesTests(unittest.TestCase):
    def test_legacy_result_is_normalized_without_invented_covariance(self):
        content = """quantity,value
central_utc,2025-09-03T21:21:55.931
RA_deg,43.487971321757456
Dec_deg,15.440525752826698
O-C_xi_mas,13.9
O-C_eta_mas,26.1
error_xi_mas,16.0
error_eta_mas,13.0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            path.write_text(content, encoding="utf-8")
            result = load_ground_series_result(
                path, series_id="20250903_R", filter_name="R"
            )
        self.assertEqual(result.loc[0, "source"], "pulkovo")
        self.assertEqual(result.loc[0, "cov_xi_xi_mas2"], 256.0)
        self.assertTrue(np.isnan(result.loc[0, "cov_xi_eta_mas2"]))
        self.assertEqual(result.loc[0, "cov_eta_eta_mas2"], 169.0)


if __name__ == "__main__":
    unittest.main()
