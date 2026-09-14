import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fredschaaf_astrometry.series import (
    build_orbit_ground_series,
    fit_series_motion,
    load_ground_series_result,
    load_series_config,
)


class GroundSeriesTests(unittest.TestCase):
    def test_series_config_records_missing_calibration_frames(self):
        path = Path(__file__).resolve().parents[1] / "configs/20250903_R.toml"
        config = load_series_config(path)
        self.assertEqual(
            config.calibration,
            {"bias": False, "dark": False, "flat": False},
        )

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
        self.assertEqual(result.loc[0, "source"], "ground")
        self.assertEqual(result.loc[0, "cov_xi_xi_mas2"], 256.0)
        self.assertTrue(np.isnan(result.loc[0, "cov_xi_eta_mas2"]))
        self.assertEqual(result.loc[0, "cov_eta_eta_mas2"], 169.0)

    def test_joint_motion_fit_retains_cross_covariance(self):
        times = 2_460_000.0 + np.arange(5) / 1440.0
        centered = np.arange(5) - 2
        residual_xi = np.array([1.0, -2.0, 2.0, -2.0, 1.0])
        residual_eta = 2.0 * residual_xi
        stacks = pd.DataFrame(
            {
                "middle_jd": times,
                "psf_xi_mas": 10.0 + 3.0 * centered + residual_xi,
                "psf_eta_mas": -4.0 - centered + residual_eta,
            }
        )
        fit = fit_series_motion(stacks)
        np.testing.assert_allclose(fit.intercept_mas, [10.0, -4.0], atol=1e-8)
        np.testing.assert_allclose(
            fit.intercept_covariance_mas2,
            [[14 / 15, 28 / 15], [28 / 15, 56 / 15]],
            atol=1e-8,
        )
        self.assertEqual(fit.degrees_of_freedom, 3)

    def test_orbit_series_replaces_correlated_measurement(self):
        base = pd.DataFrame(
            {
                "series_id": ["night_1", "night_2"],
                "source": ["full", "full"],
                "central_utc": ["2025-01-01T00:00:00", "2025-01-02T00:00:00"],
                "cov_xi_xi_mas2": [4.0, 9.0],
                "cov_xi_eta_mas2": [0.0, 1.0],
                "cov_eta_eta_mas2": [5.0, 10.0],
                "quality_ok": [True, True],
            }
        )
        replacement = base.iloc[[1]].assign(
            series_id="night_2_appulse",
            source="appulse",
            cov_xi_xi_mas2=16.0,
            event_source_id=32637666936447744,
        )
        result = build_orbit_ground_series(base, {"night_2": replacement})
        self.assertEqual(len(result), 2)
        selected = result.loc[result.series_id == "night_2"].iloc[0]
        self.assertEqual(selected.source, "appulse")
        self.assertEqual(selected.cov_xi_xi_mas2, 16.0)
        self.assertEqual(selected.event_source_id, 32637666936447744)
        self.assertTrue(np.isfinite(result.jd_utc).all())


if __name__ == "__main__":
    unittest.main()
