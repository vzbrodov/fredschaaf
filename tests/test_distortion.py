import unittest

import numpy as np
import pandas as pd

from fredschaaf_astrometry.distortion import build_distortion_grid


class DistortionGridTests(unittest.TestCase):
    def test_balances_sources_and_removes_affine_modes(self):
        rows = []
        for frame in range(6):
            for source, x, y in [(1, 10, 10), (2, 30, 10), (3, 10, 30), (4, 30, 30)]:
                rows.append((frame, source, x, y, 2 + 0.5 * x - 0.2 * y, -3 + 0.1 * x + 0.4 * y))
        samples = pd.DataFrame(rows, columns=["frame", "source_id", "x", "y", "oc_xi_mas", "oc_eta_mas"])
        grid = build_distortion_grid(
            samples, detector_width=40, detector_height=40, bins_x=2, bins_y=2,
            min_sources=1, min_frames=5, min_source_frames=5
        )
        np.testing.assert_allclose(grid["correction_xi_mas"], 0.0, atol=1e-10)
        np.testing.assert_allclose(grid["correction_eta_mas"], 0.0, atol=1e-10)
        self.assertTrue((grid["n_measurements"] == 6).all())


if __name__ == "__main__":
    unittest.main()
