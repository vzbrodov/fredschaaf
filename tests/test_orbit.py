import unittest

import numpy as np

from fredschaaf_astrometry.orbit import FPR_AU_METRES, IAU_AU_METRES, LB, fpr_state_to_tdb


class FprStateTests(unittest.TestCase):
    def test_published_scale_conversion(self):
        result = fpr_state_to_tdb(np.ones(6))
        rho = FPR_AU_METRES / IAU_AU_METRES
        np.testing.assert_allclose(result[:3], rho * (1 - LB))
        np.testing.assert_allclose(result[3:], rho)


if __name__ == "__main__":
    unittest.main()
