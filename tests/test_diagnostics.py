import unittest

import numpy as np
import pandas as pd

from scripts.diagnose_ground_dcr import fit_dcr_fixed_effects


class GroundDiagnosticsTests(unittest.TestCase):
    def test_dcr_fixed_effects_recovers_colour_term(self):
        rng = np.random.default_rng(12)
        rows = []
        coefficient = 80.0
        colours = np.linspace(0.4, 2.4, 10)
        star_offsets = rng.normal(0.0, 50.0, (len(colours), 2))
        for frame in range(8):
            airmass = 1.05 + 0.06 * frame
            tan_z = np.sqrt(airmass**2 - 1.0)
            angle = 0.15 * frame
            zenith = np.array([np.cos(angle), np.sin(angle)])
            frame_offset = rng.normal(0.0, 20.0, 2)
            for source_id, colour in enumerate(colours):
                displacement = coefficient * (colour - np.median(colours)) * tan_z
                observed = (
                    star_offsets[source_id]
                    + frame_offset
                    + displacement * zenith
                    + rng.normal(0.0, 0.5, 2)
                )
                rows.append(
                    {
                        "source_id": source_id,
                        "frame": frame,
                        "bp_rp": colour,
                        "airmass": airmass,
                        "zenith_xi_unit": zenith[0],
                        "zenith_eta_unit": zenith[1],
                        "oc_xi_mas": observed[0],
                        "oc_eta_mas": observed[1],
                    }
                )
        result = fit_dcr_fixed_effects(pd.DataFrame(rows))
        self.assertAlmostEqual(
            result["dcr_mas_per_bp_rp_per_tan_z"], coefficient, delta=1.0
        )
        self.assertTrue(result["identifiable"])


if __name__ == "__main__":
    unittest.main()
