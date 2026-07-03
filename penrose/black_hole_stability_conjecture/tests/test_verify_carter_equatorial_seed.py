import unittest

import verify_carter_equatorial_seed as verifier


class VerifyCarterEquatorialSeedTests(unittest.TestCase):
    def test_spin_half_equatorial_seed(self):
        result = verifier.verify()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["chart_family"], "equatorial_Q_equals_zero_seed")
        self.assertEqual(result["spin"], "a/M=1/2")
        self.assertEqual(result["regular_constraint"], "eta=Q/E^2=0")
        self.assertEqual(
            result["equatorial_radius_equation"],
            "r^3 - 6*M*r^2 + 9*M^2*r - 4*M*a^2 = 0",
        )
        self.assertEqual(
            result["checked_identities"],
            [
                "R_cleared_by_cubic",
                "dR_dr_cleared_by_cubic",
                "eta_cleared_by_cubic",
                "radial_hessian_reduces_mod_cubic",
            ],
        )

        prograde = result["branches"]["prograde"]
        self.assertEqual(prograde["root_isolation"]["interval"], ["2347/1000", "587/250"])
        self.assertEqual(prograde["root_isolation"]["monotonicity"], "decreasing")
        self.assertEqual(prograde["intervals"]["eta"], ["0", "0"])
        self.assertEqual(prograde["intervals"]["r_minus_M_over_M"], ["1347/1000", "337/250"])
        self.assertIn("radial_hessian_positive", prograde["checked_interval_conditions"])

        retrograde = result["branches"]["retrograde"]
        self.assertEqual(retrograde["root_isolation"]["interval"], ["883/250", "3533/1000"])
        self.assertEqual(retrograde["root_isolation"]["monotonicity"], "increasing")
        self.assertEqual(retrograde["intervals"]["eta"], ["0", "0"])
        self.assertEqual(retrograde["intervals"]["r_minus_M_over_M"], ["633/250", "2533/1000"])
        self.assertTrue(result["denominator_certificates"]["retrograde"]["Delta"]["avoids_zero"])


if __name__ == "__main__":
    unittest.main()
