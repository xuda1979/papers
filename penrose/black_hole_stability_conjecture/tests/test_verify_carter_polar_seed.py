import unittest

import verify_carter_polar_seed as verifier


class VerifyCarterPolarSeedTests(unittest.TestCase):
    def test_spin_half_polar_seed(self):
        result = verifier.verify()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["chart_family"], "polar_L_equals_zero_seed")
        self.assertEqual(result["spin"], "a/M=1/2")
        self.assertEqual(result["regular_constraint"], "lambda=L/E=0")
        self.assertEqual(result["polar_radius_equation"], "r^3 - 3*M*r^2 + a^2*r + M*a^2 = 0")
        self.assertEqual(result["root_isolation"]["interval"], ["576643/200000", "144161/50000"])
        self.assertTrue(result["root_isolation"]["derivative_positive_on_interval"])
        self.assertEqual(result["interval_bounds"]["r_over_M"], ["576643/200000", "144161/50000"])
        self.assertEqual(result["interval_bounds"]["r_minus_M_over_M"], ["376643/200000", "94161/50000"])
        self.assertEqual(
            result["interval_bounds"]["radial_hessian_over_M_squared"],
            ["30329398596306569/627740000000000", "2843423104006187/58850468750000"],
        )
        self.assertIn("lambda_zero_cubic", result["checked_identities"])
        self.assertIn("R_cleared_by_cubic", result["checked_identities"])
        self.assertIn("Delta_positive", result["checked_interval_conditions"])
        self.assertTrue(result["denominator_certificates"]["r_minus_M"]["avoids_zero"])


if __name__ == "__main__":
    unittest.main()
