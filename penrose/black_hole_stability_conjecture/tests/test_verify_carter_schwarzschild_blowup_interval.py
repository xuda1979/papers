import unittest

import verify_carter_schwarzschild_blowup_interval as verifier


class VerifyCarterSchwarzschildBlowupIntervalTests(unittest.TestCase):
    def test_rational_blowup_box(self):
        result = verifier.verify()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["chart_family"], "zero_spin_schwarzschild_blowup_interval")
        self.assertEqual(result["parent_transition"], "zero_spin_schwarzschild_transition_blowup")
        self.assertEqual(result["box"]["abs_a_over_M_max"], "1/64")
        self.assertEqual(result["box"]["abs_sigma_max"], "1")
        self.assertEqual(result["intervals"]["r_over_M"], ["191/64", "193/64"])
        self.assertEqual(result["intervals"]["r_minus_M_over_M"], ["127/64", "129/64"])
        self.assertEqual(result["intervals"]["Delta_over_M_squared"], ["11777/4096", "6401/2048"])
        self.assertEqual(result["intervals"]["j_over_M_squared"], ["861695/33282", "1859610627/66064384"])
        self.assertEqual(
            result["intervals"]["radial_hessian_over_M_squared"],
            ["441225089/8520192", "464904769/8258048"],
        )
        self.assertIn("Delta_positive", result["checked_conditions"])
        self.assertIn("j_contains_schwarzschild_value", result["checked_conditions"])
        self.assertIn("radial_hessian_contains_schwarzschild_value", result["checked_conditions"])


if __name__ == "__main__":
    unittest.main()
