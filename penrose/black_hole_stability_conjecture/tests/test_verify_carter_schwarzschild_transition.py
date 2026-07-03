import unittest

import verify_carter_schwarzschild_transition as verifier


class VerifyCarterSchwarzschildTransitionTests(unittest.TestCase):
    def test_blowup_transition_identities(self):
        result = verifier.verify()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["chart_family"], "zero_spin_schwarzschild_transition_blowup")
        self.assertEqual(result["matched_seed_chart"], "zero_spin_schwarzschild_limit")
        self.assertEqual(result["j_limit"], "27*M^2")
        self.assertEqual(result["radial_hessian_limit"], "54*M^2")
        self.assertEqual(
            result["blowup_coordinates"],
            ["sigma=(r-3*M)/a", "j=lambda^2+eta"],
        )
        self.assertEqual(result["remaining_j_denominator"], "(r-M)^2")
        self.assertEqual(result["limiting_old_variables"]["lambda"], "-9*M*sigma/2")
        self.assertEqual(result["limiting_old_variables"]["eta"], "27*M^2 - 81*M^2*sigma^2/4")
        self.assertIn("j_pole_cancellation", result["checked_identities"])
        self.assertIn("lambda_eta_reconstruct_j_limit", result["checked_identities"])
        self.assertTrue(result["denominator_certificates"]["Delta_limit"]["avoids_zero"])
        self.assertFalse(result["denominator_certificates"]["old_a_denominator"]["old_chart_valid_at_zero_spin"])


if __name__ == "__main__":
    unittest.main()
