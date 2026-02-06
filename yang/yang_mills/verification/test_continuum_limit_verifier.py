import unittest
import math

from verification.continuum_limit_verifier import RGFlowVerifier


class TestRGFlowVerifier(unittest.TestCase):
    def test_flow_stays_in_tube_for_perturbative_beta(self):
        # Choose a perturbative starting point. With beta=60, g^2=0.1.
        v = RGFlowVerifier(beta_start=60.0, max_steps=25, tube_radius=0.2, epsilon0=0.0)
        ok, fail_step, max_eps, final_g_sq = v.verify()
        self.assertTrue(ok)
        self.assertEqual(fail_step, -1)
        self.assertTrue(math.isfinite(max_eps))
        self.assertTrue(math.isfinite(final_g_sq))
        self.assertGreater(final_g_sq, 0.0)

    def test_rejects_non_positive_beta(self):
        with self.assertRaises(ValueError):
            RGFlowVerifier(beta_start=0.0)

    def test_rejects_non_positive_block_scale(self):
        with self.assertRaises(ValueError):
            RGFlowVerifier(beta_start=10.0, block_scale_L=0.0)


if __name__ == "__main__":
    unittest.main()
