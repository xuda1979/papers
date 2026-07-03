import unittest

import verify_schwarzschild_photon_sphere as verifier


class VerifySchwarzschildPhotonSphereTests(unittest.TestCase):
    def test_schwarzschild_seed_chart_identities(self):
        result = verifier.verify()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["chart_family"], "zero_spin_schwarzschild_limit")
        self.assertEqual(result["photon_radius"], "3*M")
        self.assertEqual(result["normalized_total_angular_momentum"], "27*M^2")
        self.assertEqual(result["radial_hessian"], "54*M^2")
        self.assertEqual(
            result["checked_identities"],
            [
                "R_at_photon_sphere",
                "dR_dr_at_photon_sphere",
                "impact_parameter_formula",
                "photon_radius_equation",
                "radial_hessian_at_photon_sphere",
            ],
        )
        self.assertTrue(result["denominator_certificates"]["Delta_at_photon_sphere"]["avoids_zero"])


if __name__ == "__main__":
    unittest.main()
