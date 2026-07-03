import unittest

import verify_carter_radial_identities as verifier


class VerifyCarterRadialIdentitiesTests(unittest.TestCase):
    def test_spherical_orbit_formulas_solve_radial_constraints(self):
        result = verifier.verify()
        self.assertTrue(result["accepted"])
        self.assertEqual(
            result["checked_identities"],
            [
                "B_formula_cleared",
                "P_formula_cleared",
                "R_cleared",
                "dR_dr_cleared",
                "radial_hessian_formula",
            ],
        )


if __name__ == "__main__":
    unittest.main()
