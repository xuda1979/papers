import copy
import json
import unittest
from pathlib import Path

import verify_carter_trapped_constraint_derivation as verifier


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    "certificates/carter_chart_certificates/"
    "spin_half_clean_chart_trapped_constraint_derivation_certificate.json"
)


class VerifyCarterTrappedConstraintDerivationTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_trapped_constraint_derivation(self):
        result = verifier.verify_file(ROOT / CERTIFICATE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_trapped_constraint_derivation")
        self.assertEqual(result["derivation_status"], "exact_trapped_constraints")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertEqual(
            result["derived_constraints"],
            {
                "radial_constraint_R": "rho_R",
                "radial_constraint_dR": "rho_dR",
            },
        )
        self.assertIn("spherical_R_constraint_clears_to_zero", result["checked_identities"])
        self.assertIn("spherical_dR_constraint_clears_to_zero", result["checked_identities"])

    def test_rejects_wrong_residual_variable(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["constraint_derivation"]["normalized_residual_variables"]["radial_constraint_R"] = "wrong"
        with self.assertRaisesRegex(ValueError, "normalized residual variables"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_wrong_backend_path(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["backend_contract_certificate_path"] = "spin_half_clean_chart_principal_symbol_trace_candidate.json"
        with self.assertRaises((KeyError, ValueError)):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_wrong_derivation_status(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "candidate"
        with self.assertRaisesRegex(ValueError, "derivation_status"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
