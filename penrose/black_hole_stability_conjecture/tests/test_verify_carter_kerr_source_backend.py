import copy
import json
import unittest
from pathlib import Path

import verify_carter_kerr_source_backend as verifier


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    "certificates/carter_chart_certificates/"
    "spin_half_clean_chart_kerr_source_backend_contract_candidate.json"
)


class VerifyCarterKerrSourceBackendTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_backend_contract_candidate(self):
        result = verifier.verify_file(ROOT / CERTIFICATE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_kerr_source_backend")
        self.assertEqual(result["derivation_status"], "candidate_backend_contract")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertEqual(
            result["checked_identities"],
            [
                "backend_exported_source_density_matches_source_ideal",
                "backend_exported_constraints_match_source_ideal",
                "source_density_equals_symbolic_normal_form_mod_constraint_ideal",
            ],
        )
        self.assertEqual(
            result["remaining_exact_derivation_obligations"],
            [
                "derive the gauge-reduced linearized vacuum Einstein principal symbol in this chart",
                "compute the normalized gravitational principal density q_grav in the listed phase variables",
                "derive rho_R and rho_dR as the trapped characteristic constraints R=0 and dR=0",
                "prove the exported source density equals q_grav modulo those trapped constraints",
            ],
        )

    def test_rejects_backend_source_density_mismatch(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["backend_contract"]["interface"]["exported_source_density_expression"]["sum"].pop()
        with self.assertRaisesRegex(ValueError, "source density interface mismatch"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_backend_constraint_mismatch(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["backend_contract"]["interface"]["exported_constraint_expressions"][0]["expression"] = {"const": "0"}
        with self.assertRaisesRegex(ValueError, "constraint interface mismatch"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_wrong_operator_chart(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["backend_contract"]["operator_family"]["chart"] = "wrong_chart"
        with self.assertRaisesRegex(ValueError, "operator_family.chart"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unimplemented_gravitational_proof_upgrade(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "principal-symbol derivation engine"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
