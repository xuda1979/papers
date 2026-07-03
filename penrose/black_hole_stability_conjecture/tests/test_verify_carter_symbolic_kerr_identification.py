import copy
import json
import unittest
from pathlib import Path

import verify_carter_symbolic_kerr_identification as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterSymbolicKerrIdentificationTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_symbolic_identity_candidate(self):
        result = verifier.verify_file(
            ROOT
            / "certificates/carter_chart_certificates/spin_half_clean_chart_symbolic_kerr_identification_candidate.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_symbolic_kerr_identification")
        self.assertEqual(result["derivation_status"], "candidate_symbolic_identity")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertEqual(result["checked_identities"], ["symbolic_density_equals_six_coefficient_normal_form"])

    def test_rejects_bad_symbolic_normal_form(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_symbolic_kerr_identification_candidate.json"
        )
        data = copy.deepcopy(data)
        data["density_expression"]["sum"].pop()
        with self.assertRaisesRegex(ValueError, "normal-form identity failed"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unbacked_kerr_gravitational_proved_claim(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_symbolic_kerr_identification_candidate.json"
        )
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "exact Kerr derivation backend"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_non_quadratic_symbolic_density(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_symbolic_kerr_identification_candidate.json"
        )
        data = copy.deepcopy(data)
        data["density_expression"] = {"const": "0"}
        with self.assertRaisesRegex(ValueError, "normal-form identity failed|homogeneous quadratic"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
