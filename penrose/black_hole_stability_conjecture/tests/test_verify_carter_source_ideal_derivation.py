import copy
import json
import unittest
from pathlib import Path

import verify_carter_source_ideal_derivation as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterSourceIdealDerivationTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_source_ideal_candidate(self):
        result = verifier.verify_file(
            ROOT
            / "certificates/carter_chart_certificates/spin_half_clean_chart_source_ideal_derivation_candidate.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_source_ideal_derivation")
        self.assertEqual(result["derivation_status"], "candidate_source_ideal")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertEqual(
            result["checked_identities"],
            ["source_density_equals_symbolic_normal_form_mod_constraint_ideal"],
        )
        self.assertEqual(
            result["checked_constraint_terms"],
            ["radial_constraint_R", "radial_constraint_dR"],
        )

    def test_rejects_bad_source_ideal_identity(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_source_ideal_derivation_candidate.json"
        )
        data = copy.deepcopy(data)
        data["constraint_ideal_terms"].pop()
        with self.assertRaisesRegex(ValueError, "source ideal identity failed"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_phase_dependent_constraint(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_source_ideal_derivation_candidate.json"
        )
        data = copy.deepcopy(data)
        data["constraint_ideal_terms"][0]["constraint_expression"] = {"prod": [{"var": "rho_R"}, {"var": "x"}]}
        with self.assertRaisesRegex(ValueError, "phase-free"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unbacked_kerr_gravitational_proved_claim(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_source_ideal_derivation_candidate.json"
        )
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "exact Kerr derivation backend"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
