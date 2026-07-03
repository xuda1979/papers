import copy
import json
import unittest
from pathlib import Path

import verify_carter_defect_transport_morphism as verifier


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    "certificates/carter_chart_certificates/"
    "spin_half_clean_chart_defect_transport_shadow_candidate.json"
)


class VerifyCarterDefectTransportMorphismTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_defect_transport_shadow_candidate(self):
        result = verifier.verify_file(ROOT / CERTIFICATE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_defect_transport_morphism")
        self.assertEqual(result["derivation_status"], "candidate_transport_morphism")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["source_chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["target_chart"], "spin_half_r_7_2_to_18_5_sub04_transport_shadow")
        self.assertEqual(result["phase_norm_lower_bound"], "1")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertEqual(
            result["checked_nonnegative_transport_terms"],
            ["transport_shadow_positive_square_X_minus_L"],
        )
        self.assertEqual(
            result["checked_identities"][0],
            "target_density_equals_pulled_source_plus_nonnegative_transport_mod_ideal",
        )

    def test_rejects_transport_density_mismatch(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["target_density_expression"]["sum"].pop()
        with self.assertRaisesRegex(ValueError, "transport density identity failed"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_missing_phase_pullback_entry(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["phase_pullback_map"].pop("q")
        with self.assertRaisesRegex(ValueError, "phase_pullback_map"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_negative_transport_square_coefficient(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["nonnegative_transport_terms"][0]["coefficient"] = "-1/8192"
        with self.assertRaisesRegex(ValueError, "coefficient must be nonnegative"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_too_large_phase_norm_lower_bound(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["phase_norm_lower_bound"] = "2"
        with self.assertRaisesRegex(ValueError, "phase norm lower-bound identity failed"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unimplemented_gravitational_proof_upgrade(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "target chart"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
