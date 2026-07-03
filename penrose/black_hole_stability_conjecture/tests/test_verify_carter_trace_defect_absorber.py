import copy
import json
import unittest
from pathlib import Path

import verify_carter_trace_defect_absorber as verifier


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    "certificates/carter_chart_certificates/"
    "spin_half_clean_chart_trace_defect_absorber_candidate.json"
)


class VerifyCarterTraceDefectAbsorberTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_trace_defect_absorber_candidate(self):
        result = verifier.verify_file(ROOT / CERTIFICATE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_trace_defect_absorber")
        self.assertEqual(result["derivation_status"], "candidate_defect_absorber")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertEqual(result["checked_constraint_terms"], ["gauge_constraint_sigma"])
        self.assertEqual(result["checked_nonnegative_defects"], ["positive_trace_defect_x_plus_q"])
        self.assertEqual(
            result["checked_identities"][0],
            "candidate_density_equals_trace_plus_nonnegative_defects_mod_constraint_ideal",
        )

    def test_rejects_missing_nonnegative_defect_identity(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["nonnegative_defect_terms"] = []
        with self.assertRaisesRegex(ValueError, "trace defect absorber identity failed"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_negative_nonnegative_defect_coefficient(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["nonnegative_defect_terms"][0]["coefficient"] = "-1/4096"
        with self.assertRaisesRegex(ValueError, "coefficient must be nonnegative"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_non_linear_square_defect(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["nonnegative_defect_terms"][0]["square_expression"] = {"square": {"var": "x"}}
        with self.assertRaisesRegex(ValueError, "phase degree 1"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_phase_dependent_constraint(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["constraint_ideal_terms"][0]["constraint_expression"] = {
            "prod": [
                {"var": "sigma_gauge"},
                {"var": "x"}
            ]
        }
        with self.assertRaisesRegex(ValueError, "phase-free"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unimplemented_gravitational_proof_upgrade(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "Kerr principal-symbol derivation"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
