import copy
import json
import unittest
from pathlib import Path

import verify_carter_rational_coefficient_extraction as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterRationalCoefficientExtractionTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_candidate_extraction(self):
        result = verifier.verify_file(
            ROOT
            / "certificates/carter_chart_certificates/spin_half_clean_chart_rational_coefficient_extraction_candidate.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_rational_coefficient_extraction")
        self.assertEqual(result["derivation_status"], "candidate_extraction")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertEqual(result["extracted_coefficient_intervals"]["x*q"], "[-1/8192, 1/8192]")

    def test_rejects_extraction_outside_enclosure(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_rational_coefficient_extraction_candidate.json"
        )
        data = copy.deepcopy(data)
        data["interval_symbols"]["bilinear_slack"] = "[1/1024, 1/1024]"
        with self.assertRaisesRegex(ValueError, "not contained in accepted enclosure"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_division_by_interval_meeting_zero(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_rational_coefficient_extraction_candidate.json"
        )
        data = copy.deepcopy(data)
        data["interval_symbols"]["bad_denom"] = "[-1, 1]"
        data["coefficient_expressions"]["x*q"] = {
            "div": {"numerator": {"const": "1"}, "denominator": {"var": "bad_denom"}}
        }
        with self.assertRaisesRegex(ValueError, "meeting zero"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unbacked_kerr_gravitational_proved_claim(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_rational_coefficient_extraction_candidate.json"
        )
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "exact Kerr derivation backend"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
