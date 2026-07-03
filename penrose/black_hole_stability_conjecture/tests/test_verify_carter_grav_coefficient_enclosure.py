import copy
import json
import unittest
from pathlib import Path

import verify_carter_grav_coefficient_enclosure as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterGravCoefficientEnclosureTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_candidate_inside_tube(self):
        result = verifier.verify_file(
            ROOT / "certificates/carter_chart_certificates/spin_half_clean_chart_grav_coefficient_enclosure_candidate.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_grav_coefficient_enclosure")
        self.assertEqual(result["derivation_status"], "candidate_enclosure")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["base_coercivity_delta"], "1/256")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertEqual(result["coefficient_deviations"]["x^2"], "1/4096")
        self.assertEqual(result["coefficient_deviations"]["x*q"], "1/8192")

    def test_rejects_candidate_outside_tube(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_grav_coefficient_enclosure_candidate.json"
        )
        data = copy.deepcopy(data)
        data["coefficient_intervals"]["x*q"] = "[1/1024, 1/1024]"
        with self.assertRaisesRegex(ValueError, "leaves certified tube"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unbacked_kerr_gravitational_proved_claim(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_grav_coefficient_enclosure_candidate.json"
        )
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "exact Kerr derivation backend"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_missing_quadratic_coefficient(self):
        data = self.load(
            "certificates/carter_chart_certificates/spin_half_clean_chart_grav_coefficient_enclosure_candidate.json"
        )
        data = copy.deepcopy(data)
        del data["coefficient_intervals"]["ell*q"]
        with self.assertRaisesRegex(ValueError, "exactly"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
