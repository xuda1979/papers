import copy
import json
import unittest
from pathlib import Path

import verify_carter_coefficient_robustness as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterCoefficientRobustnessTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_model_robustness_certificate(self):
        result = verifier.verify_file(
            ROOT / "certificates/carter_chart_certificates/spin_half_clean_chart_model_robustness_certificate.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["base_chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["base_coercivity_delta"], "1/256")
        self.assertEqual(result["worst_coefficient_loss"], "3/4096")
        self.assertEqual(result["remaining_delta"], "13/4096")
        self.assertEqual(result["target_final_delta"], "1/512")

    def test_rejects_excessive_coefficient_loss(self):
        with self.assertRaisesRegex(ValueError, "target_final_delta"):
            verifier.verify_file(
                ROOT / "certificates/carter_chart_certificates/bad_spin_half_clean_chart_model_robustness_certificate.json"
            )

    def test_rejects_uncontrolled_variable_loss(self):
        data = self.load("certificates/carter_chart_certificates/spin_half_clean_chart_model_robustness_certificate.json")
        data = copy.deepcopy(data)
        data["control_variables"] = ["x", "ell"]
        with self.assertRaisesRegex(ValueError, "uncontrolled variable"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
