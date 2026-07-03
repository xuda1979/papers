import copy
import json
import unittest
from pathlib import Path

import verify_carter_finite_cone_package as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterFiniteConePackageTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_toy_finite_cone_package(self):
        result = verifier.verify_file(ROOT / "certificates/examples/toy_finite_carter_cone_package.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "finite_carter_cone_assembly")
        self.assertEqual(result["checked_charts"], ["spin_half_r_7_2_to_18_5_sub04"])
        self.assertEqual(result["minimum_local_slack"], "3/64")
        self.assertEqual(result["total_assembly_error"], "15/1024")

    def test_accepts_spin_half_model_finite_cone_package(self):
        result = verifier.verify_file(
            ROOT / "certificates/examples/spin_half_clean_chart_model_finite_carter_cone_package.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["checked_charts"], ["spin_half_r_7_2_to_18_5_sub04"])
        self.assertEqual(result["minimum_local_slack"], "3/512")
        self.assertEqual(result["total_assembly_error"], "5/4096")
        self.assertEqual(result["global_delta_lower"], "1/256")
        self.assertEqual(result["minimum_coercivity_delta"], "1/256")

    def test_rejects_global_delta_above_local_coercivity_delta(self):
        data = self.load("certificates/examples/spin_half_clean_chart_model_finite_carter_cone_package.json")
        data = copy.deepcopy(data)
        data["assembly"]["global_delta"] = "[1/128, 1/128]"
        with self.assertRaisesRegex(ValueError, "global_delta"):
            verifier.verify(data, base_dir=ROOT / "certificates/examples")

    def test_rejects_bad_assembly_margin(self):
        with self.assertRaisesRegex(ValueError, "assembly errors"):
            verifier.verify_file(ROOT / "certificates/examples/bad_finite_carter_cone_package.json")

    def test_rejects_nonzero_partition_residual(self):
        data = self.load("certificates/examples/toy_finite_carter_cone_package.json")
        data = copy.deepcopy(data)
        data["quadratic_partition"]["sum_square_residual"] = "[0, 1/1024]"
        with self.assertRaisesRegex(ValueError, "sum_square_residual"):
            verifier.verify(data, base_dir=ROOT / "certificates/examples")

    def test_rejects_chart_mismatch(self):
        data = self.load("certificates/examples/toy_finite_carter_cone_package.json")
        data = copy.deepcopy(data)
        data["local_cone_certificates"][0]["chart"] = "wrong_chart"
        with self.assertRaisesRegex(ValueError, "must match referenced certificate chart"):
            verifier.verify(data, base_dir=ROOT / "certificates/examples")


if __name__ == "__main__":
    unittest.main()
