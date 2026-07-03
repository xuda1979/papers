import copy
import json
import tempfile
import unittest
from pathlib import Path

import verify_carter_master_certificate as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterMasterCertificateTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_toy_master_certificate(self):
        result = verifier.verify_file(ROOT / "certificates/examples/toy_carter_master_certificate.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "carter_master_certificate")
        self.assertEqual(result["finite_checked_charts"], ["spin_half_r_7_2_to_18_5_sub04"])
        self.assertEqual(result["wick_exported_quantization_loss"], "1/1024")
        self.assertEqual(result["finite_quantization_loss_bound"], "1/1024")

    def test_accepts_spin_half_model_master_certificate(self):
        result = verifier.verify_file(
            ROOT / "certificates/examples/spin_half_clean_chart_model_carter_master_certificate.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["finite_checked_charts"], ["spin_half_r_7_2_to_18_5_sub04"])
        self.assertEqual(result["wick_exported_quantization_loss"], "1/2048")
        self.assertEqual(result["finite_quantization_loss_bound"], "1/2048")

    def test_rejects_bad_wick_certificate(self):
        with self.assertRaisesRegex(ValueError, "exported quantization loss"):
            verifier.verify_file(ROOT / "certificates/examples/bad_carter_master_certificate.json")

    def test_rejects_derivative_loss_over_budget(self):
        data = self.load("certificates/examples/toy_carter_master_certificate.json")
        data = copy.deepcopy(data)
        data["compatibility"]["derivative_loss_budget"] = 0
        with self.assertRaisesRegex(ValueError, "derivative_loss_budget"):
            verifier.verify(data, base_dir=ROOT / "certificates/examples")

    def test_rejects_finite_package_that_does_not_include_wick_loss(self):
        master = copy.deepcopy(self.load("certificates/examples/toy_carter_master_certificate.json"))
        finite = copy.deepcopy(self.load("certificates/examples/toy_finite_carter_cone_package.json"))
        finite["assembly"]["quantization_loss_bound"] = "[0, 1/2048]"
        finite["local_cone_certificates"][0]["certificate_path"] = str(
            ROOT / "certificates/carter_chart_certificates/toy_clean_chart_certificate.json"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            finite_path = tmp / "finite.json"
            finite_path.write_text(json.dumps(finite), encoding="utf-8")
            master["finite_cone_package_path"] = str(finite_path)
            with self.assertRaisesRegex(ValueError, "quantization_loss_bound"):
                verifier.verify(master, base_dir=ROOT / "certificates/examples")


if __name__ == "__main__":
    unittest.main()
