import copy
import json
import unittest
from pathlib import Path

import verify_carter_chart_certificate as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterChartCertificateTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_toy_clean_chart_certificate(self):
        result = verifier.verify_file(ROOT / "certificates/carter_chart_certificates/toy_clean_chart_certificate.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_level_carter_cone_certificate")
        self.assertEqual(result["target_density_kind"], "protocol_toy")
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["checked_identities"], ["two_square_decomposition"])
        self.assertEqual(result["gram_basis_size"], 3)
        self.assertEqual(result["checked_principal_minors"], 7)

    def test_accepts_spin_half_model_cone_certificate(self):
        result = verifier.verify_file(
            ROOT / "certificates/carter_chart_certificates/spin_half_clean_chart_model_cone_certificate.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["target_density_kind"], "chart_local_carter_normalized_model")
        self.assertEqual(result["checked_identities"], [
            "carter_normalized_model_square_expansion",
            "strict_delta_residual_identity",
        ])
        self.assertEqual(result["coercivity_delta"], "1/256")
        self.assertEqual(result["coercivity_residual_gram_basis_size"], 3)

    def test_rejects_bad_identity(self):
        with self.assertRaisesRegex(ValueError, "residual"):
            verifier.verify_file(ROOT / "certificates/carter_chart_certificates/bad_clean_chart_certificate.json")

    def test_rejects_chart_mismatch(self):
        data = self.load("certificates/carter_chart_certificates/toy_clean_chart_certificate.json")
        data["chart"] = "wrong_chart"
        with self.assertRaisesRegex(ValueError, "chart must match"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_identity_variable_mismatch(self):
        data = self.load("certificates/carter_chart_certificates/toy_clean_chart_certificate.json")
        data = copy.deepcopy(data)
        data["identity_certificate"]["variables"] = ["x", "ell", "z"]
        with self.assertRaisesRegex(ValueError, "identity variables"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_non_dominating_margin(self):
        data = self.load("certificates/carter_chart_certificates/toy_clean_chart_certificate.json")
        data["margins"]["positive_margin"] = "[1/128, 1/128]"
        with self.assertRaisesRegex(ValueError, "dominate"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_bad_gram_certificate(self):
        data = self.load("certificates/carter_chart_certificates/toy_clean_chart_certificate.json")
        data["gram_certificate"]["gram_matrix"][2][2] = "-1"
        with self.assertRaisesRegex(ValueError, "principal minor"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unbacked_kerr_gravitational_claim(self):
        data = self.load("certificates/carter_chart_certificates/toy_clean_chart_certificate.json")
        data = copy.deepcopy(data)
        data["metadata"]["target_density_kind"] = "kerr_gravitational"
        with self.assertRaisesRegex(ValueError, "kerr_gravitational requires"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_bad_coercivity_residual(self):
        data = self.load("certificates/carter_chart_certificates/spin_half_clean_chart_model_cone_certificate.json")
        data = copy.deepcopy(data)
        data["coercivity_certificate"]["residual_polynomial"]["q^2"] = "0"
        with self.assertRaisesRegex(ValueError, "coercivity residual identity"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
