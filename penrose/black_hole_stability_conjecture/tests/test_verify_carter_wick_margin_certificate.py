import copy
import json
import unittest
from pathlib import Path

import verify_carter_wick_margin_certificate as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterWickMarginCertificateTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_toy_margin_certificate(self):
        result = verifier.verify_file(ROOT / "certificates/examples/toy_carter_wick_margin_certificate.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "carter_wick_finite_margin")
        self.assertEqual(result["total_quantization_loss"], "1/1024")
        self.assertEqual(result["exported_quantization_loss_upper"], "1/1024")

    def test_accepts_spin_half_model_margin_certificate(self):
        result = verifier.verify_file(
            ROOT / "certificates/examples/spin_half_clean_chart_model_carter_wick_margin_certificate.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["total_quantization_loss"], "1/2048")
        self.assertEqual(result["exported_quantization_loss_upper"], "1/2048")

    def test_rejects_exported_loss_that_misses_total(self):
        with self.assertRaisesRegex(ValueError, "exported quantization loss"):
            verifier.verify_file(ROOT / "certificates/examples/bad_carter_wick_margin_certificate.json")

    def test_rejects_false_contract_field(self):
        data = self.load("certificates/examples/toy_carter_wick_margin_certificate.json")
        data = copy.deepcopy(data)
        data["calculus_contract"]["weyl_commutator_principal_symbol"] = False
        with self.assertRaisesRegex(ValueError, "must be true"):
            verifier.verify(data)

    def test_rejects_noncompact_symbol_neighborhood(self):
        data = self.load("certificates/examples/toy_carter_wick_margin_certificate.json")
        data = copy.deepcopy(data)
        data["symbol_class"]["compact_phase_neighborhood"] = False
        with self.assertRaisesRegex(ValueError, "compact_phase_neighborhood"):
            verifier.verify(data)


if __name__ == "__main__":
    unittest.main()
