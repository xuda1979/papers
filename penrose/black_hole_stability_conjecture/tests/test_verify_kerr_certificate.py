import json
import unittest
from pathlib import Path

import verify_kerr_certificate as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyKerrCertificateTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_toy_certificate(self):
        verifier.validate(self.load("certificates/examples/toy_certificate.json"))

    def test_rejects_negative_margin_certificate(self):
        with self.assertRaisesRegex(ValueError, "dominate"):
            verifier.validate(self.load("certificates/examples/bad_negative_margin.json"))

    def test_interval_dominance_is_strict(self):
        left = verifier.Interval(verifier.Fraction(1, 8), verifier.Fraction(1, 4))
        right = verifier.Interval(verifier.Fraction(0), verifier.Fraction(1, 16))
        self.assertTrue(left.strictly_dominates(right))
        self.assertFalse(right.strictly_dominates(left))


if __name__ == "__main__":
    unittest.main()
