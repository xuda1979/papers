import json
import unittest
from fractions import Fraction
from pathlib import Path

import generate_fourier_certificate as generator


ROOT = Path(__file__).resolve().parents[1]


class GenerateFourierCertificateTests(unittest.TestCase):
    def test_toy_coefficients_have_positive_margin(self):
        with (ROOT / "certificates/fourier_inputs/toy_torus_coefficients.json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        result = generator.generate(data)
        self.assertTrue(result["accepted_by_margin_test"])
        self.assertEqual(result["resonant_average_lower_bound"], "[7/32, 7/32]")

    def test_rejects_missing_constant_mode(self):
        with self.assertRaisesRegex(ValueError, "constant mode"):
            generator.generate({"coefficients": {"1": "1/2"}, "resonances": [["0"]]})

    def test_triangle_bound_for_complex_coefficients(self):
        lower = generator.lower_bound_for_resonance({"0": "1", "1": ["1/4", "1/8"]}, ["0", "1"])
        self.assertEqual(lower, Fraction(5, 8))


if __name__ == "__main__":
    unittest.main()
