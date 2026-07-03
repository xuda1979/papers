import json
import unittest
from fractions import Fraction
from pathlib import Path

import generate_carter_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]


class GenerateCarterAtlasTests(unittest.TestCase):
    def test_spin_half_candidate_charts_generate(self):
        with (ROOT / "certificates/carter_inputs/spin_half_radial_charts.json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        result = atlas.generate(data)
        self.assertEqual(result["metadata"]["spin"], "1/2")
        self.assertEqual(len(result["charts"]), 4)
        self.assertEqual(len(result["transitions"]), 3)
        self.assertEqual(result["transitions"][0]["shared_endpoint"], "5/2")
        self.assertEqual(result["transitions"][0]["endpoint_values"]["lambda"], "3")
        self.assertEqual(result["transitions"][0]["endpoint_values"]["eta"], "125/12")
        for chart in result["charts"]:
            self.assertIn("lambda_interval", chart)
            self.assertIn("eta_interval", chart)
            self.assertIn("trapped_scale_interval", chart)
            certificates = chart["denominator_certificates"]
            for key in ["a", "r_minus_M", "(r_minus_M)^2", "Delta", "r_plus"]:
                self.assertTrue(certificates[key]["avoids_zero"], key)
                self.assertNotEqual(certificates[key]["lower_abs_bound"], "0", key)

    def test_lambda_interval_matches_point_formula(self):
        r = atlas.point(Fraction(3))
        a = atlas.point(Fraction(1, 2))
        lam = atlas.lambda_interval(r, a)
        self.assertEqual(lam.lo, lam.hi)
        self.assertEqual(lam.lo, Fraction(-1))

    def test_eta_interval_matches_point_formula(self):
        r = atlas.point(Fraction(3))
        a = atlas.point(Fraction(1, 2))
        lam = atlas.lambda_interval(r, a)
        eta = atlas.eta_interval(r, a, lam)
        self.assertEqual(eta.lo, eta.hi)
        self.assertEqual(eta.lo, Fraction(27))

    def test_rejects_chart_crossing_r_equals_m(self):
        with self.assertRaisesRegex(ValueError, "crosses r=M"):
            atlas.generate({"spin": "1/2", "radial_charts": [{"r_interval": "[1/2, 3/2]"}]})

    def test_rejects_zero_spin_denominator(self):
        with self.assertRaisesRegex(ValueError, "requires nonzero spin"):
            atlas.generate({"spin": "0", "radial_charts": [{"r_interval": "[3, 4]"}]})

    def test_denominator_certificate_detects_mixed_interval(self):
        cert = atlas.sign_certificate(atlas.Interval(Fraction(-1), Fraction(2)))
        self.assertFalse(cert["avoids_zero"])
        self.assertEqual(cert["sign"], "mixed")
        self.assertEqual(cert["lower_abs_bound"], "0")


if __name__ == "__main__":
    unittest.main()
