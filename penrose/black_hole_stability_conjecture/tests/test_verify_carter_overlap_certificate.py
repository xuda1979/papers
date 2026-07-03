import copy
import json
import unittest
from fractions import Fraction
from pathlib import Path

import generate_carter_atlas as atlas
import verify_carter_overlap_certificate as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterOverlapCertificateTests(unittest.TestCase):
    def overlap_atlas(self):
        spin = Fraction(1, 2)
        digits = 12
        left = atlas.generate_chart(
            spin,
            {"chart": "left_overlap_chart"},
            digits,
            r_override=atlas.Interval(Fraction(7, 3), Fraction(5, 2)),
        )
        right = atlas.generate_chart(
            spin,
            {"chart": "right_overlap_chart"},
            digits,
            r_override=atlas.Interval(Fraction(12, 5), Fraction(8, 3)),
        )
        return {
            "metadata": {
                "name": "toy overlapping Carter atlas",
                "spin": "1/2",
                "mass_normalization": "M=1",
                "sqrt_enclosure_decimal_digits": digits,
                "status": "toy overlap atlas; not a Kerr proof certificate",
            },
            "charts": [left, right],
        }

    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_generated_overlap(self):
        result = verifier.verify(self.overlap_atlas())
        self.assertTrue(result["accepted"])
        self.assertEqual(result["overlap_count"], 1)
        self.assertEqual(result["overlaps"][0]["overlap_interval"], "[12/5, 5/2]")

    def test_accepts_exact_stored_overlap_certificate(self):
        data = self.overlap_atlas()
        data["overlap_certificates"] = verifier.computed_overlap_certificates(data)
        result = verifier.verify(data, require_certificates=True)
        self.assertEqual(result["overlap_count"], 1)

    def test_rejects_missing_required_overlap_certificates(self):
        with self.assertRaisesRegex(ValueError, "overlap_certificates"):
            verifier.verify(self.overlap_atlas(), require_certificates=True)

    def test_rejects_stale_overlap_certificate(self):
        data = self.overlap_atlas()
        data["overlap_certificates"] = verifier.computed_overlap_certificates(data)
        data["overlap_certificates"][0]["overlap_values"]["lambda_interval"] = "[0, 1]"
        with self.assertRaisesRegex(ValueError, "stale or incorrect"):
            verifier.verify(data)

    def test_rejects_chart_interval_missing_overlap_values(self):
        data = self.overlap_atlas()
        data["charts"][0]["lambda_interval"] = "[0, 1]"
        with self.assertRaisesRegex(ValueError, "lambda_interval misses overlap interval"):
            verifier.verify(data)

    def test_accepts_nonoverlapping_subdivided_atlas(self):
        data = self.load("certificates/carter_atlases/spin_half_candidate_atlas_subdiv4.json")
        result = verifier.verify(data)
        self.assertEqual(result["overlap_count"], 0)

    def test_rejects_duplicate_chart_names(self):
        data = self.overlap_atlas()
        data = copy.deepcopy(data)
        data["charts"][1]["chart"] = data["charts"][0]["chart"]
        with self.assertRaisesRegex(ValueError, "duplicate chart name"):
            verifier.verify(data)


if __name__ == "__main__":
    unittest.main()
