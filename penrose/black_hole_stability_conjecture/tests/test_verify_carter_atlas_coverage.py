import copy
import json
import unittest
from fractions import Fraction
from pathlib import Path

import verify_carter_atlas_coverage as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterAtlasCoverageTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_base_atlas_coverage(self):
        data = self.load("certificates/carter_atlases/spin_half_candidate_atlas.json")
        result = verifier.verify(data, expected_start=Fraction(7, 3), expected_end=Fraction(18, 5))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["chart_count"], 4)
        self.assertEqual(len(result["transitions"]), 3)

    def test_accepts_spin_half_subdivided_atlas_coverage(self):
        data = self.load("certificates/carter_atlases/spin_half_candidate_atlas_subdiv4.json")
        result = verifier.verify(data, expected_start=Fraction(7, 3), expected_end=Fraction(18, 5))
        self.assertEqual(result["chart_count"], 16)
        self.assertEqual(len(result["transitions"]), 15)

    def test_rejects_gap(self):
        data = self.load("certificates/carter_atlases/spin_half_candidate_atlas.json")
        data = copy.deepcopy(data)
        data["charts"][1]["r_interval"] = "[8/3, 3]"
        with self.assertRaisesRegex(ValueError, "gap"):
            verifier.verify(data)

    def test_rejects_overlap(self):
        data = self.load("certificates/carter_atlases/spin_half_candidate_atlas.json")
        data = copy.deepcopy(data)
        data["charts"][1]["r_interval"] = "[12/5, 3]"
        with self.assertRaisesRegex(ValueError, "overlap"):
            verifier.verify(data)

    def test_rejects_unexpected_endpoint(self):
        data = self.load("certificates/carter_atlases/spin_half_candidate_atlas.json")
        with self.assertRaisesRegex(ValueError, "expected"):
            verifier.verify(data, expected_start=Fraction(2), expected_end=Fraction(18, 5))

    def test_rejects_missing_transition_list(self):
        data = copy.deepcopy(self.load("certificates/carter_atlases/spin_half_candidate_atlas.json"))
        del data["transitions"]
        with self.assertRaisesRegex(ValueError, "transitions list"):
            verifier.verify(data)

    def test_rejects_stale_transition_endpoint_value(self):
        data = copy.deepcopy(self.load("certificates/carter_atlases/spin_half_candidate_atlas.json"))
        data["transitions"][0]["endpoint_values"]["lambda"] = "0"
        with self.assertRaisesRegex(ValueError, "stale or incorrect"):
            verifier.verify(data)

    def test_rejects_transition_not_in_sorted_order(self):
        data = copy.deepcopy(self.load("certificates/carter_atlases/spin_half_candidate_atlas.json"))
        data["transitions"][0], data["transitions"][1] = data["transitions"][1], data["transitions"][0]
        with self.assertRaisesRegex(ValueError, "sorted atlas order"):
            verifier.verify(data)

    def test_rejects_chart_interval_missing_transition_value(self):
        data = copy.deepcopy(self.load("certificates/carter_atlases/spin_half_candidate_atlas.json"))
        data["charts"][0]["lambda_interval"] = "[0, 1]"
        with self.assertRaisesRegex(ValueError, "lambda interval misses endpoint value"):
            verifier.verify(data)

    def test_rejects_duplicate_chart_names(self):
        data = copy.deepcopy(self.load("certificates/carter_atlases/spin_half_candidate_atlas.json"))
        data["charts"][1]["chart"] = data["charts"][0]["chart"]
        with self.assertRaisesRegex(ValueError, "duplicate chart name"):
            verifier.verify(data)


if __name__ == "__main__":
    unittest.main()
