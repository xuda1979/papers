import json
import unittest
from pathlib import Path

import analyze_carter_atlas as analyzer


ROOT = Path(__file__).resolve().parents[1]


class AnalyzeCarterAtlasTests(unittest.TestCase):
    def load_atlas(self):
        with (ROOT / "certificates/carter_atlases/spin_half_candidate_atlas_subdiv4.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_ranks_all_charts(self):
        result = analyzer.analyze(self.load_atlas())
        self.assertEqual(len(result["ranked_charts"]), 16)
        self.assertEqual(result["best_chart"], result["ranked_charts"][0]["chart"])

    def test_sign_class(self):
        self.assertEqual(analyzer.sign_class(analyzer.Interval(analyzer.Fraction(1), analyzer.Fraction(2))), "positive")
        self.assertEqual(analyzer.sign_class(analyzer.Interval(analyzer.Fraction(-2), analyzer.Fraction(-1))), "negative")
        self.assertEqual(analyzer.sign_class(analyzer.Interval(analyzer.Fraction(-1), analyzer.Fraction(1))), "mixed")


if __name__ == "__main__":
    unittest.main()
