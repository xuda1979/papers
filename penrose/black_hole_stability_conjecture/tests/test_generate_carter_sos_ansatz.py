import json
import unittest
from pathlib import Path

import analyze_carter_atlas as analyzer
import generate_carter_sos_ansatz as ansatz


ROOT = Path(__file__).resolve().parents[1]


class GenerateCarterSosAnsatzTests(unittest.TestCase):
    def load_analysis(self):
        with (ROOT / "certificates/carter_atlases/spin_half_candidate_atlas_subdiv4.json").open("r", encoding="utf-8") as handle:
            atlas = json.load(handle)
        return analyzer.analyze(atlas)

    def test_generates_for_best_chart(self):
        result = ansatz.generate(self.load_analysis())
        self.assertIn("monomial_basis", result)
        self.assertEqual(result["monomial_basis"][0], "1")
        self.assertIn("target_identity", result)

    def test_rejects_unknown_chart(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            ansatz.generate(self.load_analysis(), chart_name="missing")


if __name__ == "__main__":
    unittest.main()
