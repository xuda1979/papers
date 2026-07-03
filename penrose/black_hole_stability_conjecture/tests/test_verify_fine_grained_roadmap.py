import copy
import json
import unittest
from pathlib import Path

import verify_fine_grained_roadmap as verifier


ROOT = Path(__file__).resolve().parents[1]
ROADMAP = "certificates/completion_packages/fine_grained_completion_roadmap.json"


class VerifyFineGrainedRoadmapTests(unittest.TestCase):
    def load(self):
        with (ROOT / ROADMAP).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_fine_grained_roadmap(self):
        result = verifier.verify_file(ROOT / ROADMAP)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "fine_grained_completion_roadmap")
        self.assertEqual(result["node_count"], 49)
        self.assertEqual(result["status_counts"]["proved"], 1)
        self.assertIn("FG-02a1a-variable-inventory", result["frontier"])
        self.assertIn("FG-03a1a-raw-superradiant-factors", result["frontier"])
        self.assertIn("FG-04a1a-angular-variable-inventory", result["frontier"])
        self.assertIn("FG-15a-teukolsky-boundary-contract", result["frontier"])

    def test_rejects_unknown_dependency(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["nodes"][1]["dependencies"].append("missing-node")
        with self.assertRaisesRegex(ValueError, "unknown node"):
            verifier.verify(data)

    def test_rejects_dependency_cycle(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["nodes"][0]["dependencies"].append("FG-02a1-gauge-principal-contract")
        with self.assertRaisesRegex(ValueError, "cycle detected"):
            verifier.verify(data)

    def test_rejects_gap_without_blocker(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["nodes"][1]["blockers"] = []
        with self.assertRaisesRegex(ValueError, "lists no blockers"):
            verifier.verify(data)


if __name__ == "__main__":
    unittest.main()
