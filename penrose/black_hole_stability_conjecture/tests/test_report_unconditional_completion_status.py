import json
import unittest
from pathlib import Path

import report_unconditional_completion_status as report


ROOT = Path(__file__).resolve().parents[1]


class ReportUnconditionalCompletionStatusTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_renders_current_public_status(self):
        data = self.load("certificates/completion_packages/current_public_status_package.json")
        text = report.render_markdown(data)
        self.assertIn("# Unconditional Kerr Completion Status", text)
        self.assertIn("curvature_to_metric_reconstruction", text)
        self.assertIn("seed-level dependent angular overlap-policy certificates", text)
        self.assertIn("complete angular-torus atlas", text)
        self.assertIn("Verifier Audit", text)
        self.assertIn("proof_blocks[0].status must be proved", text)
        self.assertIn("proof_blocks[6].status must be proved", text)

    def test_rejects_duplicate_blocks(self):
        data = self.load("certificates/completion_packages/current_public_status_package.json")
        data["proof_blocks"].append(dict(data["proof_blocks"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate proof block"):
            report.render_markdown(data)


if __name__ == "__main__":
    unittest.main()
