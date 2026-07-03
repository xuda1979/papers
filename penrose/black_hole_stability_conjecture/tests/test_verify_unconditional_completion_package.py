import copy
import json
import unittest
from pathlib import Path

import verify_unconditional_completion_package as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyUnconditionalCompletionPackageTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def toy(self):
        return self.load("certificates/completion_packages/toy_unconditional_completion_package.json")

    def test_accepts_toy_completion_package(self):
        result = verifier.verify_file(ROOT / "certificates/completion_packages/toy_unconditional_completion_package.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["total_derivative_loss"], 6)
        self.assertIn("trapped_superradiant_morawetz", result["checked_blocks"])

    def test_rejects_open_block(self):
        with self.assertRaisesRegex(ValueError, "status must be proved"):
            verifier.verify_file(ROOT / "certificates/completion_packages/bad_open_completion_package.json")

    def test_audit_collects_multiple_current_status_errors(self):
        data = self.load("certificates/completion_packages/current_public_status_package.json")
        result = verifier.audit(data)
        self.assertFalse(result["accepted"])
        self.assertGreaterEqual(len(result["errors"]), 7)
        self.assertTrue(any("proof_blocks[0].status must be proved" in error for error in result["errors"]))
        self.assertTrue(any("all proof blocks must use the same normalization" in error for error in result["errors"]))

    def test_rejects_missing_required_block(self):
        data = self.toy()
        data["proof_blocks"] = data["proof_blocks"][:-1]
        with self.assertRaisesRegex(ValueError, "missing required blocks"):
            verifier.verify(data)

    def test_rejects_incompatible_gauge(self):
        data = self.toy()
        data["proof_blocks"][0]["gauge"] = "different gauge"
        with self.assertRaisesRegex(ValueError, "must match normalization.gauge"):
            verifier.verify(data)

    def test_rejects_excessive_derivative_loss(self):
        data = self.toy()
        data["compatibility"]["available_derivative_buffer"] = 5
        with self.assertRaisesRegex(ValueError, "available_derivative_buffer"):
            verifier.verify(data)

    def test_rejects_non_dominating_margin(self):
        data = self.toy()
        data = copy.deepcopy(data)
        data["proof_blocks"][1]["positive_margin"] = "[1/128, 1/128]"
        with self.assertRaisesRegex(ValueError, "dominate"):
            verifier.verify(data)

    def test_rejects_uniform_extremal_claim(self):
        data = self.toy()
        data["near_extremal_policy"]["uniform_to_extremality_claimed"] = True
        with self.assertRaisesRegex(ValueError, "uniform-to-extremality"):
            verifier.verify(data)


if __name__ == "__main__":
    unittest.main()
