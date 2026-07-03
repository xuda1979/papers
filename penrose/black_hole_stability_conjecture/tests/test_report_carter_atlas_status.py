import copy
import json
import unittest
from pathlib import Path

import report_carter_atlas_status as report


ROOT = Path(__file__).resolve().parents[1]


class ReportCarterAtlasStatusTests(unittest.TestCase):
    def load(self):
        path = ROOT / "certificates/carter_atlases/spin_half_atlas_status.json"
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_renders_missing_degenerate_families(self):
        data = self.load()
        text = report.render_markdown(data, base_dir=ROOT / "certificates/carter_atlases")
        self.assertIn("# Carter Atlas Completeness Status", text)
        self.assertIn("generic_nonzero_spin_nonpolar_radial", text)
        self.assertIn("polar_L_equals_zero", text)
        self.assertIn("equatorial_Q_equals_zero", text)
        self.assertIn("seed_angular_overlap_policy_verified", text)
        self.assertIn("zero_spin_schwarzschild_limit", text)
        self.assertIn("interval_transition_seed_verified", text)
        self.assertIn("Local Certification Audit", text)
        self.assertIn("near_extremal_uniformity", text)
        self.assertIn("not yet fully verified", text)

    def test_audit_records_blocking_families(self):
        result = report.audit(self.load(), base_dir=ROOT / "certificates/carter_atlases")
        self.assertFalse(result["accepted"])
        self.assertIn("polar_L_equals_zero", result["blocking_families"])
        self.assertIn("equatorial_Q_equals_zero", result["blocking_families"])
        self.assertIn("zero_spin_schwarzschild_limit", result["blocking_families"])
        self.assertEqual(result["errors"], [])

    def test_local_certification_audit_only_blocks_missing_local_evidence(self):
        result = report.local_certification_audit(self.load(), base_dir=ROOT / "certificates/carter_atlases")
        self.assertFalse(result["accepted"])
        self.assertIn("near_extremal_uniformity", result["locally_open_families"])
        self.assertNotIn("polar_L_equals_zero", result["locally_open_families"])
        self.assertNotIn("equatorial_Q_equals_zero", result["locally_open_families"])
        self.assertNotIn("zero_spin_schwarzschild_limit", result["locally_open_families"])

    def test_rejects_duplicate_families(self):
        data = self.load()
        data["chart_families"].append(copy.deepcopy(data["chart_families"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate chart family"):
            report.render_markdown(data, base_dir=ROOT / "certificates/carter_atlases")


if __name__ == "__main__":
    unittest.main()
