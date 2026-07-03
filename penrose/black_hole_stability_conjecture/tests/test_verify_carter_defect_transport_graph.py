import copy
import json
import unittest
from pathlib import Path

import verify_carter_defect_transport_graph as verifier


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    "certificates/carter_chart_certificates/"
    "spin_half_clean_chart_defect_transport_graph_candidate.json"
)


class VerifyCarterDefectTransportGraphTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_defect_transport_graph_candidate(self):
        result = verifier.verify_file(ROOT / CERTIFICATE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "finite_defect_transport_graph")
        self.assertEqual(result["derivation_status"], "candidate_transport_graph")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(
            result["certified_charts"],
            [
                "spin_half_r_7_2_to_18_5_sub04",
                "spin_half_r_7_2_to_18_5_sub04_transport_shadow",
                "spin_half_r_7_2_to_18_5_sub04_transport_shadow_second",
            ],
        )
        self.assertEqual(result["minimum_local_coercivity_delta"], "1/512")
        self.assertEqual(
            result["checked_edges"],
            ["first shadow transport edge", "second shadow transport edge"],
        )
        self.assertEqual(
            result["chart_deltas"]["spin_half_r_7_2_to_18_5_sub04_transport_shadow_second"],
            "1/512",
        )
        self.assertEqual(
            result["chart_paths"]["spin_half_r_7_2_to_18_5_sub04_transport_shadow_second"],
            [
                "spin_half_r_7_2_to_18_5_sub04",
                "spin_half_r_7_2_to_18_5_sub04_transport_shadow",
                "spin_half_r_7_2_to_18_5_sub04_transport_shadow_second",
            ],
        )

    def test_rejects_unknown_edge_source(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["transport_edges"][1]["source_chart"] = "not_yet_certified"
        with self.assertRaisesRegex(ValueError, "has not been certified"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_duplicate_target_chart(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["transport_edges"][1]["target_chart"] = "spin_half_r_7_2_to_18_5_sub04_transport_shadow"
        with self.assertRaisesRegex(ValueError, "duplicates an existing certified chart"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_missing_required_target(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["required_target_charts"].append("missing_chart")
        with self.assertRaisesRegex(ValueError, "required target chart"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_minimum_delta_above_transport_product(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["minimum_required_delta"] = "1/256"
        with self.assertRaisesRegex(ValueError, "below required"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_edge_density_mismatch(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["transport_edges"][1]["target_density_expression"]["sum"].pop()
        with self.assertRaisesRegex(ValueError, "transport density identity failed"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_graph_gravitational_proof_upgrade(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "every graph node"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_edge_gravitational_proof_upgrade(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["transport_edges"][0]["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "edge target chart"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
