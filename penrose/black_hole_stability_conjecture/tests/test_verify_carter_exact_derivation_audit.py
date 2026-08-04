import copy
import json
import unittest
from pathlib import Path

import verify_carter_exact_derivation_audit as verifier


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    "certificates/carter_chart_certificates/"
    "spin_half_clean_chart_exact_derivation_audit_candidate.json"
)


class VerifyCarterExactDerivationAuditTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_exact_derivation_audit_candidate(self):
        result = verifier.verify_file(ROOT / CERTIFICATE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_exact_derivation_audit")
        self.assertEqual(result["audit_status"], "candidate_audit")
        self.assertFalse(result["exact_derivation_complete"])
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertIn("radial_hessian_packet", result["primitive_entries_exact"])
        self.assertIn("superradiant_square_packet", result["primitive_entries_exact"])
        self.assertIn("carter_scale_packet", result["primitive_entries_exact"])
        self.assertIn("radial_trapped_constraint", result["primitive_entries_exact"])
        self.assertIn("radial_derivative_trapped_constraint", result["primitive_entries_exact"])
        self.assertIn("diagonal_slack_packet", result["primitive_entries_exact"])
        self.assertIn("bilinear_slack_packet", result["primitive_entries_exact"])
        # TCDEDT (2026-07-08) closed the last two primitive-packet entries.
        self.assertIn("trapped_radial_constraint_correction", result["primitive_entries_exact"])
        self.assertIn("trapped_derivative_constraint_correction", result["primitive_entries_exact"])
        # All 9 primitive-packet entries are now exact.
        self.assertEqual(len(result["primitive_entries_open"]), 0)
        self.assertEqual(len(result["primitive_entries_exact"]), 9)
        self.assertIn(
            "compute the normalized gravitational principal density q_grav in the listed phase variables",
            result["backend_obligations_open"],
        )
        self.assertIn(
            "derive rho_R and rho_dR as the trapped characteristic constraints R=0 and dR=0",
            result["backend_obligations_exact"],
        )
        self.assertEqual(
            result["checked_identities"][:2],
            [
                "all_trace_primitives_registered_for_exact_derivation",
                "all_backend_obligations_registered_for_exact_derivation",
            ],
        )

    def test_rejects_missing_primitive_registry_entry(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["primitive_derivation_registry"].pop()
        with self.assertRaisesRegex(ValueError, "primitive_derivation_registry must contain exactly"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unknown_primitive_registry_entry(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["primitive_derivation_registry"][0]["name"] = "not_a_trace_primitive"
        with self.assertRaisesRegex(ValueError, "primitive_derivation_registry names must match"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_bad_source_target(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["backend_obligation_registry"][0]["source_target"] = "handwave"
        with self.assertRaisesRegex(ValueError, "source_target"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_completion_upgrade_with_open_entries(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["metadata"]["audit_status"] = "exact_derivation_complete"
        with self.assertRaisesRegex(ValueError, "requires all primitive and obligation entries"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
