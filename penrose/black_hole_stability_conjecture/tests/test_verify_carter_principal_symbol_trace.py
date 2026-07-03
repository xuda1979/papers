import copy
import json
import unittest
from pathlib import Path

import verify_carter_principal_symbol_trace as verifier


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    "certificates/carter_chart_certificates/"
    "spin_half_clean_chart_principal_symbol_trace_candidate.json"
)


class VerifyCarterPrincipalSymbolTraceTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_spin_half_principal_symbol_trace_candidate(self):
        result = verifier.verify_file(ROOT / CERTIFICATE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "chart_principal_symbol_trace")
        self.assertEqual(result["derivation_status"], "candidate_trace")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["chart"], "spin_half_r_7_2_to_18_5_sub04")
        self.assertEqual(result["local_coercivity_delta"], "1/512")
        self.assertIn("radial_hessian_packet", result["checked_primitives"])
        self.assertEqual(
            result["checked_identities"][:2],
            [
                "principal_symbol_density_trace_matches_backend_source",
                "principal_symbol_constraint_traces_match_backend_constraints",
            ],
        )

    def test_rejects_density_trace_mismatch(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["derived_formulas"][-1]["expression"]["sum"].pop()
        with self.assertRaisesRegex(ValueError, "density trace does not match"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_phase_dependent_constraint_trace(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["primitive_formulas"][-2]["expression"] = {
            "prod": [
                {"var": "rho_R"},
                {"var": "x"}
            ]
        }
        with self.assertRaisesRegex(ValueError, "phase-free"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_cyclic_formula_reference(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["derived_formulas"].append(
            {
                "name": "cycle_a",
                "rule": "bad cycle probe",
                "expression": {"ref": "cycle_b"},
            }
        )
        data["derived_formulas"].append(
            {
                "name": "cycle_b",
                "rule": "bad cycle probe",
                "expression": {"ref": "cycle_a"},
            }
        )
        data["density_trace"] = {"ref": "cycle_a"}
        with self.assertRaisesRegex(ValueError, "cyclic trace formula reference"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")

    def test_rejects_unimplemented_gravitational_proof_upgrade(self):
        data = self.load(CERTIFICATE)
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "principal-symbol differentiation engine"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_chart_certificates")


if __name__ == "__main__":
    unittest.main()
