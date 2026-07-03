import copy
import json
import tempfile
import unittest
from pathlib import Path

import verify_carter_degenerate_seed_bridge as verifier


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "certificates/carter_identities/degenerate_seed_bridge_spin_half_certificate.json"


class VerifyCarterDegenerateSeedBridgeTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_verifies_stored_bridge_package(self):
        result = verifier.verify_package(PACKAGE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["bridge_count"], 3)
        self.assertEqual(
            [bridge["containing_generic_chart"] for bridge in result["bridges"]],
            [
                "spin_half_r_5_2_to_3_sub04",
                "spin_half_r_7_3_to_5_2_sub01",
                "spin_half_r_7_2_to_18_5_sub02",
            ],
        )

    def test_computed_bridge_contains_regular_seed_data(self):
        atlas = self.load("certificates/carter_atlases/spin_half_candidate_atlas_subdiv4.json")
        polar = self.load("certificates/carter_identities/polar_spin_half_seed_certificate.json")
        equatorial = self.load("certificates/carter_identities/equatorial_spin_half_seed_certificate.json")
        certificate = verifier.compute_bridge_certificate(atlas, polar, equatorial)
        polar_bridge = certificate["bridges"][0]
        self.assertEqual(polar_bridge["seed_family"], "polar_L_equals_zero")
        self.assertEqual(polar_bridge["regular_seed_values"]["lambda_interval"], "[0, 0]")
        self.assertIn(
            "regular_seed_lambda_eta_hessian_inside_generic_bridge_values",
            polar_bridge["verified_containments"],
        )

        prograde_bridge = certificate["bridges"][1]
        self.assertEqual(prograde_bridge["seed_family"], "equatorial_Q_equals_zero_prograde")
        self.assertEqual(prograde_bridge["regular_seed_values"]["eta_interval"], "[0, 0]")

    def test_rejects_stale_stored_bridge(self):
        stale = copy.deepcopy(self.load("certificates/carter_identities/degenerate_seed_bridge_spin_half_certificate.json"))
        stale["bridges"][1]["containing_generic_chart"] = "not_the_chart"
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=PACKAGE.parent,
            delete=False,
        )
        try:
            with handle:
                json.dump(stale, handle, indent=2, sort_keys=True)
            with self.assertRaisesRegex(ValueError, "stale or incorrect"):
                verifier.verify_package(Path(handle.name))
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def test_rejects_missing_containing_chart(self):
        atlas = self.load("certificates/carter_atlases/spin_half_candidate_atlas_subdiv4.json")
        atlas["charts"] = [
            chart
            for chart in atlas["charts"]
            if chart["chart"] != "spin_half_r_5_2_to_3_sub04"
        ]
        polar = self.load("certificates/carter_identities/polar_spin_half_seed_certificate.json")
        equatorial = self.load("certificates/carter_identities/equatorial_spin_half_seed_certificate.json")
        with self.assertRaisesRegex(ValueError, "no generic radial chart contains polar_L_equals_zero"):
            verifier.compute_bridge_certificate(atlas, polar, equatorial)


if __name__ == "__main__":
    unittest.main()
