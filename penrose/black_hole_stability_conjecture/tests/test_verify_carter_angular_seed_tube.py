import copy
import json
import tempfile
import unittest
from pathlib import Path

import verify_carter_angular_seed_tube as verifier


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "certificates/carter_identities/angular_seed_tube_spin_half_certificate.json"
PATCH = ROOT / "certificates/carter_identities/angular_patch_coverage_spin_half_certificate.json"


class VerifyCarterAngularSeedTubeTests(unittest.TestCase):
    def load(self, path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_sources(self):
        patch = self.load(PATCH)
        overlap_path = PATCH.parent / patch["metadata"]["angular_overlap_path"]
        overlap = self.load(overlap_path)
        corridor_path = overlap_path.parent / overlap["metadata"]["angular_corridor_path"]
        corridor = self.load(corridor_path)
        bridge_path = corridor_path.parent / corridor["metadata"]["degenerate_seed_bridge_path"]
        return patch, self.load(bridge_path)

    def test_verifies_stored_seed_tube_package(self):
        result = verifier.verify_package(PACKAGE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["polar_seed_tube_count"], 1)
        self.assertEqual(result["equatorial_seed_tube_count"], 2)
        self.assertIn("polar_two_sided_parameter_seed_tube", result["polar_seed_tubes"])
        self.assertIn("equatorial_retrograde_eta_zeta_seed_tube", result["equatorial_seed_tubes"])

    def test_computed_seed_tube_records_thickened_margins(self):
        patch, bridge = self.load_sources()
        certificate = verifier.compute_certificate(patch, bridge)
        self.assertEqual(certificate["symbolic_identities"]["polar_seed_tube_regularization"], "0")
        self.assertEqual(certificate["symbolic_identities"]["equatorial_seed_tube_factorization"], "0")

        polar = certificate["polar_seed_tubes"][0]
        self.assertEqual(polar["tube_type"], "polar_two_sided_lambda_thickening")
        self.assertEqual(polar["lambda_tube_interval"], "[-1/6000, 1/6000]")
        self.assertIn("ordinary_annulus_Theta_positive_for_full_bridge_lambda", polar["checked_conditions"])
        self.assertIn("two_sided_lambda_tube_inside_radial_bridge", polar["checked_conditions"])

        equatorial = certificate["equatorial_seed_tubes"][0]
        self.assertEqual(equatorial["tube_type"], "equatorial_eta_zeta_thickening")
        self.assertEqual(equatorial["eta_tube_interval"], "[0, 1/1024]")
        self.assertEqual(equatorial["zeta_tube_interval"], "[0, 1/48]")
        self.assertIn("equatorial_factor_positive_on_eta_tube", equatorial["checked_conditions"])
        self.assertIn("ordinary_overlap_Theta_positive", equatorial["checked_conditions"])

    def test_rejects_stale_seed_tube_package(self):
        stale = copy.deepcopy(self.load(PACKAGE))
        stale["equatorial_seed_tubes"][0]["zeta_tube_interval"] = "[0, 1/32]"
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

    def test_rejects_missing_bridge(self):
        patch, bridge = self.load_sources()
        bridge["bridges"] = []
        with self.assertRaisesRegex(ValueError, "bridge package must contain bridges"):
            verifier.compute_certificate(patch, bridge)


if __name__ == "__main__":
    unittest.main()
