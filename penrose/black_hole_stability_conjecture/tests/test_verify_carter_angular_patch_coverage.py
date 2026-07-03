import copy
import json
import tempfile
import unittest
from pathlib import Path

import verify_carter_angular_patch_coverage as verifier


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "certificates/carter_identities/angular_patch_coverage_spin_half_certificate.json"
OVERLAP = ROOT / "certificates/carter_identities/angular_overlap_spin_half_certificate.json"


class VerifyCarterAngularPatchCoverageTests(unittest.TestCase):
    def load(self, path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_sources(self):
        overlap = self.load(OVERLAP)
        corridor_path = OVERLAP.parent / overlap["metadata"]["angular_corridor_path"]
        return overlap, self.load(corridor_path)

    def test_verifies_stored_patch_coverage_package(self):
        result = verifier.verify_package(PACKAGE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["polar_patch_count"], 2)
        self.assertEqual(result["equatorial_patch_count"], 2)
        self.assertIn("polar_positive_L_full_v_patch", result["polar_patch_cover"])
        self.assertIn("equatorial_retrograde_extended_zeta_patch", result["equatorial_patch_cover"])

    def test_computed_patch_records_cover_and_extension(self):
        overlap, corridor = self.load_sources()
        certificate = verifier.compute_certificate(overlap, corridor)
        self.assertEqual(certificate["symbolic_identities"]["ordinary_u_to_v"], "0")
        self.assertEqual(certificate["symbolic_identities"]["equatorial_zeta_extension"], "0")

        polar = certificate["polar_patch_cover"][0]
        self.assertEqual(polar["patch_type"], "polar_corridor_plus_ordinary_annulus")
        self.assertEqual(polar["target_sin_squared_theta_interval"], "[0, 1]")
        self.assertEqual(polar["ordinary_sin_squared_theta_interval"], "[1/64, 1]")
        self.assertIn("corridor_and_annulus_cover_full_v_interval", polar["checked_conditions"])
        self.assertIn("ordinary_annulus_Theta_positive", polar["checked_conditions"])

        equatorial = certificate["equatorial_patch_cover"][0]
        self.assertEqual(equatorial["patch_type"], "equatorial_extended_blowup_with_ordinary_eta_overlap")
        self.assertEqual(equatorial["extended_zeta_interval"], "[0, 1/128]")
        self.assertIn("extended_blowup_factor_positive", equatorial["checked_conditions"])
        self.assertIn("ordinary_patch_Theta_positive", equatorial["checked_conditions"])

    def test_rejects_stale_patch_coverage_package(self):
        stale = copy.deepcopy(self.load(PACKAGE))
        stale["polar_patch_cover"][0]["ordinary_sin_squared_theta_interval"] = "[1/32, 1]"
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

    def test_rejects_missing_overlap(self):
        overlap, corridor = self.load_sources()
        overlap["polar_overlap_strips"] = []
        with self.assertRaisesRegex(ValueError, "polar_positive_L_outer_overlap"):
            verifier.compute_certificate(overlap, corridor)


if __name__ == "__main__":
    unittest.main()
