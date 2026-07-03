import copy
import json
import tempfile
import unittest
from pathlib import Path

import verify_carter_angular_overlap as verifier


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "certificates/carter_identities/angular_overlap_spin_half_certificate.json"
CORRIDOR = ROOT / "certificates/carter_identities/angular_corridor_spin_half_certificate.json"


class VerifyCarterAngularOverlapTests(unittest.TestCase):
    def load(self, path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_verifies_stored_overlap_package(self):
        result = verifier.verify_package(PACKAGE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["polar_overlap_count"], 2)
        self.assertEqual(result["equatorial_overlap_count"], 2)
        self.assertIn("polar_positive_L_outer_overlap", result["polar_overlap_strips"])
        self.assertIn("equatorial_retrograde_eta_overlap", result["equatorial_overlap_strips"])

    def test_computed_overlap_records_gluing_identities(self):
        certificate = verifier.compute_certificate(self.load(CORRIDOR))
        self.assertEqual(certificate["symbolic_identities"]["polar_overlap_identity"], "0")
        self.assertEqual(certificate["symbolic_identities"]["equatorial_overlap_identity"], "0")

        polar = certificate["polar_overlap_strips"][0]
        self.assertEqual(polar["overlap_type"], "polar_to_ordinary_angular")
        self.assertEqual(polar["sin_squared_theta_strip"], "[1/64, 1/16]")
        self.assertIn("sin_squared_theta_away_from_zero", polar["checked_conditions"])
        self.assertIn("ordinary_polar_Theta_positive", polar["checked_conditions"])

        equatorial = certificate["equatorial_overlap_strips"][0]
        self.assertEqual(equatorial["overlap_type"], "equatorial_to_ordinary_angular")
        self.assertEqual(equatorial["eta_strip"], "[1/8192, 1/4096]")
        self.assertIn("eta_away_from_zero_on_overlap", equatorial["checked_conditions"])
        self.assertIn("ordinary_equatorial_Theta_positive", equatorial["checked_conditions"])

    def test_rejects_stale_overlap_package(self):
        stale = copy.deepcopy(self.load(PACKAGE))
        stale["polar_overlap_strips"][0]["sin_squared_theta_strip"] = "[0, 1]"
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

    def test_rejects_missing_corridor(self):
        corridor = self.load(CORRIDOR)
        corridor["polar_corridors"] = []
        with self.assertRaisesRegex(ValueError, "missing"):
            verifier.compute_certificate(corridor)


if __name__ == "__main__":
    unittest.main()
