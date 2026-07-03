import copy
import json
import tempfile
import unittest
from pathlib import Path

import verify_carter_angular_overlap_policy as verifier


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "certificates/carter_identities/angular_overlap_policy_spin_half_certificate.json"
GENERIC = ROOT / "certificates/carter_identities/angular_generic_overlap_spin_half_certificate.json"


class VerifyCarterAngularOverlapPolicyTests(unittest.TestCase):
    def load(self, path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_sources(self):
        generic = self.load(GENERIC)
        seed_path = GENERIC.parent / generic["metadata"]["angular_seed_tube_path"]
        return generic, self.load(seed_path)

    def test_verifies_stored_overlap_policy_package(self):
        result = verifier.verify_package(PACKAGE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["polar_policy_count"], 1)
        self.assertEqual(result["equatorial_policy_count"], 2)
        self.assertIn("polar_seed_dependent_overlap_policy", result["polar_overlap_policy"])
        self.assertIn(
            "equatorial_retrograde_dependent_overlap_policy",
            result["equatorial_overlap_policy"],
        )

    def test_computed_policy_records_dependent_transition_margins(self):
        generic, seed = self.load_sources()
        certificate = verifier.compute_certificate(generic, seed)
        self.assertEqual(certificate["symbolic_identities"]["polar_dependent_overlap"], "0")
        self.assertEqual(certificate["symbolic_identities"]["equatorial_dependent_overlap"], "0")

        polar = certificate["polar_overlap_policy"][0]
        self.assertEqual(polar["policy_type"], "polar_regular_axis_plus_ordinary_annulus")
        self.assertEqual(polar["target_sin_squared_theta_interval"], "[0, 1]")
        self.assertEqual(polar["dependent_lambda_interval"], "[-1/6000, 1/6000]")
        self.assertIn("polar_charts_cover_full_sin_squared_theta_interval", polar["checked_conditions"])
        self.assertIn("dependent_lambda_square_fits_coordinate_map", polar["checked_conditions"])

        equatorial = certificate["equatorial_overlap_policy"][0]
        self.assertEqual(equatorial["policy_type"], "equatorial_blowup_with_ordinary_eta_collar")
        self.assertEqual(equatorial["ordinary_eta_collar_interval"], "[1/8192, 1/1024]")
        self.assertEqual(equatorial["ordinary_cos_squared_theta_image_interval"], "[0, 1/49152]")
        self.assertIn("equatorial_transition_jacobian_positive", equatorial["checked_conditions"])
        self.assertIn("dependent_regularized_factor_positive", equatorial["checked_conditions"])

    def test_rejects_stale_overlap_policy_package(self):
        stale = copy.deepcopy(self.load(PACKAGE))
        stale["polar_overlap_policy"][0]["dependent_lambda_interval"] = "[-1/5000, 1/5000]"
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

    def test_rejects_missing_generic_doorway(self):
        generic, seed = self.load_sources()
        generic["polar_generic_overlaps"] = []
        with self.assertRaisesRegex(ValueError, "polar_seed_to_generic_ordinary_overlap"):
            verifier.compute_certificate(generic, seed)


if __name__ == "__main__":
    unittest.main()
