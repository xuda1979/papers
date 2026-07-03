import copy
import json
import tempfile
import unittest
from pathlib import Path

import verify_carter_angular_generic_overlap as verifier


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "certificates/carter_identities/angular_generic_overlap_spin_half_certificate.json"
SEED = ROOT / "certificates/carter_identities/angular_seed_tube_spin_half_certificate.json"


class VerifyCarterAngularGenericOverlapTests(unittest.TestCase):
    def load(self, path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_verifies_stored_generic_overlap_package(self):
        result = verifier.verify_package(PACKAGE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["polar_generic_overlap_count"], 1)
        self.assertEqual(result["equatorial_generic_overlap_count"], 2)
        self.assertIn("polar_seed_to_generic_ordinary_overlap", result["polar_generic_overlaps"])
        self.assertIn(
            "equatorial_retrograde_seed_to_generic_ordinary_overlap",
            result["equatorial_generic_overlaps"],
        )

    def test_computed_generic_overlap_records_transition_margins(self):
        certificate = verifier.compute_certificate(self.load(SEED))
        self.assertEqual(certificate["symbolic_identities"]["polar_generic_inverse"], "0")
        self.assertEqual(certificate["symbolic_identities"]["equatorial_generic_inverse"], "0")

        polar = certificate["polar_generic_overlaps"][0]
        self.assertEqual(polar["overlap_type"], "polar_regular_to_generic_ordinary")
        self.assertEqual(polar["generic_lambda_overlap_interval"], "[-1/12000, 1/12000]")
        self.assertEqual(polar["generic_sin_squared_theta_overlap_interval"], "[1/64, 1/16]")
        self.assertIn("polar_transition_denominator_positive", polar["checked_conditions"])
        self.assertIn("inverse_mu_squared_inside_seed_tube", polar["checked_conditions"])

        equatorial = certificate["equatorial_generic_overlaps"][0]
        self.assertEqual(equatorial["overlap_type"], "equatorial_blowup_to_generic_ordinary")
        self.assertEqual(equatorial["generic_eta_overlap_interval"], "[1/8192, 1/1024]")
        self.assertEqual(equatorial["generic_cos_squared_theta_overlap_interval"], "[0, 1/393216]")
        self.assertIn("equatorial_transition_jacobian_positive", equatorial["checked_conditions"])
        self.assertIn("inverse_zeta_inside_seed_tube", equatorial["checked_conditions"])

    def test_rejects_stale_generic_overlap_package(self):
        stale = copy.deepcopy(self.load(PACKAGE))
        stale["polar_generic_overlaps"][0]["generic_lambda_overlap_interval"] = "[-1/6000, 1/6000]"
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

    def test_rejects_missing_seed_tube(self):
        seed = self.load(SEED)
        seed["equatorial_seed_tubes"] = []
        with self.assertRaisesRegex(ValueError, "equatorial_prograde_eta_zeta_seed_tube"):
            verifier.compute_certificate(seed)


if __name__ == "__main__":
    unittest.main()
