import copy
import json
import unittest
from pathlib import Path

import verify_carter_tame_orbit_cohomology as verifier


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = "certificates/carter_identities/spin_half_tame_orbit_cohomology_shadow_certificate.json"


class VerifyCarterTameOrbitCohomologyTests(unittest.TestCase):
    def load(self):
        with (ROOT / CERTIFICATE).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_shadow_tame_orbit_cohomology_certificate(self):
        result = verifier.verify_file(ROOT / CERTIFICATE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "carter_tame_orbit_cohomology_certificate")
        self.assertEqual(result["derivation_status"], "candidate_tame_orbit_cohomology")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(result["minimum_nonresonant_denominator"], "1")
        self.assertEqual(result["maximum_escape_derivative_l1"], "101/180")
        self.assertEqual(result["minimum_tame_margin"], "3893/23040")

    def test_rejects_denominator_floor_too_large(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["tame_budget"]["nonresonant_denominator_floor"] = "2"
        with self.assertRaisesRegex(ValueError, "below floor"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_identities")

    def test_rejects_escape_derivative_bound_too_small(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["tame_budget"]["escape_derivative_l1_bound"] = "[0, 1/4]"
        with self.assertRaisesRegex(ValueError, "exceeds tame bound"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_identities")

    def test_rejects_required_tame_margin_too_large(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["tame_budget"]["required_tame_margin"] = "[1/5, 1/5]"
        with self.assertRaisesRegex(ValueError, "below required"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_identities")

    def test_rejects_gravitational_proof_upgrade(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "symbol seminorm"):
            verifier.verify(data, base_dir=ROOT / "certificates/carter_identities")


if __name__ == "__main__":
    unittest.main()
