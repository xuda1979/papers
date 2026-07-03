import copy
import json
import unittest
from pathlib import Path

import verify_carter_orbit_cohomology_certificate as verifier


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = "certificates/carter_identities/spin_half_orbit_cohomology_shadow_certificate.json"


class VerifyCarterOrbitCohomologyCertificateTests(unittest.TestCase):
    def load(self):
        with (ROOT / CERTIFICATE).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_shadow_orbit_cohomology_certificate(self):
        result = verifier.verify_file(ROOT / CERTIFICATE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["claim_scope"], "carter_orbit_cohomology_certificate")
        self.assertEqual(result["derivation_status"], "candidate_orbit_cohomology")
        self.assertFalse(result["proves_kerr_gravitational_density"])
        self.assertEqual(
            result["checked_charts"],
            [
                "spin_half_shadow_torus_1_1_resonance",
                "spin_half_shadow_torus_2_1_nonresonant",
            ],
        )
        self.assertEqual(result["minimum_orbit_cohomology_margin"], "91/512")

    def test_rejects_wrong_coboundary_solution(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["torus_charts"][0]["coboundary_modes"][0]["escape_coefficient"] = ["0", "-1/16"]
        with self.assertRaisesRegex(ValueError, "cohomological equation"):
            verifier.verify(data)

    def test_rejects_resonant_mode_in_coboundaries(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["torus_charts"][0]["coboundary_modes"][0]["mode"] = [1, -1]
        with self.assertRaisesRegex(ValueError, "already listed as resonant"):
            verifier.verify(data)

    def test_rejects_nonresonant_mode_in_resonant_list(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["torus_charts"][1]["resonant_modes"].append(
            {"mode": [1, 0], "coefficient": ["1/100", "0"]}
        )
        with self.assertRaisesRegex(ValueError, "not resonant"):
            verifier.verify(data)

    def test_rejects_missing_conjugate_mode(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["torus_charts"][0]["resonant_modes"].pop()
        with self.assertRaisesRegex(ValueError, "missing conjugate"):
            verifier.verify(data)

    def test_rejects_margin_below_global_requirement(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["globalization"]["required_global_margin"] = "[1/4, 1/4]"
        with self.assertRaisesRegex(ValueError, "below required"):
            verifier.verify(data)

    def test_rejects_gravitational_proof_upgrade(self):
        data = self.load()
        data = copy.deepcopy(data)
        data["metadata"]["derivation_status"] = "kerr_gravitational_proved"
        with self.assertRaisesRegex(ValueError, "complete torus atlas"):
            verifier.verify(data)


if __name__ == "__main__":
    unittest.main()
