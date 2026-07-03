import copy
import json
import tempfile
import unittest
from pathlib import Path

import verify_carter_angular_corridor as verifier


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "certificates/carter_identities/angular_corridor_spin_half_certificate.json"
BRIDGE = ROOT / "certificates/carter_identities/degenerate_seed_bridge_spin_half_certificate.json"


class VerifyCarterAngularCorridorTests(unittest.TestCase):
    def load(self, path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_verifies_stored_angular_corridor_package(self):
        result = verifier.verify_package(PACKAGE)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["polar_corridor_count"], 2)
        self.assertEqual(result["equatorial_corridor_count"], 2)
        self.assertIn("polar_positive_L_corridor", result["polar_corridors"])
        self.assertIn("equatorial_retrograde_eta_corridor", result["equatorial_corridors"])

    def test_computed_corridor_records_regular_variables(self):
        certificate = verifier.compute_certificate(self.load(BRIDGE))
        self.assertEqual(certificate["symbolic_identities"]["polar_regularization"], "0")
        self.assertEqual(certificate["symbolic_identities"]["equatorial_blowup"], "0")

        polar = certificate["polar_corridors"][0]
        self.assertEqual(polar["coordinate_relation"], "lambda^2 = sin^2(theta)*mu^2")
        self.assertEqual(polar["lambda_interval"], "[0, 1/8192]")
        self.assertIn("regularized_Theta_positive", polar["checked_conditions"])

        equatorial = certificate["equatorial_corridors"][0]
        self.assertEqual(equatorial["coordinate_relation"], "cos^2(theta)=eta*zeta")
        self.assertEqual(equatorial["eta_thickness_interval"], "[0, 1/4096]")
        self.assertIn("equatorial_blowup_factor_positive", equatorial["checked_conditions"])

    def test_rejects_stale_stored_corridor(self):
        stale = copy.deepcopy(self.load(PACKAGE))
        stale["equatorial_corridors"][0]["factor_interval"] = "[0, 0]"
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

    def test_rejects_missing_seed_bridge(self):
        bridge = self.load(BRIDGE)
        bridge["bridges"] = [
            item for item in bridge["bridges"] if item["seed_family"] != "equatorial_Q_equals_zero_retrograde"
        ]
        with self.assertRaisesRegex(ValueError, "missing seed families"):
            verifier.compute_certificate(bridge)


if __name__ == "__main__":
    unittest.main()
