import copy
import json
import unittest
from fractions import Fraction
from pathlib import Path

import verify_good_cover_cech_cocycle as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyGoodCoverCechCocycleTests(unittest.TestCase):
    def load(self, relative_path: str) -> dict:
        with (ROOT / relative_path).open() as f:
            return json.load(f)

    def test_accepts_valid_certificate(self):
        cert_path = ROOT / "certificates/gc3t/spin_half_good_cover_cech_cocycle.json"
        result = verifier.verify_certificate(cert_path, ROOT)
        self.assertTrue(result["accepted"], msg=json.dumps(result, indent=2))
        self.assertEqual(result["theorem"], "GC3T")
        self.assertEqual(result["atlas"]["num_charts"], 16)
        self.assertEqual(result["atlas"]["chain_ok"], True)
        self.assertEqual(result["nerve"]["dimension"], 2)
        self.assertEqual(result["nerve"]["num_triple_overlaps"], 14)
        self.assertEqual(result["nerve"]["num_quadruple_overlaps_nonempty"], 0)
        self.assertTrue(result["nerve"]["triple_overlaps_nonempty"])
        self.assertTrue(result["nerve"]["quadruple_overlaps_empty"])
        self.assertEqual(result["factorization"]["num_triples_checked"], 14)
        self.assertTrue(result["factorization"]["all_factorizations_hold"])
        self.assertTrue(result["good_cover"]["declared"])
        self.assertEqual(result["good_cover"]["nerve_dimension"], 2)

    def test_rejects_missing_theorem(self):
        cert_path = ROOT / "certificates/gc3t/spin_half_good_cover_cech_cocycle.json"
        cert = json.loads(cert_path.read_text())
        cert.pop("theorem")
        tmp = ROOT / "certificates/gc3t/_tmp_bad.json"
        tmp.write_text(json.dumps(cert))
        try:
            result = verifier.verify_certificate(tmp, ROOT)
            self.assertFalse(result["accepted"])
            self.assertTrue(any("theorem" in e for e in result["errors"]))
        finally:
            tmp.unlink(missing_ok=True)

    def test_rejects_wrong_theorem(self):
        cert_path = ROOT / "certificates/gc3t/spin_half_good_cover_cech_cocycle.json"
        cert = json.loads(cert_path.read_text())
        cert["theorem"] = "NOT_GC3T"
        tmp = ROOT / "certificates/gc3t/_tmp_bad.json"
        tmp.write_text(json.dumps(cert))
        try:
            result = verifier.verify_certificate(tmp, ROOT)
            self.assertFalse(result["accepted"])
        finally:
            tmp.unlink(missing_ok=True)

    def test_rejects_nonpositive_thickening(self):
        cert_path = ROOT / "certificates/gc3t/spin_half_good_cover_cech_cocycle.json"
        cert = json.loads(cert_path.read_text())
        cert["thickenings"][5] = "0"
        tmp = ROOT / "certificates/gc3t/_tmp_bad.json"
        tmp.write_text(json.dumps(cert))
        try:
            result = verifier.verify_certificate(tmp, ROOT)
            self.assertFalse(result["accepted"])
        finally:
            tmp.unlink(missing_ok=True)

    def test_rejects_thickening_violating_quad_empty(self):
        """If we make epsilons too large, quadruple overlaps become nonempty."""
        cert_path = ROOT / "certificates/gc3t/spin_half_good_cover_cech_cocycle.json"
        cert = json.loads(cert_path.read_text())
        # Set all thickenings to a huge value so quad-empty fails
        cert["thickenings"] = ["1/2"] * 16
        tmp = ROOT / "certificates/gc3t/_tmp_bad.json"
        tmp.write_text(json.dumps(cert))
        try:
            result = verifier.verify_certificate(tmp, ROOT)
            self.assertFalse(result["accepted"])
            self.assertFalse(result["nerve"]["quadruple_overlaps_empty"])
        finally:
            tmp.unlink(missing_ok=True)

    def test_rejects_thickening_violating_triple_nonempty(self):
        """If we make epsilons too small, triple overlaps become empty."""
        cert_path = ROOT / "certificates/gc3t/spin_half_good_cover_cech_cocycle.json"
        cert = json.loads(cert_path.read_text())
        # Set all thickenings to a tiny value so triple-nonempty fails
        cert["thickenings"] = ["1/1000"] * 16
        tmp = ROOT / "certificates/gc3t/_tmp_bad.json"
        tmp.write_text(json.dumps(cert))
        try:
            result = verifier.verify_certificate(tmp, ROOT)
            self.assertFalse(result["accepted"])
            self.assertFalse(result["nerve"]["triple_overlaps_nonempty"])
        finally:
            tmp.unlink(missing_ok=True)

    def test_rational_parsing(self):
        self.assertEqual(verifier.parse_rational("3/4", "test"), Fraction(3, 4))
        self.assertEqual(verifier.parse_rational("5", "test"), Fraction(5, 1))
        self.assertEqual(verifier.parse_rational("-2/3", "test"), Fraction(-2, 3))
        with self.assertRaises(ValueError):
            verifier.parse_rational("1/0", "test")
        with self.assertRaises(ValueError):
            verifier.parse_rational("", "test")

    def test_rational_to_string_roundtrip(self):
        for s in ["3/4", "5", "1/24", "9/80"]:
            q = verifier.parse_rational(s, "test")
            self.assertEqual(verifier.rational_to_string(q), s)


if __name__ == "__main__":
    unittest.main()
