"""Tests for the SPGCPT verifier.

Covers the accept path on the canonical certificate and the main
rejection paths:
  - missing top-level keys
  - wrong theorem / seed_theorem
  - non-positive seed margin
  - mu outside (0, 1)
  - bin half-width larger than min propagation radius (C4 failure)
  - mismatched propagation radius (C3 failure)
  - non-positive Lipschitz modulus (C2 failure)
  - coverage gap (C5 failure)
  - seed-bin margins not matching the GC3T seed margins
"""

import json
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import verify_spin_parametric_good_cover_propagation as verifier


ROOT = Path(__file__).resolve().parents[1]
CERT_PATH = ROOT / "certificates/spgcpt/spin_parametric_good_cover_propagation.json"


class VerifySPGCPTAcceptTests(unittest.TestCase):
    def test_canonical_certificate_is_accepted(self):
        result = verifier.verify_certificate(CERT_PATH, ROOT)
        self.assertTrue(result["accepted"], msg=json.dumps(result, indent=2))
        self.assertEqual(result["theorem"], "SPGCPT")
        self.assertEqual(result["seed_theorem"], "GC3T")
        self.assertEqual(result["seed_atlas_num_charts"], 16)
        self.assertEqual(result["seed_margins"]["num_triple"], 14)
        self.assertEqual(result["seed_margins"]["num_quad"], 13)
        self.assertEqual(result["seed_margins"]["min_seed_margin"], "1/240")
        self.assertEqual(result["partition"]["num_bins"], 123)
        self.assertTrue(result["coverage"]["coverage_holds"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["errors"], [])

    def test_canonical_certificate_seed_bin_is_at_half(self):
        """Bin 61 must be the seed bin, with seed_spin = 1/2."""
        result = verifier.verify_certificate(CERT_PATH, ROOT)
        self.assertTrue(result["accepted"])
        seed_bin = result["partition"]["bin_results"][61]
        self.assertEqual(seed_bin["seed_spin"], "1/2")
        self.assertTrue(seed_bin["bin_containment_holds"])

    def test_canonical_certificate_all_bins_satisfy_containment(self):
        result = verifier.verify_certificate(CERT_PATH, ROOT)
        self.assertTrue(result["accepted"])
        for b in result["partition"]["bin_results"]:
            self.assertTrue(b["bin_containment_holds"],
                            f"bin {b['idx']} failed containment")


class VerifySPGCPTRejectionTests(unittest.TestCase):
    def _load_cert(self):
        return json.loads(CERT_PATH.read_text())

    def _run_on_temp(self, cert):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp_cert.json"
            tmp.write_text(json.dumps(cert))
            return verifier.verify_certificate(tmp, ROOT)

    def test_rejects_missing_top_level_key(self):
        cert = self._load_cert()
        del cert["strict_slack_mu"]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("strict_slack_mu" in e for e in result["errors"]))

    def test_rejects_wrong_theorem(self):
        cert = self._load_cert()
        cert["theorem"] = "NOT_SPGCPT"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("theorem must be 'SPGCPT'" in e for e in result["errors"]))

    def test_rejects_wrong_seed_theorem(self):
        cert = self._load_cert()
        cert["seed_theorem"] = "NOT_GC3T"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("seed_theorem must be 'GC3T'" in e for e in result["errors"]))

    def test_rejects_mu_outside_unit_interval(self):
        cert = self._load_cert()
        cert["strict_slack_mu"] = "2"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("strict_slack_mu" in e for e in result["errors"]))

    def test_rejects_mu_zero(self):
        cert = self._load_cert()
        cert["strict_slack_mu"] = "0"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("strict_slack_mu" in e for e in result["errors"]))

    def test_rejects_half_width_exceeding_propagation_radius(self):
        """Make one bin too wide -> C4 fails."""
        cert = self._load_cert()
        # Widen bin 0 by raising hi; keep lo fixed.  The half-width is
        # recomputed by the verifier as (hi - lo)/2, so this should
        # violate containment.
        cert["partition"][0]["hi"] = "1/4"  # half-width becomes 1/8 - 1/488 ~ way too big
        cert["partition"][0]["half_width"] = "1/8"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("bin_containment_holds" in e or "half_width" in e
                for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_mismatched_propagation_radius(self):
        """Corrupt one propagation radius -> C3 fails."""
        cert = self._load_cert()
        cert["partition"][5]["propagation_radii_strict"][0] = "1/1000000"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("propagation_radii_strict" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_nonpositive_lipschitz_modulus(self):
        """A zero Lipschitz modulus is invalid."""
        cert = self._load_cert()
        cert["partition"][10]["lipschitz_moduli"][0] = "0"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("lipschitz_moduli" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_coverage_gap(self):
        """Introduce a gap by removing a middle bin."""
        cert = self._load_cert()
        # Shift all bins from index 60 onward to the right by one bin
        # width, creating a gap.
        shift = Fraction(1, 122)
        for b in cert["partition"][60:]:
            lo = Fraction(b["lo"]) + shift
            hi = Fraction(b["hi"]) + shift
            seed = Fraction(b["seed_spin"]) + shift
            b["lo"] = str(lo)
            b["hi"] = str(hi)
            b["seed_spin"] = str(seed)
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("gap" in e.lower() or "cover" in e.lower() for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_seed_bin_margin_mismatch(self):
        """Corrupt a seed-bin margin so it no longer matches the GC3T seed."""
        cert = self._load_cert()
        # bin 61 is the seed bin; corrupt its first margin
        cert["partition"][61]["margins"][0] = "1/1000"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("seed margin" in e or "does not match" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )


class VerifySPGCPTRationalTests(unittest.TestCase):
    def test_parse_rational_accepts_int_and_string(self):
        self.assertEqual(verifier.parse_rational(3, "t"), Fraction(3))
        self.assertEqual(verifier.parse_rational("3/4", "t"), Fraction(3, 4))
        self.assertEqual(verifier.parse_rational("  3/4 ", "t"), Fraction(3, 4))

    def test_parse_rational_rejects_bool(self):
        with self.assertRaises(ValueError):
            verifier.parse_rational(True, "t")

    def test_parse_rational_rejects_zero_denominator(self):
        with self.assertRaises(ValueError):
            verifier.parse_rational("1/0", "t")

    def test_rational_to_string(self):
        self.assertEqual(verifier.rational_to_string(Fraction(3, 4)), "3/4")
        self.assertEqual(verifier.rational_to_string(Fraction(5, 1)), "5/1")


if __name__ == "__main__":
    unittest.main()
