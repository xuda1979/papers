"""Tests for the SPBCPT verifier.

Covers the accept path on the canonical certificate and the main
rejection paths:
  - missing top-level keys
  - wrong theorem / status / seed_theorem
  - wrong spin_range
  - wrong strict_slack_mu
  - wrong lipschitz_modulus_L_beta
  - wrong netcct_uniform_margin
  - non-positive beta_bin
  - wrong J_C_at_mid
  - wrong beta_TSBCB_at_mid
  - wrong lipschitz_correction
  - wrong netcct_compensation
  - wrong beta_bin (non-seed bin)
  - gap in partition
  - wrong global_lower_bound
  - missing audit-chain input
  - missing check flag
  - empty obligations_discharged
and rational-parser/construction unit tests.
"""

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

import verify_spin_parametric_bulk_coercivity_propagation as verifier


class VerifySPBCPTAcceptTests(unittest.TestCase):
    def setUp(self):
        self.cert_path = (
            ROOT
            / "certificates"
            / "spbcpt"
            / "spin_parametric_bulk_coercivity_propagation.json"
        )
        if not self.cert_path.exists():
            verifier.main()

    def test_canonical_certificate_accepts(self):
        result = verifier.verify_certificate(self.cert_path, ROOT)
        self.assertTrue(
            result["accepted"],
            msg=f"canonical certificate should be accepted; errors: {result['errors']}",
        )
        self.assertEqual(result["errors"], [])

    def test_canonical_beta_global_is_strictly_positive(self):
        result = verifier.verify_certificate(self.cert_path, ROOT)
        self.assertTrue(result["accepted"])
        beta = Fraction(result["computed"]["beta_global"])
        self.assertGreater(beta, 0)
        self.assertEqual(beta, Fraction(21, 1024))

    def test_canonical_has_8_bins(self):
        result = verifier.verify_certificate(self.cert_path, ROOT)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["computed"]["n_bins"], 8)


class VerifySPBCPTRejectionTests(unittest.TestCase):
    def setUp(self):
        self.cert_path = (
            ROOT
            / "certificates"
            / "spbcpt"
            / "spin_parametric_bulk_coercivity_propagation.json"
        )
        if not self.cert_path.exists():
            verifier.main()
        with open(self.cert_path) as f:
            self.cert = json.load(f)

    def _run_on_temp(self, cert):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp_cert.json"
            tmp.write_text(json.dumps(cert))
            return verifier.verify_certificate(tmp, ROOT)

    def test_rejects_missing_top_level_key(self):
        cert = self.cert.copy()
        del cert["global_lower_bound"]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("global_lower_bound" in e for e in result["errors"]))

    def test_rejects_wrong_theorem(self):
        cert = self.cert.copy()
        cert["theorem"] = "NOT_SPBCPT"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("theorem must be 'SPBCPT'" in e for e in result["errors"]))

    def test_rejects_wrong_status(self):
        cert = self.cert.copy()
        cert["status"] = "candidate"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("status must be 'certified'" in e for e in result["errors"]))

    def test_rejects_wrong_seed_theorem(self):
        cert = self.cert.copy()
        cert["seed_theorem"] = "NOT_TSBCB"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("seed_theorem must be 'TSBCB'" in e for e in result["errors"]))

    def test_rejects_wrong_spin_range_lo(self):
        cert = self.cert.copy()
        cert["spin_range"] = dict(cert["spin_range"])
        cert["spin_range"]["lo"] = "1/4"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("spin_range.lo" in e for e in result["errors"]))

    def test_rejects_wrong_strict_slack_mu(self):
        cert = self.cert.copy()
        cert["strict_slack_mu"] = "1/200"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("strict_slack_mu" in e for e in result["errors"]))

    def test_rejects_wrong_lipschitz_modulus(self):
        cert = self.cert.copy()
        cert["lipschitz_modulus_L_beta"] = "1/100"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("lipschitz_modulus_L_beta" in e for e in result["errors"]))

    def test_rejects_wrong_netcct_margin(self):
        cert = self.cert.copy()
        cert["netcct_uniform_margin"] = "1/100"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("netcct_uniform_margin" in e for e in result["errors"]))

    def test_rejects_wrong_J_C_at_mid(self):
        cert = self.cert.copy()
        # Corrupt J_C at bin 0.
        cert["partition"] = list(cert["partition"])
        bin0 = dict(cert["partition"][0])
        bin0["J_C_at_mid"] = "99"
        cert["partition"][0] = bin0
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("J_C_at_mid" in e and "must equal" in e for e in result["errors"]))

    def test_rejects_wrong_beta_TSBCB_at_mid(self):
        cert = self.cert.copy()
        cert["partition"] = list(cert["partition"])
        bin0 = dict(cert["partition"][0])
        bin0["beta_TSBCB_at_mid"] = "1/1000"
        cert["partition"][0] = bin0
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("beta_TSBCB_at_mid" in e and "must equal" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_lipschitz_correction(self):
        cert = self.cert.copy()
        cert["partition"] = list(cert["partition"])
        bin0 = dict(cert["partition"][0])
        bin0["lipschitz_correction"] = "0"
        # Also corrupt beta_bin so the failure comes from lipschitz_correction.
        bin0["beta_bin"] = str(
            Fraction(bin0["beta_TSBCB_at_mid"]) - Fraction(0)
            + (Fraction(bin0["netcct_compensation"]) if bin0["netcct_compensation"] != "0" else Fraction(0))
        )
        cert["partition"][0] = bin0
        # Also corrupt global_lower_bound.
        cert["global_lower_bound"] = dict(cert["global_lower_bound"])
        bins = cert["partition"]
        betas = [Fraction(b["beta_bin"]) for b in bins]
        cert["global_lower_bound"]["beta_global"] = str(min(betas))
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("lipschitz_correction" in e and "must equal" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_global_lower_bound(self):
        cert = self.cert.copy()
        cert["global_lower_bound"] = dict(cert["global_lower_bound"])
        cert["global_lower_bound"]["beta_global"] = "1/1000000"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("beta_global" in e and "must equal" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_missing_audit_input(self):
        cert = self.cert.copy()
        cert["audit_chain_position"] = dict(cert["audit_chain_position"])
        cert["audit_chain_position"]["inputs"] = ["TSBCB", "SPGCPT"]  # missing NETCCT, RRBB, etc.
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("missing NETCCT" in e for e in result["errors"]))

    def test_rejects_missing_check_flag(self):
        cert = self.cert.copy()
        cert["checks"] = dict(cert["checks"])
        del cert["checks"]["P4_global_minimum"]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("P4_global_minimum" in e for e in result["errors"]))

    def test_rejects_empty_obligations_discharged(self):
        cert = self.cert.copy()
        cert["obligations_discharged"] = []
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("obligations_discharged" in e for e in result["errors"]))

    def test_rejects_nonpositive_beta_bin(self):
        cert = self.cert.copy()
        cert["partition"] = list(cert["partition"])
        bin0 = dict(cert["partition"][0])
        bin0["beta_bin"] = "0"
        cert["partition"][0] = bin0
        # Also corrupt global_lower_bound so the failure comes from beta_bin.
        cert["global_lower_bound"] = dict(cert["global_lower_bound"])
        cert["global_lower_bound"]["beta_global"] = "0"
        cert["global_lower_bound"]["strictly_positive"] = False
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("beta_bin" in e and "must be > 0" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_seed_bin_beta(self):
        cert = self.cert.copy()
        cert["partition"] = list(cert["partition"])
        # Find the seed bin and corrupt its beta_bin.
        for i, b in enumerate(cert["partition"]):
            if b["is_seed_bin"]:
                bin_seed = dict(b)
                bin_seed["beta_bin"] = "1/1000"
                cert["partition"][i] = bin_seed
                break
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("beta_bin" in e and "TSBCB seed" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_gap_in_partition(self):
        cert = self.cert.copy()
        cert["partition"] = list(cert["partition"])
        # Create a gap by shifting bin 4 up so that its lo > bin 3's hi.
        # Original bin 4: lo=1/2, hi=5/8, mid=9/16.  Shift to lo=9/16, hi=11/16, mid=5/8.
        bin4 = dict(cert["partition"][4])
        bin4["lo"] = "9/16"
        bin4["hi"] = "11/16"
        bin4["mid"] = "5/8"
        # Recompute all derived quantities for bin 4 so the only failure
        # is the gap check (between bin 3's hi=1/2 and bin 4's lo=9/16).
        mid = Fraction(bin4["mid"])
        half_width = (Fraction(bin4["hi"]) - Fraction(bin4["lo"])) / 2
        J_C = verifier.compute_J_C_at_mid(mid)
        beta_mid = verifier.compute_beta_TSBCB_at_mid(mid)
        lip_corr = verifier.LIPSCHITZ_MODULUS_L_BETA * half_width
        is_ne = mid >= verifier.NEAR_EXTREMAL_CHI_LO
        netcct = verifier.NETCCT_M_UNIFORM * half_width if is_ne else Fraction(0)
        bin4["J_C_at_mid"] = str(J_C)
        bin4["beta_TSBCB_at_mid"] = str(beta_mid)
        bin4["lipschitz_correction"] = str(lip_corr)
        bin4["netcct_compensation"] = str(netcct)
        bin4["is_near_extremal_bin"] = is_ne
        # bin 4 is no longer the seed bin (1/2 not in [9/16, 11/16]).
        bin4["is_seed_bin"] = False
        bin4["beta_bin"] = str(beta_mid - lip_corr + netcct)
        cert["partition"][4] = bin4
        # Recompute global lower bound.
        betas = [Fraction(b["beta_bin"]) for b in cert["partition"]]
        cert["global_lower_bound"] = dict(cert["global_lower_bound"])
        cert["global_lower_bound"]["beta_global"] = str(min(betas))
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("gap" in e.lower() for e in result["errors"]),
            msg=str(result["errors"]),
        )


class VerifySPBCPTRationalTests(unittest.TestCase):
    def test_parse_rational_accepts_int_and_string(self):
        self.assertEqual(verifier.parse_rational(3, "t"), Fraction(3))
        self.assertEqual(verifier.parse_rational("3/4", "t"), Fraction(3, 4))
        self.assertEqual(verifier.parse_rational(Fraction(5, 7), "t"), Fraction(5, 7))

    def test_parse_rational_rejects_bad_string(self):
        with self.assertRaises(ValueError):
            verifier.parse_rational("not_a_rational", "t")

    def test_parse_rational_rejects_bad_type(self):
        with self.assertRaises(ValueError):
            verifier.parse_rational(1.5, "t")

    def test_parse_bool(self):
        self.assertTrue(verifier.parse_bool(True, "t"))
        with self.assertRaises(ValueError):
            verifier.parse_bool(1, "t")  # int, not bool
        with self.assertRaises(ValueError):
            verifier.parse_bool("true", "t")

    def test_compute_J_C_at_mid_seed(self):
        self.assertEqual(
            verifier.compute_J_C_at_mid(Fraction(1, 2)), Fraction(8)
        )

    def test_compute_J_C_at_mid_nonseed(self):
        # J_C(chi) = 8 + (chi - 1/2) * 8.
        self.assertEqual(
            verifier.compute_J_C_at_mid(Fraction(1, 16)),
            Fraction(8) + (Fraction(1, 16) - Fraction(1, 2)) * Fraction(8),
        )

    def test_compute_beta_TSBCB_at_mid_seed(self):
        self.assertEqual(
            verifier.compute_beta_TSBCB_at_mid(Fraction(1, 2)),
            Fraction(21, 512),
        )

    def test_construct_spin_partition_8_bins(self):
        bins = verifier.construct_spin_partition(n_bins=8)
        self.assertEqual(len(bins), 8)
        # Seed bins contain chi = 1/2.
        seed_bins = [b for b in bins if b.is_seed_bin]
        self.assertGreaterEqual(len(seed_bins), 1)
        for b in seed_bins:
            self.assertLessEqual(b.lo, Fraction(1, 2))
            self.assertGreaterEqual(b.hi, Fraction(1, 2))
            self.assertEqual(b.beta_bin, Fraction(21, 512))

    def test_compute_global_lower_bound_canonical(self):
        bins = verifier.construct_spin_partition(n_bins=8)
        glb = verifier.compute_global_lower_bound(bins)
        self.assertGreater(glb, 0)
        self.assertEqual(glb, Fraction(21, 1024))


class VerifySPBCPTGenerateTests(unittest.TestCase):
    def test_generate_canonical_certificate_consistency(self):
        cert = verifier.generate_canonical_certificate(n_bins=8)
        self.assertEqual(cert["theorem"], "SPBCPT")
        self.assertEqual(cert["status"], "certified")
        self.assertEqual(cert["seed_theorem"], "TSBCB")
        beta = Fraction(cert["global_lower_bound"]["beta_global"])
        self.assertGreater(beta, 0)
        self.assertEqual(beta, Fraction(21, 1024))


if __name__ == "__main__":
    unittest.main()
