"""Tests for the EIACRT verifier.

Covers the accept path on the canonical certificate and the main
rejection paths:
  - missing top-level keys
  - wrong theorem / status / seed_theorem
  - wrong near-extremal parameter (eps_0, n_bins, bin_width)
  - wrong Aretakis anchor (wrong n_omega, wrong H_omega, non-decreasing)
  - wrong renormalization exponent
  - wrong SPBCPT inheritance (wrong beta_global_value)
  - malformed eps-bin partition (non-adjacent, wrong mid, wrong flag)
  - wrong per-bin compatibility (beta_tilde, lipschitz, residual)
  - wrong interface loss budget (total != n*per, over budget)
  - wrong H_compat (non-positive, inconsistent with bins)
  - missing obligations / audit inputs

Plus construction and rational-parser unit tests.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys_path_added = False
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
    sys_path_added = True

import verify_extremal_interface_aretakis_renormalization as verifier

if sys_path_added:
    pass  # leave path in place for submodule imports


class VerifyEIACRTAcceptTests(unittest.TestCase):
    def setUp(self):
        self.cert_path = (
            ROOT
            / "certificates"
            / "eiacrt"
            / "extremal_interface_aretakis_renormalization.json"
        )
        if not self.cert_path.exists():
            verifier.main()
        with open(self.cert_path) as f:
            self.cert = json.load(f)

    def test_canonical_certificate_accepts(self):
        result = verifier.verify_certificate(self.cert_path, ROOT)
        self.assertTrue(result["accepted"], msg=str(result["errors"]))

    def test_canonical_theorem_is_EIACRT(self):
        self.assertEqual(self.cert["theorem"], "EIACRT")

    def test_canonical_seed_is_SPBCPT(self):
        self.assertEqual(self.cert["seed_theorem"], "SPBCPT")

    def test_canonical_H_compat_positive(self):
        H = Fraction(self.cert["aretakis_compatibility_lower_bound"]["H_compat"])
        self.assertGreater(H, 0)

    def test_canonical_H_compat_value(self):
        # H_compat = min over bins of H_2 * eps_mid^4, with bin 0 = H_2.
        # H_2 = 21/128 * 1/2 * 2/6 = 21/128 * 1/6 = 21/768.
        # Wait, recompute: H_1 = H_0 * 1/(2*1) = 21/256.
        # H_2 = H_1 * 2/(2*3) = 21/256 * 1/3 = 21/768 = 7/256.
        # eps_mid values: 1/32, 3/32, 5/32, 7/32; bin 0 extremal -> H_2.
        # min over non-extremal bins: H_2 * (1/32)^4 = 7/256 * 1/1048576
        #   = 7 / 268435456.
        # But the extremal bin contributes H_2 = 7/256, which is larger.
        # So H_compat = 7/268435456.  However the verifier computed
        # 567/268435456 -- let me check.
        # Actually 7/256 = 7 * 1048576 / 268435456 = 7340032/268435456,
        # and 7/268435456 is much smaller.  The computed value is
        # 567/268435456 = 7 * 81 / 268435456.  Hmm, 81 = 3^4.  So
        # 567/268435456 = 7/256 * (9/32)^2 / ... no.
        # Let me recompute: H_2 = 7/256.  eps_mid for bin 1 = 3/32.
        # (3/32)^4 = 81 / 1048576.  H_2 * (3/32)^4 = 7/256 * 81/1048576
        #   = 567 / 268435456.  Yes!  The min over non-extremal bins is
        #   at the smallest non-extremal eps_mid = 3/32 (bin 1), giving
        #   567/268435456.  (Bin 0 is the extremal bin, contributing
        #   H_2 = 7/256, which is larger, so the min is at bin 1.)
        H = Fraction(self.cert["aretakis_compatibility_lower_bound"]["H_compat"])
        self.assertEqual(H, Fraction(567, 268435456))

    def test_canonical_has_4_bins(self):
        result = verifier.verify_certificate(self.cert_path, ROOT)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["computed"]["n_bins"], 4)

    def test_canonical_n_omega_is_2(self):
        self.assertEqual(self.cert["aretakis_anchor"]["n_omega"], 2)

    def test_canonical_H_0_matches_netcct(self):
        H_0 = Fraction(self.cert["aretakis_anchor"]["H_0_value"])
        self.assertEqual(H_0, Fraction(21, 128))

    def test_canonical_anchors_strictly_decreasing(self):
        anchors = self.cert["aretakis_anchor"]["anchors"]
        prev = None
        for a in anchors:
            H = Fraction(a["H_omega"])
            if prev is not None:
                self.assertLess(H, prev)
            prev = H

    def test_canonical_interface_loss_within_budget(self):
        ilb = self.cert["interface_loss_budget"]
        total = Fraction(ilb["total_interface_loss"])
        budget = Fraction(ilb["budget"])
        self.assertLessEqual(total, budget)
        self.assertTrue(ilb["within_budget"])


class VerifyEIACRTRejectionTests(unittest.TestCase):
    def setUp(self):
        self.cert_path = (
            ROOT
            / "certificates"
            / "eiacrt"
            / "extremal_interface_aretakis_renormalization.json"
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

    def test_rejects_wrong_theorem(self):
        cert = copy.deepcopy(self.cert)
        cert["theorem"] = "NOT_EIACRT"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("theorem must be 'EIACRT'" in e for e in result["errors"]))

    def test_rejects_wrong_status(self):
        cert = copy.deepcopy(self.cert)
        cert["status"] = "draft"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("status must be 'certified'" in e for e in result["errors"]))

    def test_rejects_wrong_seed_theorem(self):
        cert = copy.deepcopy(self.cert)
        cert["seed_theorem"] = "TSBCB"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("seed_theorem must be 'SPBCPT'" in e for e in result["errors"]))

    def test_rejects_wrong_eps_0(self):
        cert = copy.deepcopy(self.cert)
        cert["near_extremal_parameter"] = dict(cert["near_extremal_parameter"])
        cert["near_extremal_parameter"]["eps_0"] = "1/8"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("eps_0 must be" in e for e in result["errors"]))

    def test_rejects_wrong_bin_width(self):
        cert = copy.deepcopy(self.cert)
        cert["near_extremal_parameter"] = dict(cert["near_extremal_parameter"])
        cert["near_extremal_parameter"]["bin_width"] = "1/100"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("bin_width must be" in e for e in result["errors"]))

    def test_rejects_wrong_H_omega(self):
        cert = copy.deepcopy(self.cert)
        cert["aretakis_anchor"] = dict(cert["aretakis_anchor"])
        anchors = list(cert["aretakis_anchor"]["anchors"])
        anchors[1] = dict(anchors[1])
        anchors[1]["H_omega"] = "1/1000"  # wrong, doesn't match TS recursion
        cert["aretakis_anchor"]["anchors"] = anchors
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("must equal TS-recursion value" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_non_decreasing_anchors(self):
        cert = copy.deepcopy(self.cert)
        cert["aretakis_anchor"] = dict(cert["aretakis_anchor"])
        anchors = list(cert["aretakis_anchor"]["anchors"])
        # Make H_1 >= H_0 by setting H_1 = H_0.
        anchors[1] = dict(anchors[1])
        anchors[1]["H_omega"] = anchors[0]["H_omega"]
        # But this also breaks the TS recursion check.  We want to
        # isolate the monotonicity check, so also break the recursion
        # value to match (so only the monotonicity check fires).
        # Actually the monotonicity check fires first (prev = H_0,
        # H_1 = H_0 is not < H_0).  The TS recursion check also fires
        # because H_1 != expected.  Both errors are acceptable; we
        # just need at least one monotonicity-related error.
        cert["aretakis_anchor"]["anchors"] = anchors
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("must be strictly < previous" in e or "must equal TS-recursion" in e
                for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_p_renorm(self):
        cert = copy.deepcopy(self.cert)
        cert["renormalization"] = dict(cert["renormalization"])
        cert["renormalization"]["p_renorm"] = "1"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("p_renorm must be" in e for e in result["errors"]))

    def test_rejects_wrong_beta_global(self):
        cert = copy.deepcopy(self.cert)
        cert["spbcpt_inheritance"] = dict(cert["spbcpt_inheritance"])
        cert["spbcpt_inheritance"]["beta_global_value"] = "1/1000000"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("beta_global_value must equal SPBCPT" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_non_adjacent_bins(self):
        cert = copy.deepcopy(self.cert)
        bins = list(cert["eps_bins"])
        # Break adjacency between bin 0 and bin 1.
        bins[1] = dict(bins[1])
        bins[1]["eps_lo"] = "1/8"  # was 1/16
        bins[1]["eps_mid"] = "3/16"  # midpoint of [1/8, 1/4]
        # But this also breaks the mid check vs (lo+hi)/2, and the
        # beta_tilde check.  We just need at least one adjacency error.
        cert["eps_bins"] = bins
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("must equal previous eps_hi" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_beta_tilde(self):
        cert = copy.deepcopy(self.cert)
        bins = list(cert["eps_bins"])
        bins[1] = dict(bins[1])
        bins[1]["beta_tilde_at_mid"] = "1/1000000"
        cert["eps_bins"] = bins
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("beta_tilde_at_mid" in e and "must equal" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_residual(self):
        cert = copy.deepcopy(self.cert)
        bins = list(cert["eps_bins"])
        bins[1] = dict(bins[1])
        bins[1]["residual"] = "1/1000000"
        cert["eps_bins"] = bins
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("residual" in e and "must equal" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_interface_loss_over_budget(self):
        cert = copy.deepcopy(self.cert)
        cert["interface_loss_budget"] = dict(cert["interface_loss_budget"])
        # Shrink the budget so the total exceeds it.
        cert["interface_loss_budget"]["budget"] = "1/1000000"
        cert["interface_loss_budget"]["within_budget"] = False
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("exceeds budget" in e or "within_budget" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_non_positive_H_compat(self):
        cert = copy.deepcopy(self.cert)
        cert["aretakis_compatibility_lower_bound"] = dict(
            cert["aretakis_compatibility_lower_bound"]
        )
        cert["aretakis_compatibility_lower_bound"]["H_compat"] = "0"
        cert["aretakis_compatibility_lower_bound"]["strictly_positive"] = False
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("H_compat must be positive" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_H_compat(self):
        cert = copy.deepcopy(self.cert)
        cert["aretakis_compatibility_lower_bound"] = dict(
            cert["aretakis_compatibility_lower_bound"]
        )
        cert["aretakis_compatibility_lower_bound"]["H_compat"] = "1/1000000"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("H_compat" in e and "must equal" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_missing_obligation_extremal(self):
        cert = copy.deepcopy(self.cert)
        cert["obligations_discharged"] = [
            "renormalization of the SPBCPT bulk coercivity"
        ]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("extremal interface" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_missing_obligation_aretakis(self):
        cert = copy.deepcopy(self.cert)
        cert["obligations_discharged"] = [
            "extremal interface compatibility"
        ]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("Aretakis" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_missing_audit_input(self):
        cert = copy.deepcopy(self.cert)
        cert["audit_chain_position"] = dict(cert["audit_chain_position"])
        cert["audit_chain_position"]["inputs"] = [
            "NETCCT (near-extremal throat-collar compensation)",
            "TSBCB (trapped-superradiant bulk coercivity bridge)",
            "RRBB (redshift-RP buffer identity)",
        ]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("missing SPBCPT" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )


class VerifyEIACRTConstructionTests(unittest.TestCase):
    def test_aretakis_anchor_value_H0(self):
        H0 = verifier.aretakis_anchor_value(0)
        self.assertEqual(H0, Fraction(21, 128))

    def test_aretakis_anchor_value_H1(self):
        H1 = verifier.aretakis_anchor_value(1)
        # H_1 = H_0 * 1/(2*1) = 21/256.
        self.assertEqual(H1, Fraction(21, 256))

    def test_aretakis_anchor_value_H2(self):
        H2 = verifier.aretakis_anchor_value(2)
        # H_2 = H_1 * 2/(2*3) = 21/256 * 1/3 = 7/256.
        self.assertEqual(H2, Fraction(7, 256))

    def test_aretakis_anchor_decreasing(self):
        prev = None
        for omega in range(5):
            H = verifier.aretakis_anchor_value(omega)
            if prev is not None:
                self.assertLess(H, prev)
            prev = H

    def test_renormalized_coercivity_rational(self):
        beta = Fraction(21, 1024)
        eps = Fraction(3, 32)
        omega = 2
        result = verifier.renormalized_coercivity(beta, eps, omega)
        # eps^4 * beta = (3/32)^4 * 21/1024 = 81/1048576 * 21/1024
        # = 1701 / 1073741824
        self.assertEqual(result, Fraction(1701, 1073741824))

    def test_lipschitz_correction_omega0(self):
        eps_mid = Fraction(1, 32)
        hw = Fraction(1, 32)
        result = verifier.lipschitz_correction_bin(eps_mid, hw, 0)
        # omega = 0: NETCCT_LIPSCHITZ_MODULUS * hw = 21/16 * 1/32 = 21/512.
        self.assertEqual(result, Fraction(21, 512))

    def test_chi_from_eps(self):
        self.assertEqual(verifier.chi_from_eps(Fraction(0)), Fraction(1))
        self.assertEqual(verifier.chi_from_eps(Fraction(1, 2)), Fraction(3, 4))

    def test_kappa_from_eps(self):
        # kappa = eps / (2*(1+eps)).
        self.assertEqual(verifier.kappa_from_eps(Fraction(0)), Fraction(0))
        self.assertEqual(verifier.kappa_from_eps(Fraction(1, 4)), Fraction(1, 10))

    def test_build_eps_partition_coverage(self):
        partition = verifier.build_eps_partition(4, Fraction(1, 4))
        self.assertEqual(len(partition), 4)
        # First bin starts at 0.
        self.assertEqual(partition[0][0], Fraction(0))
        # Last bin ends at eps_0.
        self.assertEqual(partition[-1][1], Fraction(1, 4))
        # Adjacency.
        for i in range(len(partition) - 1):
            self.assertEqual(partition[i][1], partition[i + 1][0])
        # Extremal flag only on bin 0.
        self.assertTrue(partition[0][3])
        for i in range(1, len(partition)):
            self.assertFalse(partition[i][3])


class VerifyEIACRTParserTests(unittest.TestCase):
    def test_parse_rational_int(self):
        self.assertEqual(verifier.parse_rational(5, "t"), Fraction(5))

    def test_parse_rational_string_slash(self):
        self.assertEqual(verifier.parse_rational("21/128", "t"), Fraction(21, 128))

    def test_parse_rational_string_int(self):
        self.assertEqual(verifier.parse_rational("7", "t"), Fraction(7))

    def test_parse_rational_rejects_bad_string(self):
        with self.assertRaises(ValueError):
            verifier.parse_rational("not_a_rational", "t")

    def test_parse_rational_rejects_bad_type(self):
        with self.assertRaises(ValueError):
            verifier.parse_rational(1.5, "t")

    def test_parse_bool_true(self):
        self.assertTrue(verifier.parse_bool(True, "t"))

    def test_parse_bool_rejects_int(self):
        with self.assertRaises(ValueError):
            verifier.parse_bool(1, "t")

    def test_parse_bool_rejects_string(self):
        with self.assertRaises(ValueError):
            verifier.parse_bool("true", "t")


class VerifyEIACRTGenerateTests(unittest.TestCase):
    def test_generate_canonical_certificate_consistency(self):
        cert = verifier.build_canonical_certificate(n_bins=4, n_omega=2)
        self.assertEqual(cert["theorem"], "EIACRT")
        self.assertEqual(cert["status"], "certified")
        self.assertEqual(cert["seed_theorem"], "SPBCPT")
        H = Fraction(cert["aretakis_compatibility_lower_bound"]["H_compat"])
        self.assertGreater(H, 0)
        self.assertEqual(H, Fraction(567, 268435456))

    def test_generate_different_n_omega(self):
        cert = verifier.build_canonical_certificate(n_bins=4, n_omega=1)
        self.assertEqual(cert["aretakis_anchor"]["n_omega"], 1)
        H = Fraction(cert["aretakis_compatibility_lower_bound"]["H_compat"])
        self.assertGreater(H, 0)

    def test_generate_rejects_n_bins_too_small(self):
        with self.assertRaises(ValueError):
            verifier.build_canonical_certificate(n_bins=1, n_omega=2)


if __name__ == "__main__":
    unittest.main()
