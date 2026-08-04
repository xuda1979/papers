"""Tests for the TSBCB verifier.

Covers the accept path on the canonical certificate and the main
rejection paths:
  - missing top-level keys
  - wrong theorem / status
  - non-positive fiber length
  - fiber lengths not summing to r_far - r_red
  - weights not summing to 1
  - non-positive weight
  - non-positive transport constant
  - wrong q_min (not matching RMAR-QSD)
  - wrong beta_TSBCB (not matching the computed value)
  - missing resonant mode (not in RMAR-QSD lattice)
  - missing upstream input
  - missing audit-chain key
  - empty obligations_discharged
and the rational-parser unit tests.
"""

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

import verify_trapped_superradiant_bulk_coercivity_bridge as verifier


class VerifyTSBCBAcceptTests(unittest.TestCase):
    def setUp(self):
        self.cert_path = (
            ROOT
            / "certificates"
            / "tsbcb"
            / "trapped_superradiant_bulk_coercivity_bridge.json"
        )
        # Ensure the canonical certificate exists.
        if not self.cert_path.exists():
            verifier.main()

    def test_canonical_certificate_accepts(self):
        result = verifier.verify_certificate(self.cert_path, ROOT)
        self.assertTrue(
            result["accepted"],
            msg=f"canonical certificate should be accepted; errors: {result['errors']}",
        )
        self.assertEqual(result["errors"], [])

    def test_canonical_beta_is_strictly_positive(self):
        result = verifier.verify_certificate(self.cert_path, ROOT)
        self.assertTrue(result["accepted"])
        beta = Fraction(result["computed"]["beta_TSBCB"])
        self.assertGreater(beta, 0)
        self.assertEqual(beta, Fraction(21, 512))


class VerifyTSBCBRejectionTests(unittest.TestCase):
    def setUp(self):
        self.cert_path = (
            ROOT
            / "certificates"
            / "tsbcb"
            / "trapped_superradiant_bulk_coercivity_bridge.json"
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
        del cert["lower_bound"]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("lower_bound" in e for e in result["errors"]))

    def test_rejects_wrong_theorem(self):
        cert = self.cert.copy()
        cert["theorem"] = "NOT_TSBCB"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("theorem must be 'TSBCB'" in e for e in result["errors"]))

    def test_rejects_wrong_status(self):
        cert = self.cert.copy()
        cert["status"] = "candidate"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(any("status must be 'certified'" in e for e in result["errors"]))

    def test_rejects_nonpositive_fiber_length(self):
        cert = self.cert.copy()
        cert["cfrt_fibration"] = dict(cert["cfrt_fibration"])
        cert["cfrt_fibration"]["fiber_lengths"] = ["0", "1", "1", "1"]
        # Also fix midpoints so the rejection is from fiber_lengths.
        cert["cfrt_fibration"]["fiber_midpoints"] = ["3/2", "5/2", "7/2", "9/2"]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("fiber_lengths[0] must be > 0" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_fiber_lengths_not_summing(self):
        cert = self.cert.copy()
        cert["cfrt_fibration"] = dict(cert["cfrt_fibration"])
        # Sum is 2, not r_far - r_red = 4.
        cert["cfrt_fibration"]["fiber_lengths"] = ["1/2", "1/2", "1/2", "1/2"]
        cert["cfrt_fibration"]["fiber_midpoints"] = ["7/4", "9/4", "11/4", "13/4"]
        # Also corrupt weights so they remain consistent with fiber_lengths
        # (we want the failure to come from the lengths-sum check).
        cert["resonant_partition_of_unity"] = dict(cert["resonant_partition_of_unity"])
        cert["resonant_partition_of_unity"]["weights"] = ["1/8", "1/8", "1/8", "1/8"]
        # Corrupt T_mn similarly.
        cert["transport_constants"] = dict(cert["transport_constants"])
        cert["transport_constants"]["T_mn"] = ["4", "4", "4", "4"]
        # Corrupt lower_bound.
        cert["lower_bound"] = dict(cert["lower_bound"])
        cert["lower_bound"]["sum_wT"] = "2"
        cert["lower_bound"]["beta_TSBCB"] = "21/2048"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("sum(fiber_lengths)" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_weights_not_summing_to_one(self):
        cert = self.cert.copy()
        cert["resonant_partition_of_unity"] = dict(cert["resonant_partition_of_unity"])
        # Sum is 2, not 1.
        cert["resonant_partition_of_unity"]["weights"] = ["1/2", "1/2", "1/2", "1/2"]
        cert["resonant_partition_of_unity"]["sum_weights"] = "2"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("sum(weights)" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_nonpositive_weight(self):
        cert = self.cert.copy()
        cert["resonant_partition_of_unity"] = dict(cert["resonant_partition_of_unity"])
        cert["resonant_partition_of_unity"]["weights"] = ["0", "1/3", "1/3", "1/3"]
        cert["resonant_partition_of_unity"]["sum_weights"] = "1"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("weights[0] must be > 0" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_nonpositive_transport_constant(self):
        cert = self.cert.copy()
        cert["transport_constants"] = dict(cert["transport_constants"])
        cert["transport_constants"]["T_mn"] = ["0", "8", "8", "8"]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("T_mn[0] must be > 0" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_q_min(self):
        cert = self.cert.copy()
        cert["lower_bound"] = dict(cert["lower_bound"])
        # Wrong q_min (not 21/4096).
        cert["lower_bound"]["q_min"] = "1/100"
        cert["lower_bound"]["beta_TSBCB"] = str(Fraction(1, 100) * Fraction(8))
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("q_min" in e and "RMAR-QSD" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_beta(self):
        cert = self.cert.copy()
        cert["lower_bound"] = dict(cert["lower_bound"])
        # Wrong beta: should be 21/512 but we put 21/1000.
        cert["lower_bound"]["beta_TSBCB"] = "21/1000"
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("beta_TSBCB" in e and "must equal computed" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_missing_resonant_mode(self):
        cert = self.cert.copy()
        cert["cfrt_fibration"] = dict(cert["cfrt_fibration"])
        # Drop the (1, -1) mode and replace with a non-RMAR mode (2, 0).
        cert["cfrt_fibration"]["resonant_lattice"] = [[1, 0], [0, 1], [1, 1], [2, 0]]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("missing required RMAR-QSD mode" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_missing_upstream_input(self):
        cert = self.cert.copy()
        del cert["inputs"]["RMAR_QSD"]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("missing upstream input: RMAR_QSD" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_missing_audit_key(self):
        cert = self.cert.copy()
        del cert["audit_chain_position"]["downstream"]
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("audit_chain_position" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_empty_obligations_discharged(self):
        cert = self.cert.copy()
        cert["obligations_discharged"] = []
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("obligations_discharged" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_r_far_not_greater_than_r_red(self):
        cert = self.cert.copy()
        cert["cfrt_fibration"] = dict(cert["cfrt_fibration"])
        cert["cfrt_fibration"]["r_far"] = "3/2"  # equal to r_red
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("r_far" in e and "r_red" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )

    def test_rejects_wrong_transport_constant(self):
        cert = self.cert.copy()
        cert["transport_constants"] = dict(cert["transport_constants"])
        # Wrong T_mn[0]: should be 8 but we put 7.
        cert["transport_constants"]["T_mn"] = ["7", "8", "8", "8"]
        # Also corrupt lower_bound.
        cert["lower_bound"] = dict(cert["lower_bound"])
        sum_wT = Fraction(1, 4) * 7 + Fraction(1, 4) * 8 * 3
        cert["lower_bound"]["sum_wT"] = str(sum_wT)
        cert["lower_bound"]["beta_TSBCB"] = str(Fraction(21, 4096) * sum_wT)
        result = self._run_on_temp(cert)
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("T_mn[0]" in e and "must equal" in e for e in result["errors"]),
            msg=str(result["errors"]),
        )


class VerifyTSBCBRationalTests(unittest.TestCase):
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

    def test_parse_resonant_mode(self):
        self.assertEqual(verifier.parse_resonant_mode([1, -1], "t"), (1, -1))
        with self.assertRaises(ValueError):
            verifier.parse_resonant_mode([1, 2, 3], "t")
        with self.assertRaises(ValueError):
            verifier.parse_resonant_mode([1, "x"], "t")

    def test_construct_cfrt_fibration_equal_partition(self):
        fibration = verifier.construct_cfrt_fibration(
            r_red=Fraction(3, 2),
            r_far=Fraction(11, 2),
            resonant_lattice=[(1, 0), (0, 1), (1, 1), (1, -1)],
        )
        self.assertEqual(len(fibration.fiber_lengths), 4)
        # Each fiber length is (r_far - r_red) / 4 = 1.
        for L in fibration.fiber_lengths:
            self.assertEqual(L, Fraction(1))
        # Midpoints: 2, 3, 4, 5 (r_red + 1/2, 3/2, 5/2, 7/2).
        self.assertEqual(fibration.fiber_midpoints[0], Fraction(2))
        self.assertEqual(fibration.fiber_midpoints[3], Fraction(5))

    def test_compute_bulk_coercivity_lower_bound_canonical(self):
        fibration = verifier.construct_cfrt_fibration(
            r_red=Fraction(3, 2),
            r_far=Fraction(11, 2),
            resonant_lattice=[(1, 0), (0, 1), (1, 1), (1, -1)],
        )
        partition = verifier.construct_resonant_partition_of_unity(fibration)
        transport = verifier.construct_transport_constants(
            fibration, J_C=Fraction(8), mu_C=Fraction(1)
        )
        lb = verifier.compute_bulk_coercivity_lower_bound(
            q_min=Fraction(21, 4096),
            partition=partition,
            transport=transport,
        )
        self.assertEqual(lb.q_min, Fraction(21, 4096))
        self.assertEqual(lb.sum_wT, Fraction(8))
        self.assertEqual(lb.beta_TSBCB, Fraction(21, 512))
        self.assertTrue(lb.strictly_positive)


class VerifyTSBCBGenerateTests(unittest.TestCase):
    def test_generate_canonical_certificate_consistency(self):
        cert = verifier.generate_canonical_certificate()
        self.assertEqual(cert["theorem"], "TSBCB")
        self.assertEqual(cert["status"], "certified")
        # beta_TSBCB must be strictly positive.
        beta = Fraction(cert["lower_bound"]["beta_TSBCB"])
        self.assertGreater(beta, 0)
        self.assertEqual(beta, Fraction(21, 512))


if __name__ == "__main__":
    unittest.main()
