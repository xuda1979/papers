#!/usr/bin/env python3
"""Tests for verify_redshift_rp_buffer.py and verify_composed_cwqp_rrbb.py."""

from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import verify_redshift_rp_buffer as rrbb  # noqa: E402
import verify_composed_cwqp_rrbb as composed  # noqa: E402

RRBB_DIR = ROOT / "certificates" / "rrbb"
CWQP_DIR = ROOT / "certificates" / "cwqp_assembly"


class TestComputeBRedshift(unittest.TestCase):
    def test_product(self):
        self.assertEqual(rrbb.compute_B_redshift(Fraction(3), Fraction(4)), 12)
        self.assertEqual(
            rrbb.compute_B_redshift(Fraction(1, 4), Fraction(1, 8)),
            Fraction(1, 32),
        )


class TestComputeBRp(unittest.TestCase):
    def test_integer_eta(self):
        # C_p * r_far^{-eta_p} = (1/2) * 4^{-1} = 1/8
        self.assertEqual(
            rrbb.compute_B_rp(Fraction(1, 2), Fraction(4), Fraction(1)),
            Fraction(1, 8),
        )
        # (3) * 2^{-3} = 3/8
        self.assertEqual(
            rrbb.compute_B_rp(Fraction(3), Fraction(2), Fraction(3)),
            Fraction(3, 8),
        )

    def test_noninteger_eta_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            rrbb.compute_B_rp(Fraction(1, 2), Fraction(4), Fraction(1, 2))
        self.assertIn("eta_p must be an integer", str(ctx.exception))


class TestToyRRBBCertificate(unittest.TestCase):
    def test_toy_accepted(self):
        result = rrbb.verify_file(RRBB_DIR / "toy_rrbb_certificate.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(Fraction(result["buffer"]["B_redshift"]), Fraction(1, 32))
        self.assertEqual(Fraction(result["buffer"]["B_rp"]), Fraction(1, 8))
        self.assertEqual(Fraction(result["buffer"]["B_comm_lower_bound"]), 0)
        self.assertEqual(Fraction(result["buffer"]["B_lower_bound"]), Fraction(5, 32))

    def test_toy_with_cwqp_compat_accepted(self):
        result = rrbb.verify_file(RRBB_DIR / "toy_rrbb_with_cwqp_compat.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(Fraction(result["buffer"]["B_lower_bound"]), 6)
        self.assertTrue(result["cwqp_compatibility"]["closure"])
        self.assertEqual(Fraction(result["cwqp_compatibility"]["total_loss"]), 6)


class TestBadRRBBCertificates(unittest.TestCase):
    def test_buffer_mismatch_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify_file(RRBB_DIR / "bad_rrbb_buffer_mismatch.json")
        self.assertIn("B_redshift mismatch", str(ctx.exception))

    def test_region_order_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify_file(RRBB_DIR / "bad_rrbb_region_overlap.json")
        self.assertIn("redshift_layer.hi <= rp_far_field.lo", str(ctx.exception))

    def test_cwqp_composition_fail_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify_file(RRBB_DIR / "bad_rrbb_cwqp_composition_fail.json")
        self.assertIn("closure mismatch", str(ctx.exception))


class TestHandcraftedRRBB(unittest.TestCase):
    def _base_cert(self) -> dict:
        return {
            "metadata": {
                "name": "Redshift–r^p RRBB test",
                "generated_on": "2026-07-03",
                "spin": "toy",
                "claim_scope": "toy",
                "status": "toy",
                "warning": "toy",
            },
            "regions": {
                "redshift_layer": {"endpoints": ["1/1", "5/4"]},
                "trapped_bulk": {"endpoints": ["5/4", "4/1"]},
                "rp_far_field": {"endpoints": ["4/1", "1000000/1"]},
            },
            "redshift_certificate": {
                "kappa_red": "1/4",
                "mu_redshift": "1/8",
                "horizon_surface_gravity_lower_bound": "1/100",
            },
            "rp_certificate": {
                "C_p": "1/2",
                "eta_p": "1/1",
                "p": "5/2",
                "sobolev_index": 6,
            },
            "commutator_slack_certificate": {
                "nonnegative": True,
                "escape_symbol_reference": "esc.pdf",
                "B_comm_lower_bound": "0/1",
            },
            "buffer": {
                "B_redshift": "1/32",
                "B_rp": "1/8",
                "B_comm_lower_bound": "0/1",
                "B_lower_bound": "5/32",
            },
        }

    def test_valid(self):
        result = rrbb.verify(self._base_cert())
        self.assertTrue(result["accepted"])

    def test_negative_kappa_rejected(self):
        cert = self._base_cert()
        cert["redshift_certificate"]["kappa_red"] = "-1/4"
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify(cert)
        self.assertIn("kappa_red", str(ctx.exception))

    def test_zero_mu_rejected(self):
        cert = self._base_cert()
        cert["redshift_certificate"]["mu_redshift"] = "0/1"
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify(cert)
        self.assertIn("mu_redshift", str(ctx.exception))

    def test_p_le_2_rejected(self):
        cert = self._base_cert()
        cert["rp_certificate"]["p"] = "2/1"
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify(cert)
        self.assertIn("p > 2", str(ctx.exception))

    def test_negative_b_comm_rejected(self):
        cert = self._base_cert()
        cert["commutator_slack_certificate"]["B_comm_lower_bound"] = "-1/8"
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify(cert)
        self.assertIn("B_comm_lower_bound", str(ctx.exception))

    def test_false_nonnegative_rejected(self):
        cert = self._base_cert()
        cert["commutator_slack_certificate"]["nonnegative"] = False
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify(cert)
        self.assertIn("nonnegative", str(ctx.exception))

    def test_b_lower_bound_mismatch_rejected(self):
        cert = self._base_cert()
        cert["buffer"]["B_lower_bound"] = "1/1"
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify(cert)
        self.assertIn("B_lower_bound mismatch", str(ctx.exception))

    def test_redshift_layer_zero_width_rejected(self):
        cert = self._base_cert()
        cert["regions"]["redshift_layer"]["endpoints"] = ["1/1", "1/1"]
        with self.assertRaises(ValueError) as ctx:
            rrbb.verify(cert)
        self.assertIn("lo < hi", str(ctx.exception))


class TestComposedVerifier(unittest.TestCase):
    def test_composed_pass(self):
        result = composed.verify_composed(
            CWQP_DIR / "toy_cwqp_small_loss_certificate.json",
            RRBB_DIR / "toy_rrbb_with_cwqp_compat.json",
        )
        self.assertTrue(result["accepted"])
        self.assertTrue(result["composed_closure"])
        self.assertEqual(Fraction(result["cwqp_certificate"]["total_loss"]), 6)
        self.assertEqual(Fraction(result["rrbb_certificate"]["B_lower_bound"]), 6)
        self.assertEqual(Fraction(result["margin"]), 0)

    def test_composed_fail(self):
        result = composed.verify_composed(
            CWQP_DIR / "toy_cwqp_assembly_certificate.json",
            RRBB_DIR / "toy_rrbb_certificate.json",
        )
        # Both sub-certificates are individually valid, but the composed
        # closure fails because the CWQP total loss exceeds the RRBB buffer.
        self.assertFalse(result["accepted"])
        self.assertFalse(result["composed_closure"])
        self.assertEqual(Fraction(result["cwqp_certificate"]["total_loss"]), 132)
        self.assertEqual(Fraction(result["rrbb_certificate"]["B_lower_bound"]),
                         Fraction(5, 32))
        # Margin is negative when closure fails.
        self.assertLess(Fraction(result["margin"]), 0)


if __name__ == "__main__":
    unittest.main()
