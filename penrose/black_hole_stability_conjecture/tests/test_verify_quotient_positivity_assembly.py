#!/usr/bin/env python3
"""Tests for verify_quotient_positivity_assembly.py."""

from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

# Make the repository root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import verify_quotient_positivity_assembly as cwqp  # noqa: E402

CERT_DIR = ROOT / "certificates" / "cwqp_assembly"


class TestComputePerChartLoss(unittest.TestCase):
    def test_formula(self):
        # M/mu * (1 + C/w) = 12/3 * (1 + 5/(1/2)) = 4 * 11 = 44
        loss = cwqp.compute_per_chart_loss(
            Fraction(12), Fraction(3), Fraction(5), Fraction(1, 2)
        )
        self.assertEqual(loss, 44)

    def test_formula_zero_bump(self):
        # With C_bump = 0, loss = M/mu.
        loss = cwqp.compute_per_chart_loss(
            Fraction(7), Fraction(2), Fraction(0), Fraction(1, 3)
        )
        self.assertEqual(loss, Fraction(7, 2))

    def test_formula_rational(self):
        loss = cwqp.compute_per_chart_loss(
            Fraction(3, 2), Fraction(2, 5), Fraction(1, 3), Fraction(1, 7)
        )
        expected = Fraction(3, 2) / Fraction(2, 5) * (1 + Fraction(1, 3) / Fraction(1, 7))
        self.assertEqual(loss, expected)


class TestAsFraction(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(cwqp.as_fraction("3/4"), Fraction(3, 4))
        self.assertEqual(cwqp.as_fraction("7"), Fraction(7))
        self.assertEqual(cwqp.as_fraction("-2/3"), Fraction(-2, 3))

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            cwqp.as_fraction(3)
        with self.assertRaises(ValueError):
            cwqp.as_fraction(None)

    def test_invalid_string(self):
        with self.assertRaises(ValueError):
            cwqp.as_fraction("not a fraction")
        with self.assertRaises(ValueError):
            cwqp.as_fraction("1/0")


class TestRequire(unittest.TestCase):
    def test_passes(self):
        cwqp.require(True, "should not raise")

    def test_raises(self):
        with self.assertRaises(ValueError) as ctx:
            cwqp.require(False, "boom")
        self.assertIn("boom", str(ctx.exception))


class TestToyCertificate(unittest.TestCase):
    def test_toy_certificate_accepted(self):
        result = cwqp.verify_file(CERT_DIR / "toy_cwqp_assembly_certificate.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["atlas_chart_count"], 3)
        self.assertEqual(result["positivity_chart_count"], 3)
        budget = result["budget"]
        self.assertTrue(budget["closure"])
        self.assertEqual(Fraction(budget["total_loss"]), Fraction(132))
        self.assertEqual(Fraction(budget["derivative_buffer"]), Fraction(140))
        self.assertEqual(len(budget["per_chart"]), 3)
        for entry in budget["per_chart"]:
            self.assertEqual(Fraction(entry["loss"]), Fraction(44))
            self.assertEqual(Fraction(entry["M_alpha"]), Fraction(12))
            self.assertEqual(Fraction(entry["mu_alpha"]), Fraction(3))
            self.assertEqual(Fraction(entry["w_alpha"]), Fraction(1, 2))


class TestBadCertificates(unittest.TestCase):
    def test_budget_overflow_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify_file(CERT_DIR / "bad_cwqp_assembly_budget_overflow.json")
        self.assertIn("closure mismatch", str(ctx.exception))

    def test_atlas_gap_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify_file(CERT_DIR / "bad_cwqp_assembly_atlas_gap.json")
        self.assertIn("gap", str(ctx.exception))


class TestHandcrafted(unittest.TestCase):
    def _base_cert(self) -> dict:
        return {
            "metadata": {
                "name": "Carter–Wick CWQP test",
                "generated_on": "2026-07-03",
                "spin": "toy",
                "claim_scope": "toy",
                "status": "toy",
                "warning": "toy",
            },
            "bump": {"bump_constant": "2/1", "bump_type": "smooth"},
            "atlas": [
                {"name": "A", "endpoints": ["0/1", "2/1"], "width": "1/2",
                 "overlap_multiplicity": 2},
                {"name": "B", "endpoints": ["1/1", "3/1"], "width": "1/2",
                 "overlap_multiplicity": 2},
            ],
            "charts": [
                {"name": "A", "monomial_basis_size": 2,
                 "gram_op_norm_bound": "4/1", "local_margin": "1/1",
                 "gram_certificate_ref": "toy.json"},
                {"name": "B", "monomial_basis_size": 2,
                 "gram_op_norm_bound": "4/1", "local_margin": "1/1",
                 "gram_certificate_ref": "toy.json"},
            ],
            "derivative_buffer": "100/1",
            "budget": {},
            "closure": True,
        }

    def test_valid_handcrafted(self):
        # Per-chart loss: (4/1) * (1 + 2/(1/2)) = 4 * 5 = 20. Total = 40. Buffer 100.
        cert = self._base_cert()
        result = cwqp.verify(cert)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["budget"]["closure"])
        self.assertEqual(Fraction(result["budget"]["total_loss"]), Fraction(40))

    def test_negative_margin_rejected(self):
        cert = self._base_cert()
        cert["charts"][0]["local_margin"] = "-1/1"
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify(cert)
        self.assertIn("local_margin", str(ctx.exception))

    def test_zero_buffer_rejected(self):
        cert = self._base_cert()
        cert["derivative_buffer"] = "0/1"
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify(cert)
        self.assertIn("derivative_buffer", str(ctx.exception))

    def test_width_too_large_rejected(self):
        # width must satisfy 2*width <= hi - lo. Here width = 3/2 but hi-lo = 2.
        cert = self._base_cert()
        cert["atlas"][0]["width"] = "3/2"
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify(cert)
        self.assertIn("width", str(ctx.exception))

    def test_mismatched_chart_names_rejected(self):
        cert = self._base_cert()
        cert["charts"][0]["name"] = "X"
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify(cert)
        self.assertIn("names", str(ctx.exception))

    def test_wrong_total_loss_rejected(self):
        cert = self._base_cert()
        cert["budget"]["total_loss"] = "999/1"
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify(cert)
        self.assertIn("total_loss mismatch", str(ctx.exception))

    def test_wrong_per_chart_loss_rejected(self):
        cert = self._base_cert()
        cert["budget"]["per_chart"] = [
            {"name": "A", "loss": "1/1"},
            {"name": "B", "loss": "20/1"},
        ]
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify(cert)
        self.assertIn("loss mismatch", str(ctx.exception))

    def test_negative_bump_constant_rejected(self):
        cert = self._base_cert()
        cert["bump"]["bump_constant"] = "-1/1"
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify(cert)
        self.assertIn("bump_constant", str(ctx.exception))

    def test_missing_metadata_field_rejected(self):
        cert = self._base_cert()
        del cert["metadata"]["warning"]
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify(cert)
        self.assertIn("warning", str(ctx.exception))

    def test_wrong_overlap_multiplicity_rejected(self):
        cert = self._base_cert()
        # A overlaps B (A=[0,2], B=[1,3]); declared multiplicity for A should be 2.
        cert["atlas"][0]["overlap_multiplicity"] = 1
        with self.assertRaises(ValueError) as ctx:
            cwqp.verify(cert)
        self.assertIn("overlap_multiplicity", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
