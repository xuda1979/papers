#!/usr/bin/env python3
"""Tests for verify_carter_wick_composition.py and
verify_composed_cwcc_cwqp_rrbb.py."""

from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import verify_carter_wick_composition as cwcc  # noqa: E402
import verify_composed_cwcc_cwqp_rrbb as composed_triple  # noqa: E402

CWCC_DIR = ROOT / "certificates" / "cwcc"
CWQP_DIR = ROOT / "certificates" / "cwqp_assembly"
RRBB_DIR = ROOT / "certificates" / "rrbb"


class TestComputeRho(unittest.TestCase):
    def test_formula(self):
        # rho = kappa * N^2 * m^2 * M * (1 + C/w)
        # = (1/4) * 4 * 1 * 1 * (1 + 1/(1/2)) = (1/4)*4*3 = 3
        rho = cwcc.compute_rho(
            Fraction(1, 4), 2, 1, Fraction(1), Fraction(1), Fraction(1, 2)
        )
        self.assertEqual(rho, 3)

    def test_formula_zero_bump(self):
        # With C_bump = 0: rho = kappa * N^2 * m^2 * M.
        rho = cwcc.compute_rho(
            Fraction(2), 3, 2, Fraction(5), Fraction(0), Fraction(1, 3)
        )
        self.assertEqual(rho, Fraction(2) * 9 * 4 * 5)

    def test_formula_rational(self):
        rho = cwcc.compute_rho(
            Fraction(3, 7), 2, 3, Fraction(2, 5), Fraction(1, 4), Fraction(1, 8)
        )
        expected = (
            Fraction(3, 7) * 4 * 9 * Fraction(2, 5) * (1 + Fraction(1, 4) / Fraction(1, 8))
        )
        self.assertEqual(rho, expected)


class TestComputeLosses(unittest.TestCase):
    def test_loss_v1(self):
        # (M/mu) * (1 + C/w) = (1/4) * (1 + 2) = 3/4
        loss = cwcc.compute_loss_v1(
            Fraction(1, 4), Fraction(1), Fraction(1), Fraction(1, 2)
        )
        self.assertEqual(loss, Fraction(3, 4))

    def test_loss_comp(self):
        # rho/mu
        loss = cwcc.compute_loss_comp(Fraction(3, 16), Fraction(1))
        self.assertEqual(loss, Fraction(3, 16))

    def test_loss_v2(self):
        loss = cwcc.compute_loss_v2(Fraction(3, 4), Fraction(3, 16))
        self.assertEqual(loss, Fraction(15, 16))


class TestToyCWCCCerts(unittest.TestCase):
    def test_toy_cwcc_accepted(self):
        result = cwcc.verify_file(CWCC_DIR / "toy_cwcc_certificate.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(Fraction(result["composition_loss"]["total_loss_v2"]), 12)
        per_chart = result["composition_loss"]["per_chart"]
        self.assertEqual(len(per_chart), 2)
        for entry in per_chart:
            self.assertEqual(Fraction(entry["rho_alpha"]), 3)
            self.assertEqual(Fraction(entry["loss_comp"]), 3)
            self.assertEqual(Fraction(entry["loss_v1"]), 3)
            self.assertEqual(Fraction(entry["loss_v2"]), 6)

    def test_toy_small_accepted_with_rrbb_compat(self):
        result = cwcc.verify_file(
            CWCC_DIR / "toy_cwcc_small_composition_certificate.json"
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(
            Fraction(result["composition_loss"]["total_loss_v2"]), Fraction(15, 8)
        )
        self.assertTrue(result["rrbb_compatibility"]["closure"])


class TestBadCWCCCerts(unittest.TestCase):
    def test_rho_mismatch_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify_file(CWCC_DIR / "bad_cwcc_rho_mismatch.json")
        self.assertIn("rho_alpha mismatch", str(ctx.exception))

    def test_composition_fail_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify_file(CWCC_DIR / "bad_cwcc_composition_fail.json")
        self.assertIn("closure mismatch", str(ctx.exception))


class TestHandcraftedCWCC(unittest.TestCase):
    def _base_cert(self) -> dict:
        return {
            "metadata": {
                "name": "Carter–Wick CWCC test",
                "generated_on": "2026-07-03",
                "spin": "toy",
                "claim_scope": "toy",
                "status": "toy",
                "warning": "toy",
            },
            "carter_diagonal_constant": "1/4",
            "bump": {"bump_constant": "1/1"},
            "charts": [
                {"name": "chart_A", "monomial_basis_size": 2,
                 "max_symbol_order": 1, "gram_op_norm_bound": "1/1",
                 "local_margin": "1/1", "width": "1/2"},
                {"name": "chart_B", "monomial_basis_size": 2,
                 "max_symbol_order": 1, "gram_op_norm_bound": "1/1",
                 "local_margin": "1/1", "width": "1/2"},
            ],
            "composition_loss": {
                "per_chart": [
                    {"name": "chart_A"},
                    {"name": "chart_B"},
                ],
            },
        }

    def test_valid(self):
        result = cwcc.verify(self._base_cert())
        self.assertTrue(result["accepted"])
        self.assertEqual(Fraction(result["composition_loss"]["total_loss_v2"]), 12)

    def test_zero_kappa_rejected(self):
        cert = self._base_cert()
        cert["carter_diagonal_constant"] = "0/1"
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify(cert)
        self.assertIn("carter_diagonal_constant", str(ctx.exception))

    def test_negative_kappa_rejected(self):
        cert = self._base_cert()
        cert["carter_diagonal_constant"] = "-1/4"
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify(cert)
        self.assertIn("carter_diagonal_constant", str(ctx.exception))

    def test_zero_monomial_basis_size_rejected(self):
        cert = self._base_cert()
        cert["charts"][0]["monomial_basis_size"] = 0
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify(cert)
        self.assertIn("monomial_basis_size", str(ctx.exception))

    def test_zero_max_symbol_order_rejected(self):
        cert = self._base_cert()
        cert["charts"][0]["max_symbol_order"] = 0
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify(cert)
        self.assertIn("max_symbol_order", str(ctx.exception))

    def test_negative_width_rejected(self):
        cert = self._base_cert()
        cert["charts"][0]["width"] = "-1/2"
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify(cert)
        self.assertIn("width", str(ctx.exception))

    def test_zero_margin_rejected(self):
        cert = self._base_cert()
        cert["charts"][0]["local_margin"] = "0/1"
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify(cert)
        self.assertIn("local_margin", str(ctx.exception))

    def test_wrong_total_loss_v2_rejected(self):
        cert = self._base_cert()
        cert["composition_loss"]["total_loss_v2"] = "999/1"
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify(cert)
        self.assertIn("total_loss_v2 mismatch", str(ctx.exception))

    def test_per_chart_name_mismatch_rejected(self):
        cert = self._base_cert()
        cert["composition_loss"]["per_chart"][0]["name"] = "chart_X"
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify(cert)
        self.assertIn("name mismatch", str(ctx.exception))

    def test_per_chart_length_mismatch_rejected(self):
        cert = self._base_cert()
        cert["composition_loss"]["per_chart"] = [
            {"name": "chart_A"},
        ]
        with self.assertRaises(ValueError) as ctx:
            cwcc.verify(cert)
        self.assertIn("length must match", str(ctx.exception))


class TestComposedTripleVerifier(unittest.TestCase):
    def test_composed_triple_pass(self):
        result = composed_triple.verify_composed_triple(
            CWCC_DIR / "toy_cwcc_small_composition_certificate.json",
            CWQP_DIR / "toy_cwqp_matching_small_certificate.json",
            RRBB_DIR / "toy_rrbb_with_cwqp_compat.json",
        )
        self.assertTrue(result["accepted"])
        self.assertTrue(result["composed_closure"])
        self.assertEqual(
            Fraction(result["cwcc_certificate"]["total_loss_v2"]),
            Fraction(15, 8),
        )
        self.assertEqual(
            Fraction(result["cwqp_certificate"]["total_loss_v1"]),
            Fraction(3, 2),
        )
        self.assertEqual(
            Fraction(result["rrbb_certificate"]["B_lower_bound"]), 6
        )
        self.assertEqual(Fraction(result["v2_over_v1_excess"]),
                         Fraction(15, 8) - Fraction(3, 2))
        self.assertGreater(Fraction(result["margin"]), 0)

    def test_composed_triple_fail_loss_too_big(self):
        result = composed_triple.verify_composed_triple(
            CWCC_DIR / "toy_cwcc_certificate.json",
            CWQP_DIR / "toy_cwqp_small_loss_certificate.json",
            RRBB_DIR / "toy_rrbb_with_cwqp_compat.json",
        )
        # CWCC total_loss_v2=12 > RRBB B_lower_bound=6 -> closure fails.
        self.assertFalse(result["accepted"])
        self.assertFalse(result["composed_closure"])
        self.assertLess(Fraction(result["margin"]), 0)

    def test_composed_triple_chart_name_disagreement_rejected(self):
        # CWCC cert uses chart_A/chart_B; CWQP toy_cwqp_assembly uses chart_A/chart_B/chart_C.
        with self.assertRaises(ValueError) as ctx:
            composed_triple.verify_composed_triple(
                CWCC_DIR / "toy_cwcc_certificate.json",
                CWQP_DIR / "toy_cwqp_assembly_certificate.json",
                RRBB_DIR / "toy_rrbb_with_cwqp_compat.json",
            )
        self.assertIn("chart name sets disagree", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
