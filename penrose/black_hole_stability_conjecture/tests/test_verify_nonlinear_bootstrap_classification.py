#!/usr/bin/env python3
"""Tests for verify_nonlinear_bootstrap_classification.py and verify_composed_full_stack.py."""

from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import verify_nonlinear_bootstrap_classification as nbct  # noqa: E402
import verify_composed_full_stack as full_stack  # noqa: E402

NBCT_DIR = ROOT / "certificates" / "nbct"
CWCC_DIR = ROOT / "certificates" / "cwcc"
CWQP_DIR = ROOT / "certificates" / "cwqp_assembly"
RRBB_DIR = ROOT / "certificates" / "rrbb"

RRBB_PATH = RRBB_DIR / "toy_rrbb_with_cwqp_compat.json"


class TestComputeBudgets(unittest.TestCase):
    def test_nf_only(self):
        records = [
            {"name": "T1", "class": "NF", "coefficient": Fraction(1, 4), "weight": Fraction(1, 4)},
            {"name": "T2", "class": "NF", "coefficient": Fraction(3, 8), "weight": Fraction(3, 8)},
        ]
        budgets = nbct.compute_budgets(records)
        self.assertEqual(budgets["B_NF"], Fraction(5, 8))
        self.assertEqual(budgets["B_RS"], 0)
        self.assertEqual(budgets["B_IR"], 0)
        self.assertEqual(budgets["B_NL"], Fraction(5, 8))

    def test_mixed_classes(self):
        records = [
            {"name": "T1", "class": "NF", "coefficient": Fraction(1, 4), "weight": Fraction(1, 4)},
            {"name": "T2", "class": "RS", "coefficient": Fraction(1, 10), "weight": Fraction(1, 2)},
            {"name": "T3", "class": "IR", "coefficient": Fraction(1, 4), "weight": Fraction(1, 16)},
        ]
        budgets = nbct.compute_budgets(records)
        self.assertEqual(budgets["B_NF"], Fraction(1, 4))
        self.assertEqual(budgets["B_RS"], Fraction(1, 2))
        self.assertEqual(budgets["B_IR"], Fraction(1, 16))
        self.assertEqual(budgets["B_NL"], Fraction(13, 16))


class TestFillRsWeights(unittest.TestCase):
    def test_rs_weight(self):
        records = [
            {"name": "T1", "class": "RS", "coefficient": Fraction(1, 10), "weight": None},
            {"name": "T2", "class": "NF", "coefficient": Fraction(1, 4), "weight": Fraction(1, 4)},
        ]
        nbct.fill_rs_weights(records, Fraction(10), Fraction(1, 2))
        self.assertEqual(records[0]["weight"], Fraction(1, 2))
        self.assertEqual(records[1]["weight"], Fraction(1, 4))


class TestToyNBCT(unittest.TestCase):
    def test_toy_accepted(self):
        result = nbct.verify_file(NBCT_DIR / "toy_nbct_certificate.json")
        self.assertTrue(result["accepted"])
        self.assertEqual(Fraction(result["nonlinear_budget"]["B_NL"]), Fraction(13, 16))
        self.assertEqual(Fraction(result["bootstrap_closure"]["epsilon_boot"]), 2)
        self.assertTrue(result["bootstrap_closure"]["closure"])
        self.assertFalse(result["bootstrap_closure"]["tight"])
        self.assertEqual(result["source_terms_by_class"], {"NF": 1, "RS": 1, "IR": 1})


class TestBadNBCT(unittest.TestCase):
    def test_budget_mismatch_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            nbct.verify_file(NBCT_DIR / "bad_nbct_budget_mismatch.json")
        self.assertIn("B_NF mismatch", str(ctx.exception))

    def test_closure_fail_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            nbct.verify_file(NBCT_DIR / "bad_nbct_closure_fail.json")
        self.assertIn("closure mismatch", str(ctx.exception))

    def test_invalid_class_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            nbct.verify_file(NBCT_DIR / "bad_nbct_invalid_class.json")
        self.assertIn("class must be one of", str(ctx.exception))


class TestHandcraftedNBCT(unittest.TestCase):
    def _base_cert(self):
        return {
            "metadata": {
                "name": "Nonlinear Bootstrap NBCT test",
                "generated_on": "2026-07-03",
                "spin": "toy",
                "claim_scope": "toy",
                "status": "toy",
                "warning": "toy",
            },
            "linear_stack_reference": {
                "rrbb_certificate_ref": "certificates/rrbb/toy_rrbb_with_cwqp_compat.json",
                "B_lower_bound": "6/1",
            },
            "bootstrap_parameters": {"tau_0": "4/1", "sobolev_index": 6},
            "source_terms": [
                {"name": "T1", "class": "NF", "coefficient": "1/4"},
                {"name": "T2", "class": "RS", "coefficient": "1/10"},
                {"name": "T3", "class": "IR", "coefficient": "1/4", "decay_exponent": "1/1"},
            ],
            "nonlinear_budget": {
                "B_NF": "1/4", "B_RS": "1/2", "B_IR": "1/16", "B_NL": "13/16",
            },
            "bootstrap_closure": {
                "epsilon_boot": "2/1", "B_lower_bound": "6/1", "closure": True,
            },
        }

    def test_valid(self):
        result = nbct.verify(self._base_cert(), rrbb_path=RRBB_PATH)
        self.assertTrue(result["accepted"])

    def test_negative_coefficient_rejected(self):
        cert = self._base_cert()
        cert["source_terms"][0]["coefficient"] = "-1/4"
        with self.assertRaises(ValueError) as ctx:
            nbct.verify(cert, rrbb_path=RRBB_PATH)
        self.assertIn("coefficient", str(ctx.exception))

    def test_tau_0_below_1_rejected(self):
        cert = self._base_cert()
        cert["bootstrap_parameters"]["tau_0"] = "1/2"
        with self.assertRaises(ValueError) as ctx:
            nbct.verify(cert, rrbb_path=RRBB_PATH)
        self.assertIn("tau_0", str(ctx.exception))

    def test_ir_noninteger_decay_rejected(self):
        cert = self._base_cert()
        cert["source_terms"][2]["decay_exponent"] = "1/2"
        with self.assertRaises(ValueError) as ctx:
            nbct.verify(cert, rrbb_path=RRBB_PATH)
        self.assertIn("integer", str(ctx.exception))

    def test_b_lower_bound_mismatch_rejected(self):
        cert = self._base_cert()
        cert["bootstrap_closure"]["B_lower_bound"] = "7/1"
        with self.assertRaises(ValueError) as ctx:
            nbct.verify(cert, rrbb_path=RRBB_PATH)
        self.assertIn("B_lower_bound mismatch", str(ctx.exception))

    def test_duplicate_source_name_rejected(self):
        cert = self._base_cert()
        cert["source_terms"][1]["name"] = "T1"
        with self.assertRaises(ValueError) as ctx:
            nbct.verify(cert, rrbb_path=RRBB_PATH)
        self.assertIn("duplicates", str(ctx.exception))


class TestFullStack(unittest.TestCase):
    def test_full_stack_pass(self):
        result = full_stack.verify_full_stack(
            CWCC_DIR / "toy_cwcc_small_composition_certificate.json",
            CWQP_DIR / "toy_cwqp_matching_small_certificate.json",
            RRBB_DIR / "toy_rrbb_with_cwqp_compat.json",
            NBCT_DIR / "toy_nbct_certificate.json",
        )
        self.assertTrue(result["accepted"])
        self.assertTrue(result["composed_nonlinear_closure"])
        self.assertEqual(Fraction(result["nbct_certificate"]["B_NL"]), Fraction(13, 16))
        self.assertEqual(Fraction(result["nbct_certificate"]["epsilon_boot"]), 2)
        self.assertEqual(Fraction(result["B_lower_bound"]), 6)

    def test_full_stack_fail_closure(self):
        with self.assertRaises(ValueError) as ctx:
            full_stack.verify_full_stack(
                CWCC_DIR / "toy_cwcc_small_composition_certificate.json",
                CWQP_DIR / "toy_cwqp_matching_small_certificate.json",
                RRBB_DIR / "toy_rrbb_with_cwqp_compat.json",
                NBCT_DIR / "bad_nbct_closure_fail.json",
            )
        self.assertIn("closure mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
