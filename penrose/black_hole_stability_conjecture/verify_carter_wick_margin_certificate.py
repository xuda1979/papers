#!/usr/bin/env python3
"""Validate a finite-margin Carter-Wick quantization certificate.

The checker turns the analytic quantization step into an explicit rational loss
budget. It does not prove the underlying pseudodifferential calculus; it checks
that the certificate records the standard calculus hypotheses and that the
declared quantization losses fit inside a strict positive margin.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from verify_carter_transport_certificate import Interval, parse_interval


LOSS_FIELDS = [
    "square_garding_loss",
    "commutator_remainder",
    "subprincipal_loss",
    "smoothing_loss",
]

CONTRACT_FIELDS = [
    "real_principal_symbols",
    "properly_supported_quantization",
    "squares_quantized_as_adjoint_squares",
    "weyl_commutator_principal_symbol",
    "lower_order_terms_recorded",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_keys(obj: dict[str, Any], keys: list[str], path: str) -> None:
    for key in keys:
        require(key in obj, f"{path}.{key} is missing")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def format_fraction(value: Fraction) -> str:
    return str(value)


def interval_contains_upper(interval: Interval, value: Fraction) -> bool:
    return interval.lo <= value <= interval.hi


def validate_metadata(data: dict[str, Any]) -> None:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "claim_scope", "status"], "metadata")
    require(
        metadata["claim_scope"] == "carter_wick_finite_margin",
        "metadata.claim_scope must be carter_wick_finite_margin",
    )


def validate_symbol_class(data: dict[str, Any]) -> None:
    symbol_class = data.get("symbol_class")
    require(isinstance(symbol_class, dict), "symbol_class must be an object")
    require_keys(
        symbol_class,
        ["calculus", "operator_order", "compact_phase_neighborhood", "max_derivative_order"],
        "symbol_class",
    )
    require(isinstance(symbol_class["calculus"], str) and symbol_class["calculus"].strip(), "symbol_class.calculus must be nonempty")
    require(isinstance(symbol_class["operator_order"], int), "symbol_class.operator_order must be an integer")
    require(symbol_class["operator_order"] <= 0, "symbol_class.operator_order must be <= 0")
    require(symbol_class["compact_phase_neighborhood"] is True, "symbol_class.compact_phase_neighborhood must be true")
    require(
        isinstance(symbol_class["max_derivative_order"], int) and symbol_class["max_derivative_order"] >= 2,
        "symbol_class.max_derivative_order must be an integer >= 2",
    )


def validate_calculus_contract(data: dict[str, Any]) -> None:
    contract = data.get("calculus_contract")
    require(isinstance(contract, dict), "calculus_contract must be an object")
    require_keys(contract, CONTRACT_FIELDS, "calculus_contract")
    for field in CONTRACT_FIELDS:
        require(contract[field] is True, f"calculus_contract.{field} must be true")


def validate_loss_budget(data: dict[str, Any]) -> tuple[Fraction, Fraction, Fraction]:
    budget = data.get("loss_budget")
    require(isinstance(budget, dict), "loss_budget must be an object")
    require_keys(budget, LOSS_FIELDS + ["exported_quantization_loss_bound", "positive_margin"], "loss_budget")

    total_loss = Fraction(0)
    for field in LOSS_FIELDS:
        interval = parse_interval(budget[field], f"loss_budget.{field}")
        require(interval.nonnegative, f"loss_budget.{field} must be nonnegative")
        total_loss += interval.hi

    exported = parse_interval(
        budget["exported_quantization_loss_bound"],
        "loss_budget.exported_quantization_loss_bound",
    )
    margin = parse_interval(budget["positive_margin"], "loss_budget.positive_margin")
    require(exported.nonnegative, "loss_budget.exported_quantization_loss_bound must be nonnegative")
    require(interval_contains_upper(exported, total_loss), "exported quantization loss must contain total loss")
    require(margin.positive, "loss_budget.positive_margin must be positive")
    require(margin.dominates(exported), "loss_budget.positive_margin must dominate exported quantization loss")
    return total_loss, exported.hi, margin.lo


def validate_derivative_loss(data: dict[str, Any]) -> int:
    derivative_loss = data.get("derivative_loss")
    require(isinstance(derivative_loss, int) and derivative_loss >= 0, "derivative_loss must be a nonnegative integer")
    return derivative_loss


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    validate_metadata(data)
    validate_symbol_class(data)
    validate_calculus_contract(data)
    total_loss, exported_upper, margin_lower = validate_loss_budget(data)
    derivative_loss = validate_derivative_loss(data)
    return {
        "accepted": True,
        "claim_scope": data["metadata"]["claim_scope"],
        "total_quantization_loss": format_fraction(total_loss),
        "exported_quantization_loss_upper": format_fraction(exported_upper),
        "positive_margin_lower": format_fraction(margin_lower),
        "derivative_loss": derivative_loss,
    }


def verify_file(path: Path) -> dict[str, Any]:
    return verify(load_json(path))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter-Wick margin certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
