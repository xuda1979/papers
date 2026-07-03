#!/usr/bin/env python3
"""Validate rational coefficient-extraction certificates for Carter densities.

The checker closes the arithmetic part of a local extraction backend: rational
coefficient expressions are evaluated by exact interval arithmetic and then
compared against an already accepted coefficient enclosure.  It does not prove
that the expressions are the true Kerr gravitational coefficients; that
identification must be supplied by a separate exact derivation backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_grav_coefficient_enclosure as enclosure
import verify_carter_sos_identity as sos


CLAIM_SCOPE = "chart_rational_coefficient_extraction"
ALLOWED_DERIVATION_STATUSES = {
    "model_probe",
    "candidate_extraction",
    "kerr_gravitational_proved",
}


@dataclass(frozen=True)
class RationalInterval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"invalid interval [{self.lo}, {self.hi}]")

    def __add__(self, other: "RationalInterval") -> "RationalInterval":
        return RationalInterval(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self) -> "RationalInterval":
        return RationalInterval(-self.hi, -self.lo)

    def __sub__(self, other: "RationalInterval") -> "RationalInterval":
        return self + (-other)

    def __mul__(self, other: "RationalInterval") -> "RationalInterval":
        candidates = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        return RationalInterval(min(candidates), max(candidates))

    def reciprocal(self) -> "RationalInterval":
        if self.lo <= 0 <= self.hi:
            raise ValueError(f"division by interval meeting zero [{self.lo}, {self.hi}]")
        candidates = [Fraction(1, self.lo), Fraction(1, self.hi)]
        return RationalInterval(min(candidates), max(candidates))

    def __truediv__(self, other: "RationalInterval") -> "RationalInterval":
        return self * other.reciprocal()

    def square(self) -> "RationalInterval":
        if self.lo <= 0 <= self.hi:
            return RationalInterval(Fraction(0), max(self.lo * self.lo, self.hi * self.hi))
        candidates = [self.lo * self.lo, self.hi * self.hi]
        return RationalInterval(min(candidates), max(candidates))

    def contains(self, other: "RationalInterval") -> bool:
        return self.lo <= other.lo and other.hi <= self.hi

    def to_json(self) -> str:
        return f"[{self.lo}, {self.hi}]"


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


def resolve_path(raw_path: Any, base_dir: Path | None, path_name: str) -> Path:
    require(isinstance(raw_path, str) and raw_path.strip(), f"{path_name} must be a nonempty string")
    path = Path(raw_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def parse_interval(raw: Any, path: str) -> RationalInterval:
    require(isinstance(raw, str), f"{path} must be an interval string")
    text = raw.strip()
    require(text.startswith("[") and text.endswith("]"), f"{path} must look like [lo, hi]")
    parts = text[1:-1].split(",")
    require(len(parts) == 2, f"{path} must contain exactly two endpoints")
    return RationalInterval(Fraction(parts[0].strip()), Fraction(parts[1].strip()))


def point(value: Fraction | int) -> RationalInterval:
    rational = Fraction(value)
    return RationalInterval(rational, rational)


def validate_metadata(data: dict[str, Any]) -> str:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "claim_scope", "status", "derivation_status"], "metadata")
    require(metadata["claim_scope"] == CLAIM_SCOPE, f"metadata.claim_scope must be {CLAIM_SCOPE}")
    derivation_status = metadata["derivation_status"]
    require(
        derivation_status in ALLOWED_DERIVATION_STATUSES,
        f"metadata.derivation_status must be one of {sorted(ALLOWED_DERIVATION_STATUSES)}",
    )
    if derivation_status == "kerr_gravitational_proved":
        raise ValueError(
            "metadata.derivation_status kerr_gravitational_proved requires an exact Kerr derivation "
            "backend; no such backend is registered in this checker"
        )
    return str(derivation_status)


def validate_source_metadata(data: dict[str, Any], base_dir: Path | None) -> None:
    source = data.get("source_metadata")
    require(isinstance(source, dict), "source_metadata must be an object")
    require_keys(source, ["ansatz_path", "normalization", "derivation_note"], "source_metadata")
    ansatz_path = resolve_path(source["ansatz_path"], base_dir, "source_metadata.ansatz_path")
    require(ansatz_path.exists(), f"source_metadata.ansatz_path does not exist: {ansatz_path}")
    require(isinstance(source["normalization"], str) and source["normalization"], "source_metadata.normalization must be a nonempty string")
    require(isinstance(source["derivation_note"], str) and source["derivation_note"], "source_metadata.derivation_note must be a nonempty string")


def parse_symbols(raw_symbols: Any) -> dict[str, RationalInterval]:
    require(isinstance(raw_symbols, dict) and raw_symbols, "interval_symbols must be a nonempty object")
    symbols: dict[str, RationalInterval] = {}
    for name, raw_interval in raw_symbols.items():
        require(isinstance(name, str) and name.strip(), "interval symbol names must be nonempty strings")
        symbols[name] = parse_interval(raw_interval, f"interval_symbols.{name}")
    return symbols


def eval_expr(expr: Any, symbols: dict[str, RationalInterval]) -> RationalInterval:
    if isinstance(expr, int):
        return point(expr)
    if isinstance(expr, str):
        require(expr in symbols, f"unknown interval symbol {expr!r}")
        return symbols[expr]
    if isinstance(expr, dict):
        if "const" in expr:
            return point(sos.parse_fraction(expr["const"]))
        if "var" in expr:
            name = str(expr["var"])
            require(name in symbols, f"unknown interval symbol {name!r}")
            return symbols[name]
        if "interval" in expr:
            return parse_interval(expr["interval"], "inline interval")
        if "neg" in expr:
            return -eval_expr(expr["neg"], symbols)
        if "sum" in expr:
            items = expr["sum"]
            require(isinstance(items, list), "sum expression must be a list")
            total = point(0)
            for item in items:
                total = total + eval_expr(item, symbols)
            return total
        if "prod" in expr:
            items = expr["prod"]
            require(isinstance(items, list), "prod expression must be a list")
            total = point(1)
            for item in items:
                total = total * eval_expr(item, symbols)
            return total
        if "square" in expr:
            return eval_expr(expr["square"], symbols).square()
        if "scale" in expr:
            spec = expr["scale"]
            require(isinstance(spec, dict) and "coeff" in spec and "expr" in spec, "scale expression must contain coeff and expr")
            return point(sos.parse_fraction(spec["coeff"])) * eval_expr(spec["expr"], symbols)
        if "div" in expr:
            spec = expr["div"]
            require(
                isinstance(spec, dict) and "numerator" in spec and "denominator" in spec,
                "div expression must contain numerator and denominator",
            )
            return eval_expr(spec["numerator"], symbols) / eval_expr(spec["denominator"], symbols)
    raise ValueError(f"unsupported interval expression {expr!r}")


def evaluate_coefficient_expressions(
    raw_expressions: Any,
    symbols: dict[str, RationalInterval],
    expected_keys: set[str],
) -> dict[str, RationalInterval]:
    require(isinstance(raw_expressions, dict) and raw_expressions, "coefficient_expressions must be a nonempty object")
    actual_keys = set(raw_expressions)
    require(actual_keys == expected_keys, f"coefficient_expressions must contain exactly {sorted(expected_keys)}")
    return {key: eval_expr(raw_expressions[key], symbols) for key in sorted(expected_keys)}


def validate_expression_intervals_fit_enclosure(
    expression_intervals: dict[str, RationalInterval],
    enclosure_data: dict[str, Any],
) -> None:
    raw_enclosure_intervals = enclosure_data.get("coefficient_intervals")
    require(isinstance(raw_enclosure_intervals, dict), "enclosure coefficient_intervals must be an object")
    for key, expression_interval in expression_intervals.items():
        target = parse_interval(raw_enclosure_intervals[key], f"enclosure.coefficient_intervals.{key}")
        require(
            target.contains(expression_interval),
            f"extracted coefficient interval {key!r} is not contained in accepted enclosure",
        )


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    require_keys(
        data,
        ["chart", "variables", "enclosure_certificate_path", "interval_symbols", "coefficient_expressions"],
        "certificate",
    )
    validate_source_metadata(data, base_dir)

    enclosure_path = resolve_path(data["enclosure_certificate_path"], base_dir, "enclosure_certificate_path")
    enclosure_data = load_json(enclosure_path)
    enclosure_result = enclosure.verify(enclosure_data, base_dir=enclosure_path.parent)

    chart = data["chart"]
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")
    require(chart == enclosure_result["chart"], "certificate chart must match enclosure chart")
    variables_raw = data["variables"]
    require(isinstance(variables_raw, list) and variables_raw, "variables must be a nonempty list")
    require(all(isinstance(variable, str) for variable in variables_raw), "variables must be strings")
    require(set(variables_raw) == set(enclosure_result["variables"]), "variables must match enclosure variables")

    expected_keys = set(enclosure_data["coefficient_intervals"])
    symbols = parse_symbols(data["interval_symbols"])
    expression_intervals = evaluate_coefficient_expressions(data["coefficient_expressions"], symbols, expected_keys)
    validate_expression_intervals_fit_enclosure(expression_intervals, enclosure_data)

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "chart": chart,
        "variables": list(variables_raw),
        "enclosure_certificate": str(enclosure_path),
        "local_coercivity_delta": enclosure_result["local_coercivity_delta"],
        "extracted_coefficient_intervals": {
            key: interval.to_json() for key, interval in sorted(expression_intervals.items())
        },
    }


def verify_file(path: Path) -> dict[str, Any]:
    return verify(load_json(path), base_dir=path.parent)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter rational coefficient extraction rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
