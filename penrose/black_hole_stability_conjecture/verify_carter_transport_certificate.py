#!/usr/bin/env python3
"""Validate a Renormalized Carter transport cone certificate.

This checks the arithmetic shape of a proposed SOS-plus-transport certificate.
It does not prove the Kerr estimates or verify symbolic identities; those must
be supplied by a future exact algebra backend.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Interval:
    lo: Fraction
    hi: Fraction

    @property
    def positive(self) -> bool:
        return self.lo > 0 and self.hi >= self.lo

    @property
    def nonnegative(self) -> bool:
        return self.lo >= 0 and self.hi >= self.lo

    def dominates(self, other: "Interval") -> bool:
        return self.lo > other.hi


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_interval(raw: Any, path: str) -> Interval:
    require(isinstance(raw, str), f"{path} must be an interval string")
    text = raw.strip()
    require(text.startswith("[") and text.endswith("]"), f"{path} must look like [lo, hi]")
    parts = text[1:-1].split(",")
    require(len(parts) == 2, f"{path} must contain exactly two endpoints")
    lo = Fraction(parts[0].strip())
    hi = Fraction(parts[1].strip())
    require(lo <= hi, f"{path} has lo > hi")
    return Interval(lo, hi)


def require_keys(obj: dict[str, Any], keys: list[str], path: str) -> None:
    for key in keys:
        require(key in obj, f"{path}.{key} is missing")


def validate(data: dict[str, Any]) -> None:
    require_keys(data, ["metadata", "renormalized_variables", "cone_identity", "torus_atlas", "quantization"], "certificate")

    metadata = data["metadata"]
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "spin", "symbol_class", "trapped_scale"], "metadata")
    Fraction(str(metadata["spin"]))
    require(parse_interval(metadata["trapped_scale"], "metadata.trapped_scale").positive, "trapped_scale must be positive")

    variables = data["renormalized_variables"]
    require(isinstance(variables, dict), "renormalized_variables must be an object")
    require_keys(variables, ["e", "l", "q", "q_nonnegative_margin"], "renormalized_variables")
    require(parse_interval(variables["q_nonnegative_margin"], "renormalized_variables.q_nonnegative_margin").nonnegative, "q margin must be nonnegative")

    cone = data["cone_identity"]
    require(isinstance(cone, dict), "cone_identity must be an object")
    require_keys(
        cone,
        [
            "target_density",
            "delta",
            "transport_symbol",
            "sum_of_squares",
            "superradiant_squares",
            "constraint_squares",
            "remainder_bound",
            "positive_margin",
        ],
        "cone_identity",
    )
    delta = parse_interval(cone["delta"], "cone_identity.delta")
    remainder = parse_interval(cone["remainder_bound"], "cone_identity.remainder_bound")
    margin = parse_interval(cone["positive_margin"], "cone_identity.positive_margin")
    require(delta.positive, "cone delta must be positive")
    require(remainder.nonnegative, "cone remainder must be nonnegative")
    require(margin.positive, "cone positive_margin must be positive")
    require(margin.dominates(remainder), "cone positive_margin must dominate remainder_bound")
    require(isinstance(cone["sum_of_squares"], list) and cone["sum_of_squares"], "sum_of_squares must be nonempty")
    require(isinstance(cone["constraint_squares"], list) and cone["constraint_squares"], "constraint_squares must be nonempty")
    require(isinstance(cone["superradiant_squares"], list), "superradiant_squares must be a list")

    atlas = data["torus_atlas"]
    require(isinstance(atlas, list) and atlas, "torus_atlas must be nonempty")
    for index, chart in enumerate(atlas):
        path = f"torus_atlas[{index}]"
        require(isinstance(chart, dict), f"{path} must be an object")
        require_keys(chart, ["chart", "nonresonant_coboundary_error", "resonant_margin"], path)
        error = parse_interval(chart["nonresonant_coboundary_error"], f"{path}.nonresonant_coboundary_error")
        resonant = parse_interval(chart["resonant_margin"], f"{path}.resonant_margin")
        require(error.nonnegative, f"{path}.nonresonant_coboundary_error must be nonnegative")
        require(resonant.positive, f"{path}.resonant_margin must be positive")
        require(resonant.dominates(error), f"{path}.resonant_margin must dominate nonresonant error")

    quantization = data["quantization"]
    require(isinstance(quantization, dict), "quantization must be an object")
    require_keys(quantization, ["scheme", "subprincipal_remainder", "garding_margin", "derivative_loss"], "quantization")
    subprincipal = parse_interval(quantization["subprincipal_remainder"], "quantization.subprincipal_remainder")
    garding = parse_interval(quantization["garding_margin"], "quantization.garding_margin")
    require(subprincipal.nonnegative, "subprincipal_remainder must be nonnegative")
    require(garding.positive, "garding_margin must be positive")
    require(garding.dominates(subprincipal), "garding_margin must dominate subprincipal_remainder")
    require(isinstance(quantization["derivative_loss"], int) and quantization["derivative_loss"] >= 0, "derivative_loss must be a nonnegative integer")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_carter_transport_certificate.py CERTIFICATE.json", file=sys.stderr)
        return 2
    path = Path(argv[1])
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        validate(data)
    except ValueError as exc:
        print(f"certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(f"certificate accepted: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
