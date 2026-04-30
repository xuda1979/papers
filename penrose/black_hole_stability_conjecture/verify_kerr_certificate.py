#!/usr/bin/env python3
"""Validate a Kerr stability certificate scaffold.

This checker is deliberately modest: it verifies certificate structure and
rational interval margins. It does not discover the Kerr estimates. A passing
certificate means the proposed data has the compatibility shape required by the
paper's machine-checkable certificate definition.
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

    def strictly_dominates(self, other: "Interval") -> bool:
        """Return true when this interval is provably larger than another."""
        return self.lo > other.hi


def fail(message: str) -> None:
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def parse_fraction(raw: str) -> Fraction:
    try:
        return Fraction(raw.strip())
    except Exception as exc:  # pragma: no cover - message is for CLI users
        raise ValueError(f"invalid rational number {raw!r}") from exc


def parse_interval(raw: Any, path: str) -> Interval:
    require(isinstance(raw, str), f"{path} must be an interval string")
    text = raw.strip()
    require(text.startswith("[") and text.endswith("]"), f"{path} must look like [lo, hi]")
    parts = text[1:-1].split(",")
    require(len(parts) == 2, f"{path} must contain exactly two endpoints")
    lo = parse_fraction(parts[0])
    hi = parse_fraction(parts[1])
    require(lo <= hi, f"{path} has lo > hi")
    return Interval(lo, hi)


def require_keys(obj: dict[str, Any], keys: list[str], path: str) -> None:
    for key in keys:
        require(key in obj, f"{path}.{key} is missing")


def validate_metadata(data: dict[str, Any]) -> None:
    meta = data.get("metadata")
    require(isinstance(meta, dict), "metadata must be an object")
    require_keys(
        meta,
        ["name", "spin", "sobolev_index", "weight", "frequency_decomposition", "symbol_class"],
        "metadata",
    )
    parse_fraction(str(meta["spin"]))
    require(isinstance(meta["sobolev_index"], int) and meta["sobolev_index"] >= 8, "sobolev_index must be an integer >= 8")


def validate_energy(data: dict[str, Any]) -> None:
    entries = data.get("energy_positivity")
    require(isinstance(entries, list) and entries, "energy_positivity must be a nonempty list")
    sectors = set()
    for index, entry in enumerate(entries):
        path = f"energy_positivity[{index}]"
        require(isinstance(entry, dict), f"{path} must be an object")
        require_keys(entry, ["sector", "chart", "gram_identity", "lower_bound", "remainder_absorption"], path)
        sectors.add(entry["sector"])
        lower = parse_interval(entry["lower_bound"], f"{path}.lower_bound")
        absorption = parse_interval(entry["remainder_absorption"], f"{path}.remainder_absorption")
        require(lower.positive, f"{path}.lower_bound must be positive")
        require(absorption.nonnegative, f"{path}.remainder_absorption must be nonnegative")
        require(lower.strictly_dominates(absorption), f"{path}.lower_bound must dominate remainder_absorption")
    required = {"redshift", "exterior", "ergoregion_hidden_symmetry"}
    require(required.issubset(sectors), f"energy_positivity must include sectors {sorted(required)}")


def validate_morawetz(data: dict[str, Any]) -> None:
    entries = data.get("morawetz_positivity")
    require(isinstance(entries, list) and entries, "morawetz_positivity must be a nonempty list")
    has_ns = False
    for index, entry in enumerate(entries):
        path = f"morawetz_positivity[{index}]"
        require(isinstance(entry, dict), f"{path} must be an object")
        require_keys(entry, ["channel", "chart", "square_decomposition", "bulk_lower_bound", "all_remainders_absorbable"], path)
        has_ns = has_ns or entry["channel"] == "nonsuperradiant"
        require(parse_interval(entry["bulk_lower_bound"], f"{path}.bulk_lower_bound").positive, f"{path}.bulk_lower_bound must be positive")
        require(entry["all_remainders_absorbable"] is True, f"{path}.all_remainders_absorbable must be true")
    require(has_ns, "morawetz_positivity must include the nonsuperradiant channel")


def validate_trapped_superradiant(data: dict[str, Any]) -> None:
    trapped = data.get("trapped_superradiant")
    require(isinstance(trapped, dict), "trapped_superradiant must be an object")
    charts = trapped.get("torus_charts")
    require(isinstance(charts, list) and charts, "trapped_superradiant.torus_charts must be nonempty")
    for index, chart in enumerate(charts):
        path = f"trapped_superradiant.torus_charts[{index}]"
        require(isinstance(chart, dict), f"{path} must be an object")
        require_keys(chart, ["chart", "fourier_polynomial", "approximation_error", "resonant_average_lower_bound"], path)
        approximation = parse_interval(chart["approximation_error"], f"{path}.approximation_error")
        resonant = parse_interval(chart["resonant_average_lower_bound"], f"{path}.resonant_average_lower_bound")
        require(approximation.nonnegative, f"{path}.approximation_error must be nonnegative")
        require(resonant.positive, f"{path}.resonant_average_lower_bound must be positive")
        require(resonant.strictly_dominates(approximation), f"{path}.resonant_average_lower_bound must dominate approximation_error")
    glob = trapped.get("symbol_globalization")
    require(isinstance(glob, dict), "trapped_superradiant.symbol_globalization must be an object")
    require_keys(glob, ["escape_function", "symbol_class", "fixed_neighborhood_remainder", "all_remainders_absorbable"], "trapped_superradiant.symbol_globalization")
    require(parse_interval(glob["fixed_neighborhood_remainder"], "trapped_superradiant.symbol_globalization.fixed_neighborhood_remainder").nonnegative, "fixed_neighborhood_remainder must be nonnegative")
    require(glob["all_remainders_absorbable"] is True, "symbol globalization remainders must be absorbable")


def validate_nonlinear(data: dict[str, Any]) -> None:
    nl = data.get("nonlinear_null_hierarchy")
    require(isinstance(nl, dict), "nonlinear_null_hierarchy must be an object")
    require_keys(
        nl,
        [
            "all_quadratic_terms_classified",
            "all_cubic_terms_classified",
            "source_decay_eta",
            "source_constant",
            "modulation_decay_eta",
            "modulation_constant",
        ],
        "nonlinear_null_hierarchy",
    )
    require(nl["all_quadratic_terms_classified"] is True, "quadratic terms must all be classified")
    require(nl["all_cubic_terms_classified"] is True, "cubic terms must all be classified")
    for key in ["source_decay_eta", "source_constant", "modulation_decay_eta", "modulation_constant"]:
        require(parse_interval(nl[key], f"nonlinear_null_hierarchy.{key}").positive, f"{key} must be positive")


def validate_compatibility(data: dict[str, Any]) -> None:
    meta = data["metadata"]
    comp = data.get("compatibility")
    require(isinstance(comp, dict), "compatibility must be an object")
    require_keys(comp, ["sobolev_index", "weight", "frequency_decomposition", "symbol_class", "max_derivative_loss", "available_derivative_buffer"], "compatibility")
    for key in ["sobolev_index", "weight", "frequency_decomposition", "symbol_class"]:
        require(comp[key] == meta[key], f"compatibility.{key} must match metadata.{key}")
    require(isinstance(comp["max_derivative_loss"], int) and comp["max_derivative_loss"] >= 0, "max_derivative_loss must be a nonnegative integer")
    require(isinstance(comp["available_derivative_buffer"], int), "available_derivative_buffer must be an integer")
    require(comp["available_derivative_buffer"] >= comp["max_derivative_loss"], "available derivative buffer must cover derivative loss")
    trapped_symbol = data["trapped_superradiant"]["symbol_globalization"]["symbol_class"]
    require(trapped_symbol == meta["symbol_class"], "trapped symbol class must match metadata symbol_class")


def validate(data: dict[str, Any]) -> None:
    validate_metadata(data)
    validate_energy(data)
    validate_morawetz(data)
    validate_trapped_superradiant(data)
    validate_nonlinear(data)
    validate_compatibility(data)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_kerr_certificate.py CERTIFICATE.json", file=sys.stderr)
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
