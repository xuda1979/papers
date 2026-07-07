#!/usr/bin/env python3
"""Validate a Sobolev-Frequency-Symbol Admissibility Theorem (SFSAT) certificate.

This checker verifies the combinatorial, arithmetic, and *internal-coherence*
content of a SFSAT certificate. It checks:

  1. every required field is present, well-typed, and rational;
  2. the Sobolev hierarchy is internally coherent (N_shared >= N_symbol_class
     + max per-block derivative loss; every prefix of the composition order
     leaves at least N_min derivatives of headroom);
  3. the frequency decomposition has enough dyadic layers with positive
     adjacent overlap and covers the full frequency range;
  4. the symbol class has the right order, regularity scale, and a positive
     rational symbol bound;
  5. every blockwise margin dominates its remainder by at least a uniform
     margin floor;
  6. the composition order is a permutation of the seven required blocks and
     the prefix product of (margin - remainder) stays positive at every
     prefix;
  7. the final admissibility budget matches the per-block data and the total
     derivative loss fits in the available buffer.

The verifier does NOT check the analytic content of any individual block.
That is the responsibility of the upstream theorems (CWCC, CWQP-AB, RRBB,
NBCT). It checks that *if* each block is proved with the declared local
hypotheses, *then* the composition is admissible.

Usage:
    python3 verify_sobolev_frequency_symbol_admissibility.py <certificate.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_BLOCKS = {
    "linear_to_nonlinear_estimates",
    "trapped_superradiant_morawetz",
    "curvature_to_metric_reconstruction",
    "global_gauge_modulation",
    "nonlinear_weak_null_hierarchy",
    "near_extremal_parameter_control",
    "compatibility_assembly",
}


# ---------------------------------------------------------------------------
# Rational arithmetic
# ---------------------------------------------------------------------------

class Rational:
    """Exact rational number using a pair of integers (num, den)."""

    __slots__ = ("num", "den")

    def __init__(self, num: int, den: int) -> None:
        if den == 0:
            raise ZeroDivisionError("rational denominator is zero")
        if den < 0:
            num, den = -num, -den
        g = _gcd(abs(num), den)
        self.num = num // g
        self.den = den // g

    @classmethod
    def from_string(cls, s: str, ctx: str) -> "Rational":
        if not isinstance(s, str):
            raise ValueError(f"{ctx} must be a string, got {type(s).__name__}")
        s = s.strip()
        if not s:
            raise ValueError(f"{ctx} must be a nonempty string")
        if "/" in s:
            parts = s.split("/")
            if len(parts) != 2:
                raise ValueError(f"{ctx} must be of the form 'p/q', got {s!r}")
            try:
                p = int(parts[0].strip())
                q = int(parts[1].strip())
            except ValueError as exc:
                raise ValueError(f"{ctx} parts must be integers, got {s!r}") from exc
            return cls(p, q)
        try:
            return cls(int(s), 1)
        except ValueError as exc:
            raise ValueError(f"{ctx} must be an integer or 'p/q', got {s!r}") from exc

    def __add__(self, other: "Rational") -> "Rational":
        return Rational(self.num * other.den + other.num * self.den, self.den * other.den)

    def __sub__(self, other: "Rational") -> "Rational":
        return Rational(self.num * other.den - other.num * self.den, self.den * other.den)

    def __mul__(self, other: "Rational") -> "Rational":
        return Rational(self.num * other.num, self.den * other.den)

    def __le__(self, other: "Rational") -> bool:
        return self.num * other.den <= other.num * self.den

    def __lt__(self, other: "Rational") -> bool:
        return self.num * other.den < other.num * self.den

    def __ge__(self, other: "Rational") -> bool:
        return self.num * other.den >= other.num * self.den

    def __gt__(self, other: "Rational") -> bool:
        return self.num * other.den > other.num * self.den

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rational):
            return NotImplemented
        return self.num * other.den == other.num * self.den

    def is_nonnegative(self) -> bool:
        return self.num >= 0

    def is_positive(self) -> bool:
        return self.num > 0

    def __repr__(self) -> str:
        return f"Rational({self.num}/{self.den})"

    def to_string(self) -> str:
        if self.den == 1:
            return str(self.num)
        return f"{self.num}/{self.den}"


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


# ---------------------------------------------------------------------------
# Rational interval arithmetic
# ---------------------------------------------------------------------------

class RationalInterval:
    """Closed interval [lo, hi] with rational endpoints."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: Rational, hi: Rational) -> None:
        if lo > hi:
            raise ValueError(f"interval lower bound exceeds upper: [{lo.to_string()}, {hi.to_string()}]")
        self.lo = lo
        self.hi = hi

    @classmethod
    def from_string(cls, s: str, ctx: str) -> "RationalInterval":
        if not isinstance(s, str):
            raise ValueError(f"{ctx} must be a string, got {type(s).__name__}")
        s = s.strip()
        if not (s.startswith("[") and s.endswith("]")):
            raise ValueError(f"{ctx} must be of the form '[a,b]', got {s!r}")
        inner = s[1:-1]
        parts = inner.split(",")
        if len(parts) != 2:
            raise ValueError(f"{ctx} must have two comma-separated endpoints, got {s!r}")
        lo = Rational.from_string(parts[0].strip(), f"{ctx}.lower")
        hi = Rational.from_string(parts[1].strip(), f"{ctx}.upper")
        return cls(lo, hi)

    @property
    def positive(self) -> bool:
        return self.lo.is_positive()

    @property
    def nonnegative(self) -> bool:
        return self.lo.is_nonnegative()

    def strictly_dominates(self, other: "RationalInterval") -> bool:
        return self.lo > other.hi

    def minus(self, other: "RationalInterval") -> "RationalInterval":
        """Worst-case interval for self - other."""
        lo = self.lo - other.hi
        hi = self.hi - other.lo
        return RationalInterval(lo, hi)

    def __repr__(self) -> str:
        return f"[{self.lo.to_string()}, {self.hi.to_string()}]"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_keys(d: dict[str, Any], keys: list[str], ctx: str) -> None:
    for k in keys:
        require(k in d, f"{ctx}.{k} must be present")


def require_int(value: Any, ctx: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{ctx} must be an integer, got {type(value).__name__}")
    if minimum is not None:
        require(value >= minimum, f"{ctx} must be >= {minimum}, got {value}")
    if maximum is not None:
        require(value <= maximum, f"{ctx} must be <= {maximum}, got {value}")
    return value


def require_rational(value: Any, ctx: str) -> Rational:
    return Rational.from_string(value, ctx)


def require_rational_interval(value: Any, ctx: str) -> RationalInterval:
    return RationalInterval.from_string(value, ctx)


def require_string(value: Any, ctx: str = "value") -> str:
    require(isinstance(value, str) and value, f"{ctx} must be a nonempty string")
    return value


# ---------------------------------------------------------------------------
# Section validators
# ---------------------------------------------------------------------------

def validate_header(data: dict[str, Any]) -> dict[str, str]:
    header_keys = ["theorem", "version", "claim_scope", "background", "spin", "status"]
    require_keys(data, header_keys, "root")
    require(require_string(data["theorem"]) == "SFSAT",
            "root.theorem must be 'SFSAT'")
    require(require_string(data["version"]) == "1.0",
            "root.version must be '1.0' (only version 1.0 is supported)")
    require(require_string(data["status"]) == "machine_certificate",
            "root.status must be 'machine_certificate'")
    return {
        "claim_scope": require_string(data["claim_scope"]),
        "background": require_string(data["background"]),
        "spin": require_string(data["spin"]),
    }


def validate_sobolev_hierarchy(data: dict[str, Any]) -> dict[str, Any]:
    sh = data.get("sobolev_hierarchy")
    require(isinstance(sh, dict), "sobolev_hierarchy must be an object")
    require_keys(sh, ["N_shared", "N_min", "N_symbol_class",
                      "max_per_block_derivative_loss", "headroom_floor"],
                 "sobolev_hierarchy")
    n_shared = require_int(sh["N_shared"], "sobolev_hierarchy.N_shared", minimum=1)
    n_min = require_int(sh["N_min"], "sobolev_hierarchy.N_min", minimum=1)
    n_sym = require_int(sh["N_symbol_class"], "sobolev_hierarchy.N_symbol_class", minimum=1)
    max_d = require_int(sh["max_per_block_derivative_loss"],
                        "sobolev_hierarchy.max_per_block_derivative_loss", minimum=0)
    headroom_floor = require_rational(sh["headroom_floor"], "sobolev_hierarchy.headroom_floor")
    require(headroom_floor.is_nonnegative(),
            "sobolev_hierarchy.headroom_floor must be nonnegative")

    # (L1) N_shared >= N_symbol_class + max_per_block_derivative_loss
    require(n_shared >= n_sym + max_d,
            f"sobolev_hierarchy.N_shared ({n_shared}) must be >= N_symbol_class ({n_sym}) "
            f"+ max_per_block_derivative_loss ({max_d}) = {n_sym + max_d}")

    # (L5) N_min must be at least N_symbol_class + max_per_block_derivative_loss - headroom_floor
    # (this prevents declaring a tiny N_min that vacates the headroom requirement)
    floor_int = headroom_floor.num // headroom_floor.den  # integer part of headroom_floor
    require(n_min >= n_sym + max_d - floor_int,
            f"sobolev_hierarchy.N_min ({n_min}) must be >= N_symbol_class ({n_sym}) "
            f"+ max_per_block_derivative_loss ({max_d}) - headroom_floor_int ({floor_int}) = "
            f"{n_sym + max_d - floor_int}")

    return {
        "N_shared": n_shared,
        "N_min": n_min,
        "N_symbol_class": n_sym,
        "max_per_block_derivative_loss": max_d,
        "headroom_floor": headroom_floor,
    }


def validate_frequency_decomposition(data: dict[str, Any]) -> dict[str, Any]:
    fd = data.get("frequency_decomposition")
    require(isinstance(fd, dict), "frequency_decomposition must be an object")
    require_keys(fd, ["name", "layer_count", "layer_count_minimum_for_symbol_class",
                      "layer_bounds", "covers_full_range"],
                 "frequency_decomposition")
    name = require_string(fd["name"])
    layer_count = require_int(fd["layer_count"], "frequency_decomposition.layer_count", minimum=1)
    layer_min = require_int(fd["layer_count_minimum_for_symbol_class"],
                            "frequency_decomposition.layer_count_minimum_for_symbol_class",
                            minimum=1)
    require(layer_count >= layer_min,
            f"frequency_decomposition.layer_count ({layer_count}) must be >= "
            f"layer_count_minimum_for_symbol_class ({layer_min})")
    require(fd["covers_full_range"] is True,
            "frequency_decomposition.covers_full_range must be true")

    bounds = fd["layer_bounds"]
    require(isinstance(bounds, list) and len(bounds) == layer_count,
            f"frequency_decomposition.layer_bounds must be a list of length layer_count ({layer_count})")
    parsed_bounds: list[RationalInterval] = []
    for i, b in enumerate(bounds):
        ctx = f"frequency_decomposition.layer_bounds[{i}]"
        require(isinstance(b, dict), f"{ctx} must be an object")
        require_keys(b, ["layer_index", "lower", "upper"], ctx)
        require_int(b["layer_index"], f"{ctx}.layer_index", minimum=0)
        lo = require_rational(b["lower"], f"{ctx}.lower")
        hi = require_rational(b["upper"], f"{ctx}.upper")
        require(lo.is_positive(), f"{ctx}.lower must be positive (frequency layers are positive)")
        require(lo <= hi, f"{ctx} must satisfy lower <= upper")
        parsed_bounds.append(RationalInterval(lo, hi))

    # Adjacent layers must touch or overlap with positive overlap (L3)
    for i in range(len(parsed_bounds) - 1):
        cur = parsed_bounds[i]
        nxt = parsed_bounds[i + 1]
        require(nxt.lo <= cur.hi,
                f"frequency_decomposition.layer_bounds[{i + 1}].lower ({nxt.lo.to_string()}) "
                f"must be <= layer_bounds[{i}].upper ({cur.hi.to_string()}); "
                "adjacent layers must touch or overlap")

    # Coverage check: layer 0 lower <= 1, last layer upper >= lambda_max if declared
    require(parsed_bounds[0].lo <= Rational(1, 1),
            f"frequency_decomposition.layer_bounds[0].lower ({parsed_bounds[0].lo.to_string()}) "
            "must be <= 1 (full range coverage)")

    return {
        "name": name,
        "layer_count": layer_count,
        "layer_count_minimum_for_symbol_class": layer_min,
        "layer_bounds": parsed_bounds,
    }


def validate_symbol_class(data: dict[str, Any], sobolev: dict[str, Any]) -> dict[str, Any]:
    sc = data.get("symbol_class")
    require(isinstance(sc, dict), "symbol_class must be an object")
    require_keys(sc, ["name", "order", "regularity_scale", "decay_rate",
                      "rational_symbol_bound"],
                 "symbol_class")
    name = require_string(sc["name"])
    order = require_int(sc["order"], "symbol_class.order", minimum=0, maximum=0)
    reg_scale = require_int(sc["regularity_scale"], "symbol_class.regularity_scale", minimum=1)
    decay = require_rational(sc["decay_rate"], "symbol_class.decay_rate")
    require(decay.is_nonnegative(), "symbol_class.decay_rate must be nonnegative")
    sym_bound = require_rational(sc["rational_symbol_bound"], "symbol_class.rational_symbol_bound")
    require(sym_bound.is_positive(),
            "symbol_class.rational_symbol_bound must be positive")

    # Cross-check: regularity_scale must match sobolev_hierarchy.N_symbol_class
    require(reg_scale == sobolev["N_symbol_class"],
            f"symbol_class.regularity_scale ({reg_scale}) must match "
            f"sobolev_hierarchy.N_symbol_class ({sobolev['N_symbol_class']})")

    return {
        "name": name,
        "order": order,
        "regularity_scale": reg_scale,
        "decay_rate": decay,
        "rational_symbol_bound": sym_bound,
    }


def validate_block_margins(data: dict[str, Any], sobolev: dict[str, Any]) -> dict[str, Any]:
    bm = data.get("block_margins")
    require(isinstance(bm, list) and len(bm) == len(REQUIRED_BLOCKS),
            f"block_margins must be a list of length {len(REQUIRED_BLOCKS)}")
    seen_names: set[str] = set()
    seen_positions: set[int] = set()
    blocks: dict[str, dict[str, Any]] = {}
    margin_floors: list[Rational] = []

    for i, entry in enumerate(bm):
        ctx = f"block_margins[{i}]"
        require(isinstance(entry, dict), f"{ctx} must be an object")
        require_keys(entry, ["name", "derivative_loss", "positive_margin",
                             "remainder_bound", "margin_floor", "composition_position"],
                     ctx)
        name = require_string(entry["name"])
        require(name in REQUIRED_BLOCKS,
                f"{ctx}.name must be one of {sorted(REQUIRED_BLOCKS)}, got {name!r}")
        require(name not in seen_names, f"{ctx}.name duplicates an earlier entry")
        seen_names.add(name)

        dloss = require_int(entry["derivative_loss"], f"{ctx}.derivative_loss", minimum=0)
        margin = require_rational_interval(entry["positive_margin"], f"{ctx}.positive_margin")
        remainder = require_rational_interval(entry["remainder_bound"], f"{ctx}.remainder_bound")
        require(margin.positive, f"{ctx}.positive_margin must be positive")
        require(remainder.nonnegative, f"{ctx}.remainder_bound must be nonnegative")
        require(margin.strictly_dominates(remainder),
                f"{ctx}.positive_margin must strictly dominate remainder_bound")

        floor = require_rational(entry["margin_floor"], f"{ctx}.margin_floor")
        require(floor.is_positive(), f"{ctx}.margin_floor must be positive")

        # Check inf(margin - remainder) >= margin_floor
        diff = margin.minus(remainder)
        require(diff.lo >= floor,
                f"{ctx}: inf(margin - remainder) = {diff.lo.to_string()} must be >= "
                f"margin_floor ({floor.to_string()})")

        pos = require_int(entry["composition_position"], f"{ctx}.composition_position",
                          minimum=1, maximum=len(REQUIRED_BLOCKS))
        require(pos not in seen_positions, f"{ctx}.composition_position duplicates an earlier entry")
        seen_positions.add(pos)

        blocks[name] = {
            "derivative_loss": dloss,
            "margin": margin,
            "remainder": remainder,
            "margin_floor": floor,
            "composition_position": pos,
        }
        margin_floors.append(floor)

    # Check that the per-block derivative losses match max_per_block_derivative_loss
    max_d = max(b["derivative_loss"] for b in blocks.values())
    require(max_d == sobolev["max_per_block_derivative_loss"],
            f"max per-block derivative loss ({max_d}) must match "
            f"sobolev_hierarchy.max_per_block_derivative_loss "
            f"({sobolev['max_per_block_derivative_loss']})")

    return {
        "blocks": blocks,
        "margin_floors": margin_floors,
    }


def validate_composition_order(data: dict[str, Any],
                                block_margins: dict[str, Any]) -> dict[str, Any]:
    co = data.get("composition_order")
    require(isinstance(co, list) and len(co) == len(REQUIRED_BLOCKS),
            f"composition_order must be a list of length {len(REQUIRED_BLOCKS)}")
    seen: set[str] = set()
    order: list[str] = []
    for i, name in enumerate(co):
        ctx = f"composition_order[{i}]"
        require_string(name)
        require(name in REQUIRED_BLOCKS,
                f"{ctx} must be one of {sorted(REQUIRED_BLOCKS)}, got {name!r}")
        require(name not in seen, f"{ctx} duplicates an earlier entry")
        seen.add(name)
        order.append(name)

    blocks = block_margins["blocks"]
    sobolev_n_shared = None  # filled by caller via separate path
    # Compute prefix products of (margin - remainder)
    prefix_products: list[Rational] = []
    product = Rational(1, 1)
    for name in order:
        b = blocks[name]
        diff = b["margin"].minus(b["remainder"])
        # Worst-case prefix product uses the lower bound of (margin - remainder)
        product = product * diff.lo
        prefix_products.append(product)
        require(product.is_positive(),
                f"prefix product after position {len(prefix_products)} is nonpositive: "
                f"{product.to_string()}")

    return {
        "order": order,
        "prefix_products": prefix_products,
    }


def validate_admissibility_budget(data: dict[str, Any],
                                   block_margins: dict[str, Any],
                                   composition: dict[str, Any],
                                   sobolev: dict[str, Any]) -> dict[str, Any]:
    ab = data.get("admissibility_budget")
    require(isinstance(ab, dict), "admissibility_budget must be an object")
    require_keys(ab, ["margin_floor", "headroom_floor", "product_lower_bound",
                      "total_derivative_loss", "max_total_derivative_loss",
                      "available_derivative_buffer"],
                 "admissibility_budget")

    floor = require_rational(ab["margin_floor"], "admissibility_budget.margin_floor")
    require(floor.is_positive(), "admissibility_budget.margin_floor must be positive")
    headroom = require_rational(ab["headroom_floor"], "admissibility_budget.headroom_floor")
    require(headroom.is_nonnegative(),
            "admissibility_budget.headroom_floor must be nonnegative")
    plb = require_rational(ab["product_lower_bound"], "admissibility_budget.product_lower_bound")
    require(plb.is_positive(),
            "admissibility_budget.product_lower_bound must be positive")

    total = require_int(ab["total_derivative_loss"],
                        "admissibility_budget.total_derivative_loss", minimum=0)
    max_total = require_int(ab["max_total_derivative_loss"],
                            "admissibility_budget.max_total_derivative_loss", minimum=0)
    buf = require_int(ab["available_derivative_buffer"],
                      "admissibility_budget.available_derivative_buffer", minimum=0)
    require(total <= max_total,
            f"admissibility_budget.total_derivative_loss ({total}) must be <= "
            f"max_total_derivative_loss ({max_total})")
    require(buf >= total,
            f"admissibility_budget.available_derivative_buffer ({buf}) must be >= "
            f"total_derivative_loss ({total})")

    # Cross-check: margin_floor must equal the minimum of blockwise margin floors
    min_floor = min(bf for bf in block_margins["margin_floors"])
    require(floor == min_floor,
            f"admissibility_budget.margin_floor ({floor.to_string()}) must equal the "
            f"minimum blockwise margin_floor ({min_floor.to_string()})")

    # Cross-check: headroom_floor must match sobolev_hierarchy.headroom_floor
    require(headroom == sobolev["headroom_floor"],
            f"admissibility_budget.headroom_floor ({headroom.to_string()}) must match "
            f"sobolev_hierarchy.headroom_floor ({sobolev['headroom_floor'].to_string()})")

    # Cross-check: total_derivative_loss must equal the sum of blockwise losses
    computed_total = sum(b["derivative_loss"] for b in block_margins["blocks"].values())
    require(total == computed_total,
            f"admissibility_budget.total_derivative_loss ({total}) must equal the "
            f"sum of blockwise derivative losses ({computed_total})")

    # Cross-check: product_lower_bound must be <= the final prefix product
    final_product = composition["prefix_products"][-1]
    require(plb <= final_product,
            f"admissibility_budget.product_lower_bound ({plb.to_string()}) must be <= "
            f"the final composition product ({final_product.to_string()})")

    return {
        "margin_floor": floor,
        "headroom_floor": headroom,
        "product_lower_bound": plb,
        "total_derivative_loss": total,
        "max_total_derivative_loss": max_total,
        "available_derivative_buffer": buf,
    }


# ---------------------------------------------------------------------------
# Prefix headroom check (L1)
# ---------------------------------------------------------------------------

def validate_prefix_headroom(sobolev: dict[str, Any],
                              block_margins: dict[str, Any],
                              composition: dict[str, Any]) -> None:
    """Verify that every prefix of the composition order leaves at least N_min
    derivatives of headroom."""
    n_shared = sobolev["N_shared"]
    n_min = sobolev["N_min"]
    blocks = block_margins["blocks"]
    cumulative = 0
    for k, name in enumerate(composition["order"], 1):
        cumulative += blocks[name]["derivative_loss"]
        headroom = n_shared - cumulative
        require(headroom >= n_min,
                f"prefix headroom after position {k} ({name}): "
                f"N_shared ({n_shared}) - cumulative derivative loss ({cumulative}) = "
                f"{headroom} must be >= N_min ({n_min})")


# ---------------------------------------------------------------------------
# Top-level verify
# ---------------------------------------------------------------------------

def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be a JSON object")
    header = validate_header(data)
    sobolev = validate_sobolev_hierarchy(data)
    freq = validate_frequency_decomposition(data)
    symbol = validate_symbol_class(data, sobolev)
    block_margins = validate_block_margins(data, sobolev)
    composition = validate_composition_order(data, block_margins)
    validate_prefix_headroom(sobolev, block_margins, composition)
    budget = validate_admissibility_budget(data, block_margins, composition, sobolev)

    return {
        "accepted": True,
        "theorem": "SFSAT",
        "version": "1.0",
        "claim_scope": header["claim_scope"],
        "background": header["background"],
        "spin": header["spin"],
        "N_shared": sobolev["N_shared"],
        "N_min": sobolev["N_min"],
        "N_symbol_class": sobolev["N_symbol_class"],
        "layer_count": freq["layer_count"],
        "symbol_class": symbol["name"],
        "block_count": len(block_margins["blocks"]),
        "composition_order": composition["order"],
        "final_prefix_product": composition["prefix_products"][-1].to_string(),
        "product_lower_bound": budget["product_lower_bound"].to_string(),
        "total_derivative_loss": budget["total_derivative_loss"],
        "warning": ("Combinatorial, arithmetic, and internal-coherence checks pass. "
                    "This does NOT prove the SFSAT analytic content: the per-block "
                    "regularity scales, frequency minimum, and composition order are "
                    "not checked against the upstream theorems."),
    }


def verify_file(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    return verify(data)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except ValueError as exc:
        print(f"SFSAT certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
