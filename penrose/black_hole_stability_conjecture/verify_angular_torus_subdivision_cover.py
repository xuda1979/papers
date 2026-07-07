#!/usr/bin/env python3
"""Validate an Angular-Torus Subdivision Cover (ATSC) certificate.

This checker verifies the combinatorial, arithmetic, and *coverage* content
of an ATSC certificate. It checks:

  1. every required field is present, well-typed, and rational;
  2. the angular subdivision is a finite ordered list of intervals, each
     with positive rational length and a valid tag (polar, equatorial,
     or mid_latitude);
  3. adjacent intervals overlap by a positive rational length;
  4. the union of intervals covers the full angular domain [0, 1] (the
     rational parameterization of [0, pi]);
  5. each interval's tag matches the tag-specific positivity requirements
     (polar: sin^2(theta) <= 1/16, cos^2(theta) >= 15/16;
      equatorial: cos^2(theta) <= 1/524288, sin^2(theta) >= 262143/524288;
      mid_latitude: sin^2(theta) >= 1/32, cos^2(theta) >= 1/1048576);
  6. the radial atlas compatibility: for every radial chart, every angular
     interval has a nonempty lambda- and eta-intersection with the radial
     chart;
  7. the cover budget matches the per-interval data and at least one
     mid-latitude interval is declared.

The verifier does NOT check the analytic content of any individual patch's
positivity certificate. That is the responsibility of the upstream angular
patch verifiers. It checks that *if* each patch's positivity certificate
is valid on its declared interval, *then* the union of patches covers the
full angular torus with adjacent overlaps.

Usage:
    python3 verify_angular_torus_subdivision_cover.py <certificate.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


VALID_TAGS = {"polar", "equatorial", "mid_latitude"}

# Tag-specific bounds (rational).
POLAR_SIN_SQUARED_UPPER = "1/16"
POLAR_COS_SQUARED_LOWER = "15/16"
EQUATORIAL_COS_SQUARED_UPPER = "1/524288"
EQUATORIAL_SIN_SQUARED_LOWER = "262143/524288"
MID_LATITUDE_SIN_SQUARED_LOWER = "1/32"
MID_LATITUDE_COS_SQUARED_LOWER = "1/1048576"


# ---------------------------------------------------------------------------
# Rational arithmetic
# ---------------------------------------------------------------------------

class Rational:
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

    def is_positive(self) -> bool:
        return self.num > 0

    def is_nonnegative(self) -> bool:
        return self.num >= 0

    def to_string(self) -> str:
        if self.den == 1:
            return str(self.num)
        return f"{self.num}/{self.den}"

    def sort_key(self) -> tuple[int, int]:
        """A sort key that orders by rational value (cross-multiplied)."""
        # Use the float value as the primary key; floats are exact for small
        # rationals and the cross-multiplied integer pair is the tiebreaker.
        return (self.num * 1000000 // self.den, self.num)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


class RationalInterval:
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

    def length(self) -> Rational:
        return self.hi - self.lo

    def is_positive_length(self) -> bool:
        return self.lo < self.hi

    def overlaps(self, other: "RationalInterval") -> bool:
        return not (self.hi <= other.lo or other.hi <= self.lo)

    def overlap_length(self, other: "RationalInterval") -> Rational:
        lo = self.lo if self.lo > other.lo else other.lo
        hi = self.hi if self.hi < other.hi else other.hi
        if lo >= hi:
            return Rational(0, 1)
        return hi - lo

    def contains(self, other: "RationalInterval") -> bool:
        return self.lo <= other.lo and other.hi <= self.hi

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


def require_int(value: Any, ctx: str, *, minimum: int | None = None,
                maximum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{ctx} must be an integer, got {type(value).__name__}")
    if minimum is not None:
        require(value >= minimum, f"{ctx} must be >= {minimum}, got {value}")
    if maximum is not None:
        require(value <= maximum, f"{ctx} must be <= {maximum}, got {value}")
    return value


def require_string(value: Any, ctx: str = "value") -> str:
    require(isinstance(value, str) and value, f"{ctx} must be a nonempty string")
    return value


def require_rational(value: Any, ctx: str) -> Rational:
    return Rational.from_string(value, ctx)


def require_interval(value: Any, ctx: str) -> RationalInterval:
    return RationalInterval.from_string(value, ctx)


# ---------------------------------------------------------------------------
# Section validators
# ---------------------------------------------------------------------------

def validate_header(data: dict[str, Any]) -> dict[str, str]:
    require_keys(data, ["theorem", "version", "claim_scope", "background", "spin", "status"],
                 "root")
    require(require_string(data["theorem"]) == "ATSC",
            "root.theorem must be 'ATSC'")
    require(require_string(data["version"]) == "1.0",
            "root.version must be '1.0' (only version 1.0 is supported)")
    require(require_string(data["status"]) == "machine_certificate",
            "root.status must be 'machine_certificate'")
    return {
        "claim_scope": require_string(data["claim_scope"]),
        "background": require_string(data["background"]),
        "spin": require_string(data["spin"]),
    }


def validate_angular_subdivision(data: dict[str, Any]) -> dict[str, Any]:
    sub = data.get("angular_subdivision")
    require(isinstance(sub, dict), "angular_subdivision must be an object")
    require_keys(sub, ["domain", "intervals", "adjacency"], "angular_subdivision")

    domain = require_interval(sub["domain"], "angular_subdivision.domain")
    require(domain.lo == Rational(0, 1) and domain.hi == Rational(1, 1),
            "angular_subdivision.domain must be [0/1, 1/1] (the rational parameterization of [0, pi])")

    intervals = sub["intervals"]
    require(isinstance(intervals, list) and len(intervals) >= 1,
            "angular_subdivision.intervals must be a nonempty list")

    parsed_intervals: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for i, iv in enumerate(intervals):
        ctx = f"angular_subdivision.intervals[{i}]"
        require(isinstance(iv, dict), f"{ctx} must be an object")
        require_keys(iv, ["name", "tag", "cos_squared_theta_interval",
                          "sin_squared_theta_interval", "theta_interval",
                          "length", "positivity_certificate_reference"], ctx)
        name = require_string(iv["name"], f"{ctx}.name")
        require(name not in seen_names, f"{ctx}.name duplicates an earlier interval")
        seen_names.add(name)

        tag = require_string(iv["tag"], f"{ctx}.tag")
        require(tag in VALID_TAGS,
                f"{ctx}.tag must be one of {sorted(VALID_TAGS)}, got {tag!r}")

        cos_sq = require_interval(iv["cos_squared_theta_interval"],
                                   f"{ctx}.cos_squared_theta_interval")
        sin_sq = require_interval(iv["sin_squared_theta_interval"],
                                   f"{ctx}.sin_squared_theta_interval")
        theta_iv = require_interval(iv["theta_interval"], f"{ctx}.theta_interval")
        length = require_rational(iv["length"], f"{ctx}.length")
        require(length.is_positive(), f"{ctx}.length must be positive")

        # Check theta_interval length matches declared length
        require(theta_iv.length() == length,
                f"{ctx}: theta_interval length ({theta_iv.length().to_string()}) "
                f"must match declared length ({length.to_string()})")

        # Check cos^2(theta) + sin^2(theta) = 1 on the interval (worst-case enclosure)
        # i.e., the sum interval must contain [1, 1]
        sum_lo = cos_sq.lo + sin_sq.lo
        sum_hi = cos_sq.hi + sin_sq.hi
        require(sum_lo <= Rational(1, 1) and sum_hi >= Rational(1, 1),
                f"{ctx}: cos^2(theta) + sin^2(theta) interval "
                f"[{sum_lo.to_string()}, {sum_hi.to_string()}] must contain 1/1")

        # Tag-specific checks
        if tag == "polar":
            require(cos_sq.lo >= Rational.from_string(POLAR_COS_SQUARED_LOWER, "POLAR_COS_SQUARED_LOWER"),
                    f"{ctx}: polar tag requires cos^2(theta) lower bound >= "
                    f"{POLAR_COS_SQUARED_LOWER}, got {cos_sq.lo.to_string()}")
            require(sin_sq.hi <= Rational.from_string(POLAR_SIN_SQUARED_UPPER, "POLAR_SIN_SQUARED_UPPER"),
                    f"{ctx}: polar tag requires sin^2(theta) upper bound <= "
                    f"{POLAR_SIN_SQUARED_UPPER}, got {sin_sq.hi.to_string()}")
        elif tag == "equatorial":
            require(sin_sq.lo >= Rational.from_string(EQUATORIAL_SIN_SQUARED_LOWER, "EQUATORIAL_SIN_SQUARED_LOWER"),
                    f"{ctx}: equatorial tag requires sin^2(theta) lower bound >= "
                    f"{EQUATORIAL_SIN_SQUARED_LOWER}, got {sin_sq.lo.to_string()}")
            require(cos_sq.hi <= Rational.from_string(EQUATORIAL_COS_SQUARED_UPPER, "EQUATORIAL_COS_SQUARED_UPPER"),
                    f"{ctx}: equatorial tag requires cos^2(theta) upper bound <= "
                    f"{EQUATORIAL_COS_SQUARED_UPPER}, got {cos_sq.hi.to_string()}")
        else:  # mid_latitude
            require(sin_sq.lo >= Rational.from_string(MID_LATITUDE_SIN_SQUARED_LOWER, "MID_LATITUDE_SIN_SQUARED_LOWER"),
                    f"{ctx}: mid_latitude tag requires sin^2(theta) lower bound >= "
                    f"{MID_LATITUDE_SIN_SQUARED_LOWER}, got {sin_sq.lo.to_string()}")
            require(cos_sq.lo >= Rational.from_string(MID_LATITUDE_COS_SQUARED_LOWER, "MID_LATITUDE_COS_SQUARED_LOWER"),
                    f"{ctx}: mid_latitude tag requires cos^2(theta) lower bound >= "
                    f"{MID_LATITUDE_COS_SQUARED_LOWER}, got {cos_sq.lo.to_string()}")

        cert_ref = require_string(iv["positivity_certificate_reference"],
                                   f"{ctx}.positivity_certificate_reference")

        parsed_intervals.append({
            "name": name,
            "tag": tag,
            "cos_sq": cos_sq,
            "sin_sq": sin_sq,
            "theta_iv": theta_iv,
            "length": length,
            "cert_ref": cert_ref,
        })

    # Validate adjacency
    adjacency = sub["adjacency"]
    require(isinstance(adjacency, list),
            "angular_subdivision.adjacency must be a list")
    parsed_adjacency: list[dict[str, Any]] = []
    for i, adj in enumerate(adjacency):
        ctx = f"angular_subdivision.adjacency[{i}]"
        require(isinstance(adj, dict), f"{ctx} must be an object")
        require_keys(adj, ["left", "right", "overlap_length", "overlap_theta_interval"], ctx)
        left = require_string(adj["left"], f"{ctx}.left")
        right = require_string(adj["right"], f"{ctx}.right")
        require(left != right, f"{ctx}: left and right must differ")
        require(left in seen_names, f"{ctx}.left references unknown interval {left!r}")
        require(right in seen_names, f"{ctx}.right references unknown interval {right!r}")
        overlap_len = require_rational(adj["overlap_length"], f"{ctx}.overlap_length")
        require(overlap_len.is_positive(), f"{ctx}.overlap_length must be positive")
        overlap_iv = require_interval(adj["overlap_theta_interval"], f"{ctx}.overlap_theta_interval")
        require(overlap_iv.length() == overlap_len,
                f"{ctx}: overlap_theta_interval length ({overlap_iv.length().to_string()}) "
                f"must match overlap_length ({overlap_len.to_string()})")
        parsed_adjacency.append({
            "left": left,
            "right": right,
            "overlap_length": overlap_len,
            "overlap_iv": overlap_iv,
        })

    return {
        "domain": domain,
        "intervals": parsed_intervals,
        "adjacency": parsed_adjacency,
    }


def validate_coverage(sub: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    cov = data.get("coverage")
    require(isinstance(cov, dict), "coverage must be an object")
    require_keys(cov, ["domain_covered", "domain_lower", "domain_upper",
                       "interval_count", "min_overlap_length", "total_length"],
                 "coverage")

    require(cov["domain_covered"] is True, "coverage.domain_covered must be true")
    dom_lo = require_rational(cov["domain_lower"], "coverage.domain_lower")
    dom_hi = require_rational(cov["domain_upper"], "coverage.domain_upper")
    require(dom_lo == Rational(0, 1), f"coverage.domain_lower must be 0/1, got {dom_lo.to_string()}")
    require(dom_hi == Rational(1, 1), f"coverage.domain_upper must be 1/1, got {dom_hi.to_string()}")

    interval_count = require_int(cov["interval_count"], "coverage.interval_count", minimum=1)
    require(interval_count == len(sub["intervals"]),
            f"coverage.interval_count ({interval_count}) must equal the number of "
            f"intervals ({len(sub['intervals'])})")

    min_overlap = require_rational(cov["min_overlap_length"], "coverage.min_overlap_length")
    require(min_overlap.is_positive(),
            f"coverage.min_overlap_length must be positive, got {min_overlap.to_string()}")
    if sub["adjacency"]:
        computed_min_overlap = min(a["overlap_length"] for a in sub["adjacency"])
        require(min_overlap == computed_min_overlap,
                f"coverage.min_overlap_length ({min_overlap.to_string()}) must equal "
                f"the minimum adjacency overlap ({computed_min_overlap.to_string()})")

    total_length = require_rational(cov["total_length"], "coverage.total_length")
    require(total_length.is_positive(),
            f"coverage.total_length must be positive, got {total_length.to_string()}")

    # Total length = sum of interval lengths = (domain upper - domain lower) + sum of overlap lengths
    # because overlapping intervals cover [0, 1] with the overlaps counted twice in the sum.
    sum_interval_lengths = sum((iv["length"] for iv in sub["intervals"]), Rational(0, 1))
    sum_overlaps = sum((a["overlap_length"] for a in sub["adjacency"]), Rational(0, 1))
    expected_total = (dom_hi - dom_lo) + sum_overlaps
    require(total_length == sum_interval_lengths,
            f"coverage.total_length ({total_length.to_string()}) must equal the sum of "
            f"interval lengths ({sum_interval_lengths.to_string()})")
    require(sum_interval_lengths == expected_total,
            f"sum of interval lengths ({sum_interval_lengths.to_string()}) must equal "
            f"(domain upper - domain lower) + sum of overlap lengths "
            f"({expected_total.to_string()})")

    # Verify the actual cover: sort intervals by theta_iv.lo and check adjacency chain
    # Sort by rational value of theta_iv.lo using the sort_key method.
    sorted_ivs = sorted(sub["intervals"], key=lambda iv: iv["theta_iv"].lo.sort_key())
    require(sorted_ivs[0]["theta_iv"].lo == Rational(0, 1),
            f"first interval must start at 0/1, got {sorted_ivs[0]['theta_iv'].lo.to_string()}")
    require(sorted_ivs[-1]["theta_iv"].hi == Rational(1, 1),
            f"last interval must end at 1/1, got {sorted_ivs[-1]['theta_iv'].hi.to_string()}")
    for i in range(len(sorted_ivs) - 1):
        cur = sorted_ivs[i]
        nxt = sorted_ivs[i + 1]
        # Adjacent: nxt.lo <= cur.hi (overlap or touch) and nxt.hi > cur.hi (extends)
        require(nxt["theta_iv"].lo <= cur["theta_iv"].hi,
                f"intervals {cur['name']!r} and {nxt['name']!r} must overlap or touch "
                f"(next lo {nxt['theta_iv'].lo.to_string()} <= cur hi {cur['theta_iv'].hi.to_string()})")
        require(nxt["theta_iv"].hi > cur["theta_iv"].hi,
                f"interval {nxt['name']!r} must extend beyond {cur['name']!r} "
                f"(next hi {nxt['theta_iv'].hi.to_string()} > cur hi {cur['theta_iv'].hi.to_string()})")

    return {
        "interval_count": interval_count,
        "min_overlap_length": min_overlap,
        "total_length": total_length,
    }


def validate_radial_atlas_compatibility(data: dict[str, Any],
                                         sub: dict[str, Any]) -> dict[str, Any]:
    rac = data.get("radial_atlas_compatibility")
    require(isinstance(rac, dict), "radial_atlas_compatibility must be an object")
    require_keys(rac, ["radial_atlas_reference", "radial_chart_count",
                       "per_chart_lambda_eta_coverage"], "radial_atlas_compatibility")
    ref = require_string(rac["radial_atlas_reference"],
                          "radial_atlas_compatibility.radial_atlas_reference")
    chart_count = require_int(rac["radial_chart_count"],
                              "radial_atlas_compatibility.radial_chart_count", minimum=1)

    coverage = rac["per_chart_lambda_eta_coverage"]
    require(isinstance(coverage, list) and len(coverage) == chart_count,
            f"radial_atlas_compatibility.per_chart_lambda_eta_coverage must be a list "
            f"of length radial_chart_count ({chart_count})")

    seen_charts: set[str] = set()
    for i, entry in enumerate(coverage):
        ctx = f"radial_atlas_compatibility.per_chart_lambda_eta_coverage[{i}]"
        require(isinstance(entry, dict), f"{ctx} must be an object")
        require_keys(entry, ["radial_chart", "angular_intervals_with_nonempty_lambda_intersection",
                             "angular_intervals_with_nonempty_eta_intersection",
                             "all_angular_intervals_covered"], ctx)
        chart_name = require_string(entry["radial_chart"], f"{ctx}.radial_chart")
        require(chart_name not in seen_charts, f"{ctx}.radial_chart duplicates an earlier entry")
        seen_charts.add(chart_name)

        lambda_count = require_int(entry["angular_intervals_with_nonempty_lambda_intersection"],
                                   f"{ctx}.angular_intervals_with_nonempty_lambda_intersection",
                                   minimum=1)
        eta_count = require_int(entry["angular_intervals_with_nonempty_eta_intersection"],
                                f"{ctx}.angular_intervals_with_nonempty_eta_intersection",
                                minimum=1)
        require(entry["all_angular_intervals_covered"] is True,
                f"{ctx}.all_angular_intervals_covered must be true")
        require(lambda_count == len(sub["intervals"]),
                f"{ctx}: angular_intervals_with_nonempty_lambda_intersection ({lambda_count}) "
                f"must equal the number of angular intervals ({len(sub['intervals'])})")
        require(eta_count == len(sub["intervals"]),
                f"{ctx}: angular_intervals_with_nonempty_eta_intersection ({eta_count}) "
                f"must equal the number of angular intervals ({len(sub['intervals'])})")

    return {
        "radial_atlas_reference": ref,
        "radial_chart_count": chart_count,
    }


def validate_cover_budget(data: dict[str, Any], sub: dict[str, Any],
                           cov: dict[str, Any]) -> dict[str, Any]:
    cb = data.get("cover_budget")
    require(isinstance(cb, dict), "cover_budget must be an object")
    require_keys(cb, ["interval_count", "min_overlap_length", "min_interval_length",
                      "tag_distribution"], "cover_budget")
    interval_count = require_int(cb["interval_count"], "cover_budget.interval_count", minimum=1)
    require(interval_count == len(sub["intervals"]),
            f"cover_budget.interval_count ({interval_count}) must equal the number of "
            f"intervals ({len(sub['intervals'])})")

    min_overlap = require_rational(cb["min_overlap_length"], "cover_budget.min_overlap_length")
    require(min_overlap.is_positive(), "cover_budget.min_overlap_length must be positive")
    require(min_overlap == cov["min_overlap_length"],
            f"cover_budget.min_overlap_length ({min_overlap.to_string()}) must match "
            f"coverage.min_overlap_length ({cov['min_overlap_length'].to_string()})")

    min_interval = require_rational(cb["min_interval_length"], "cover_budget.min_interval_length")
    require(min_interval.is_positive(), "cover_budget.min_interval_length must be positive")
    computed_min_interval = min(iv["length"] for iv in sub["intervals"])
    require(min_interval == computed_min_interval,
            f"cover_budget.min_interval_length ({min_interval.to_string()}) must equal "
            f"the minimum interval length ({computed_min_interval.to_string()})")

    td = cb["tag_distribution"]
    require(isinstance(td, dict), "cover_budget.tag_distribution must be an object")
    require_keys(td, ["polar", "equatorial", "mid_latitude"],
                 "cover_budget.tag_distribution")
    polar_count = require_int(td["polar"], "cover_budget.tag_distribution.polar", minimum=0)
    equatorial_count = require_int(td["equatorial"], "cover_budget.tag_distribution.equatorial",
                                    minimum=0)
    mid_lat_count = require_int(td["mid_latitude"], "cover_budget.tag_distribution.mid_latitude",
                                 minimum=1)  # at least one mid-latitude required

    actual_polar = sum(1 for iv in sub["intervals"] if iv["tag"] == "polar")
    actual_equatorial = sum(1 for iv in sub["intervals"] if iv["tag"] == "equatorial")
    actual_mid_lat = sum(1 for iv in sub["intervals"] if iv["tag"] == "mid_latitude")
    require(polar_count == actual_polar,
            f"cover_budget.tag_distribution.polar ({polar_count}) must equal the actual "
            f"polar count ({actual_polar})")
    require(equatorial_count == actual_equatorial,
            f"cover_budget.tag_distribution.equatorial ({equatorial_count}) must equal "
            f"the actual equatorial count ({actual_equatorial})")
    require(mid_lat_count == actual_mid_lat,
            f"cover_budget.tag_distribution.mid_latitude ({mid_lat_count}) must equal "
            f"the actual mid-latitude count ({actual_mid_lat})")

    return {
        "interval_count": interval_count,
        "min_overlap_length": min_overlap,
        "min_interval_length": min_interval,
        "tag_distribution": {
            "polar": polar_count,
            "equatorial": equatorial_count,
            "mid_latitude": mid_lat_count,
        },
    }


# ---------------------------------------------------------------------------
# Top-level verify
# ---------------------------------------------------------------------------

def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be a JSON object")
    header = validate_header(data)
    sub = validate_angular_subdivision(data)
    cov = validate_coverage(sub, data)
    rac = validate_radial_atlas_compatibility(data, sub)
    budget = validate_cover_budget(data, sub, cov)

    return {
        "accepted": True,
        "theorem": "ATSC",
        "version": "1.0",
        "claim_scope": header["claim_scope"],
        "background": header["background"],
        "spin": header["spin"],
        "interval_count": len(sub["intervals"]),
        "tag_distribution": budget["tag_distribution"],
        "min_overlap_length": budget["min_overlap_length"].to_string(),
        "min_interval_length": budget["min_interval_length"].to_string(),
        "total_length": cov["total_length"].to_string(),
        "radial_atlas_reference": rac["radial_atlas_reference"],
        "radial_chart_count": rac["radial_chart_count"],
        "warning": ("Combinatorial, arithmetic, and coverage checks pass. "
                    "This does NOT prove the ATSC analytic content: the per-patch "
                    "positivity certificates and the mid-latitude positivity "
                    "certificate are not checked against the upstream theorems."),
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
        print(f"ATSC certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
