#!/usr/bin/env python3
"""Validate tame Carter orbit-cohomology certificates.

The orbit-cohomology checker proves that nonresonant Fourier modes are exact
coboundaries.  This checker adds the analytic tameness layer needed for a
pseudodifferential estimate: Fourier division must not create uncontrolled
small-denominator or derivative losses.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_orbit_cohomology_certificate as orbit
from verify_carter_transport_certificate import parse_interval


CLAIM_SCOPE = "carter_tame_orbit_cohomology_certificate"
ALLOWED_DERIVATION_STATUSES = {
    "candidate_tame_orbit_cohomology",
    "kerr_gravitational_proved",
}


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


def parse_fraction(raw: Any, path: str) -> Fraction:
    try:
        return Fraction(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"{path} must be a rational number") from exc


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
            "metadata.derivation_status kerr_gravitational_proved requires exact "
            "Kerr Fourier data, a complete torus atlas, and verified analytic "
            "symbol seminorm bounds; this checker verifies only the tame "
            "orbit-cohomology reduction"
        )
    return str(derivation_status)


def validate_tame_budget(data: dict[str, Any]) -> dict[str, Any]:
    budget = data.get("tame_budget")
    require(isinstance(budget, dict), "tame_budget must be an object")
    require_keys(
        budget,
        [
            "max_derivative_order",
            "nonresonant_denominator_floor",
            "escape_derivative_l1_bound",
            "derivative_loss_rate",
            "required_tame_margin",
        ],
        "tame_budget",
    )
    derivative_order = budget["max_derivative_order"]
    require(isinstance(derivative_order, int) and derivative_order >= 0, "tame_budget.max_derivative_order must be a nonnegative integer")
    denominator_floor = parse_fraction(
        budget["nonresonant_denominator_floor"],
        "tame_budget.nonresonant_denominator_floor",
    )
    require(denominator_floor > 0, "tame_budget.nonresonant_denominator_floor must be positive")
    escape_bound = parse_interval(budget["escape_derivative_l1_bound"], "tame_budget.escape_derivative_l1_bound")
    require(escape_bound.nonnegative, "tame_budget.escape_derivative_l1_bound must be nonnegative")
    derivative_loss_rate = parse_interval(budget["derivative_loss_rate"], "tame_budget.derivative_loss_rate")
    require(derivative_loss_rate.nonnegative, "tame_budget.derivative_loss_rate must be nonnegative")
    required_margin = parse_interval(budget["required_tame_margin"], "tame_budget.required_tame_margin")
    require(required_margin.positive, "tame_budget.required_tame_margin must be positive")
    return {
        "derivative_order": derivative_order,
        "denominator_floor": denominator_floor,
        "escape_bound_upper": escape_bound.hi,
        "derivative_loss_rate_upper": derivative_loss_rate.hi,
        "required_margin_lower": required_margin.lo,
    }


def mode_weight(mode: tuple[int, ...], derivative_order: int) -> Fraction:
    return Fraction(1 + sum(abs(entry) for entry in mode)) ** derivative_order


def chart_by_name(base_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for chart in base_result["chart_results"]:
        result[chart["chart"]] = chart
    return result


def validate_chart_tameness(
    raw_chart: dict[str, Any],
    base_chart: dict[str, Any],
    budget: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    path = f"torus_charts[{index}]"
    raw_frequency = raw_chart["frequency_vector"]
    frequency = tuple(orbit.parse_fraction(item, f"{path}.frequency_vector[{idx}]") for idx, item in enumerate(raw_frequency))
    derivative_order = budget["derivative_order"]
    denominator_floor = budget["denominator_floor"]
    derivative_loss_rate = budget["derivative_loss_rate_upper"]
    escape_derivative_l1 = Fraction(0)
    minimum_denominator: Fraction | None = None
    checked_modes: list[list[int]] = []
    for mode_index, item in enumerate(raw_chart["coboundary_modes"]):
        item_path = f"{path}.coboundary_modes[{mode_index}]"
        mode = orbit.parse_mode(item["mode"], len(frequency), f"{item_path}.mode")
        denominator = abs(orbit.dot_mode_frequency(mode, frequency))
        require(denominator >= denominator_floor, f"{item_path}.mode denominator {denominator} is below floor {denominator_floor}")
        escape_coefficient = orbit.parse_complex(item["escape_coefficient"], f"{item_path}.escape_coefficient")
        escape_derivative_l1 += mode_weight(mode, derivative_order) * escape_coefficient.l1_abs_bound()
        minimum_denominator = denominator if minimum_denominator is None else min(minimum_denominator, denominator)
        checked_modes.append(list(mode))
    require(escape_derivative_l1 <= budget["escape_bound_upper"], f"{path} escape derivative seminorm exceeds tame bound")
    base_margin = Fraction(base_chart["net_margin"])
    derivative_loss = escape_derivative_l1 * derivative_loss_rate
    tame_margin = base_margin - derivative_loss
    require(tame_margin > 0, f"{path} tame margin must be positive")
    require(
        tame_margin >= budget["required_margin_lower"],
        f"{path} tame margin {tame_margin} is below required {budget['required_margin_lower']}",
    )
    return {
        "chart": raw_chart["chart"],
        "checked_coboundary_modes": checked_modes,
        "minimum_nonresonant_denominator": str(minimum_denominator if minimum_denominator is not None else 0),
        "escape_derivative_l1": str(escape_derivative_l1),
        "base_net_margin": str(base_margin),
        "derivative_loss": str(derivative_loss),
        "tame_margin": str(tame_margin),
    }


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    require_keys(data, ["base_orbit_cohomology_certificate_path", "tame_budget"], "certificate")
    budget = validate_tame_budget(data)
    base_path = resolve_path(
        data["base_orbit_cohomology_certificate_path"],
        base_dir,
        "base_orbit_cohomology_certificate_path",
    )
    base_data = load_json(base_path)
    base_result = orbit.verify(base_data)
    base_charts = chart_by_name(base_result)
    chart_results = []
    for index, raw_chart in enumerate(base_data["torus_charts"]):
        chart_name = raw_chart["chart"]
        require(chart_name in base_charts, f"base chart {chart_name!r} was not accepted by orbit checker")
        chart_results.append(validate_chart_tameness(raw_chart, base_charts[chart_name], budget, index))
    minimum_tame_margin = min(Fraction(chart["tame_margin"]) for chart in chart_results)
    maximum_escape_derivative_l1 = max(Fraction(chart["escape_derivative_l1"]) for chart in chart_results)
    minimum_denominator = min(Fraction(chart["minimum_nonresonant_denominator"]) for chart in chart_results)
    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "base_orbit_certificate": str(base_path),
        "checked_charts": [chart["chart"] for chart in chart_results],
        "chart_results": chart_results,
        "max_derivative_order": budget["derivative_order"],
        "minimum_nonresonant_denominator": str(minimum_denominator),
        "maximum_escape_derivative_l1": str(maximum_escape_derivative_l1),
        "minimum_tame_margin": str(minimum_tame_margin),
        "required_tame_margin": str(budget["required_margin_lower"]),
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
        print(f"Carter tame orbit-cohomology certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
