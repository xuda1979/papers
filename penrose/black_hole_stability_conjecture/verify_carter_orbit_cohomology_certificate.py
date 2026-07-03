#!/usr/bin/env python3
"""Validate Carter orbit-cohomology certificates.

This checker verifies the finite analytic reduction behind a trapped-torus
escape function.  On each torus chart, nonresonant Fourier modes must be exact
coboundaries for the linearized Hamilton flow.  Resonant modes are bounded from
below by a rational triangle certificate.  The remaining approximation,
partition, extension, and quantization losses must fit inside the exported
margin.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from verify_carter_transport_certificate import Interval, parse_interval


CLAIM_SCOPE = "carter_orbit_cohomology_certificate"
ALLOWED_DERIVATION_STATUSES = {
    "candidate_orbit_cohomology",
    "kerr_gravitational_proved",
}
LOSS_FIELDS = [
    "approximation_error",
    "partition_error",
    "normal_hyperbolic_extension_error",
    "quantization_error",
]
GLOBALIZATION_FIELDS = [
    "same_symbol_class",
    "compact_trapped_neighborhood",
    "real_escape_symbol",
    "partition_of_unity_exact",
    "normal_hyperbolic_extension_recorded",
]


@dataclass(frozen=True)
class ComplexFraction:
    real: Fraction
    imag: Fraction

    def conjugate(self) -> "ComplexFraction":
        return ComplexFraction(self.real, -self.imag)

    def l1_abs_bound(self) -> Fraction:
        return abs(self.real) + abs(self.imag)


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


def parse_fraction(raw: Any, path: str) -> Fraction:
    try:
        return Fraction(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"{path} must be a rational number") from exc


def parse_complex(raw: Any, path: str) -> ComplexFraction:
    require(isinstance(raw, list) and len(raw) == 2, f"{path} must be [real, imag]")
    return ComplexFraction(parse_fraction(raw[0], f"{path}[0]"), parse_fraction(raw[1], f"{path}[1]"))


def parse_mode(raw: Any, dimension: int, path: str) -> tuple[int, ...]:
    require(isinstance(raw, list) and len(raw) == dimension, f"{path} must be an integer vector of length {dimension}")
    mode = []
    for index, value in enumerate(raw):
        require(isinstance(value, int), f"{path}[{index}] must be an integer")
        mode.append(value)
    return tuple(mode)


def dot_mode_frequency(mode: tuple[int, ...], frequency: tuple[Fraction, ...]) -> Fraction:
    return sum(Fraction(k) * omega for k, omega in zip(mode, frequency))


def derivative_coefficient(mode_dot_frequency: Fraction, escape_coefficient: ComplexFraction) -> ComplexFraction:
    # H_omega(g_k e^{ik.theta}) = i(k.omega) g_k e^{ik.theta}.
    return ComplexFraction(
        -mode_dot_frequency * escape_coefficient.imag,
        mode_dot_frequency * escape_coefficient.real,
    )


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
            "metadata.derivation_status kerr_gravitational_proved requires an exact "
            "Kerr principal-symbol derivation and a complete torus atlas; this "
            "checker verifies only the orbit-cohomology reduction"
        )
    return str(derivation_status)


def validate_globalization(data: dict[str, Any]) -> Fraction:
    globalization = data.get("globalization")
    require(isinstance(globalization, dict), "globalization must be an object")
    require_keys(globalization, GLOBALIZATION_FIELDS + ["required_global_margin"], "globalization")
    for field in GLOBALIZATION_FIELDS:
        require(globalization[field] is True, f"globalization.{field} must be true")
    required = parse_interval(globalization["required_global_margin"], "globalization.required_global_margin")
    require(required.positive, "globalization.required_global_margin must be positive")
    return required.lo


def validate_symbol_class(data: dict[str, Any]) -> str:
    symbol_class = data.get("symbol_class")
    require(isinstance(symbol_class, dict), "symbol_class must be an object")
    require_keys(symbol_class, ["name", "max_derivative_order", "compact_phase_neighborhood"], "symbol_class")
    require(isinstance(symbol_class["name"], str) and symbol_class["name"].strip(), "symbol_class.name must be nonempty")
    require(
        isinstance(symbol_class["max_derivative_order"], int) and symbol_class["max_derivative_order"] >= 2,
        "symbol_class.max_derivative_order must be an integer >= 2",
    )
    require(symbol_class["compact_phase_neighborhood"] is True, "symbol_class.compact_phase_neighborhood must be true")
    return str(symbol_class["name"])


def validate_conjugate_symmetry(coefficients: dict[tuple[int, ...], ComplexFraction], path: str) -> None:
    for mode, coefficient in coefficients.items():
        opposite = tuple(-entry for entry in mode)
        require(opposite in coefficients, f"{path} is missing conjugate mode {list(opposite)}")
        require(coefficients[opposite] == coefficient.conjugate(), f"{path} mode {list(mode)} fails conjugate symmetry")


def validate_resonant_modes(
    raw_modes: Any,
    frequency: tuple[Fraction, ...],
    dimension: int,
    path: str,
) -> tuple[Fraction, list[list[int]], dict[tuple[int, ...], ComplexFraction]]:
    require(isinstance(raw_modes, list) and raw_modes, f"{path} must be a nonempty list")
    coefficients: dict[tuple[int, ...], ComplexFraction] = {}
    checked_modes: list[list[int]] = []
    constant = ComplexFraction(Fraction(0), Fraction(0))
    loss = Fraction(0)
    for index, item in enumerate(raw_modes):
        item_path = f"{path}[{index}]"
        require(isinstance(item, dict), f"{item_path} must be an object")
        require_keys(item, ["mode", "coefficient"], item_path)
        mode = parse_mode(item["mode"], dimension, f"{item_path}.mode")
        require(mode not in coefficients, f"{item_path}.mode duplicates an earlier resonant mode")
        coefficient = parse_complex(item["coefficient"], f"{item_path}.coefficient")
        require(dot_mode_frequency(mode, frequency) == 0, f"{item_path}.mode is not resonant")
        coefficients[mode] = coefficient
        checked_modes.append(list(mode))
        if all(entry == 0 for entry in mode):
            require(coefficient.imag == 0, f"{item_path}.coefficient constant mode must be real")
            constant = coefficient
        else:
            loss += coefficient.l1_abs_bound()
    require(tuple(0 for _ in range(dimension)) in coefficients, f"{path} must include the zero mode")
    validate_conjugate_symmetry(coefficients, path)
    lower = constant.real - loss
    return lower, checked_modes, coefficients


def validate_coboundary_modes(
    raw_modes: Any,
    frequency: tuple[Fraction, ...],
    dimension: int,
    resonant_coefficients: dict[tuple[int, ...], ComplexFraction],
    path: str,
) -> tuple[list[list[int]], dict[tuple[int, ...], ComplexFraction], dict[tuple[int, ...], ComplexFraction]]:
    require(isinstance(raw_modes, list), f"{path} must be a list")
    density_coefficients: dict[tuple[int, ...], ComplexFraction] = {}
    escape_coefficients: dict[tuple[int, ...], ComplexFraction] = {}
    checked_modes: list[list[int]] = []
    for index, item in enumerate(raw_modes):
        item_path = f"{path}[{index}]"
        require(isinstance(item, dict), f"{item_path} must be an object")
        require_keys(item, ["mode", "density_coefficient", "escape_coefficient"], item_path)
        mode = parse_mode(item["mode"], dimension, f"{item_path}.mode")
        require(mode not in resonant_coefficients, f"{item_path}.mode was already listed as resonant")
        require(mode not in density_coefficients, f"{item_path}.mode duplicates an earlier coboundary mode")
        mode_dot = dot_mode_frequency(mode, frequency)
        require(mode_dot != 0, f"{item_path}.mode is resonant and cannot be solved as a coboundary")
        density_coefficient = parse_complex(item["density_coefficient"], f"{item_path}.density_coefficient")
        escape_coefficient = parse_complex(item["escape_coefficient"], f"{item_path}.escape_coefficient")
        require(
            derivative_coefficient(mode_dot, escape_coefficient) == density_coefficient,
            f"{item_path}.escape_coefficient does not solve the cohomological equation",
        )
        density_coefficients[mode] = density_coefficient
        escape_coefficients[mode] = escape_coefficient
        checked_modes.append(list(mode))
    if density_coefficients:
        validate_conjugate_symmetry(density_coefficients, f"{path}.density_coefficients")
        validate_conjugate_symmetry(escape_coefficients, f"{path}.escape_coefficients")
    return checked_modes, density_coefficients, escape_coefficients


def validate_losses(chart: dict[str, Any], path: str) -> Fraction:
    total = Fraction(0)
    for field in LOSS_FIELDS:
        interval = parse_interval(chart.get(field), f"{path}.{field}")
        require(interval.nonnegative, f"{path}.{field} must be nonnegative")
        total += interval.hi
    return total


def validate_chart(chart: Any, index: int) -> dict[str, Any]:
    path = f"torus_charts[{index}]"
    require(isinstance(chart, dict), f"{path} must be an object")
    require_keys(
        chart,
        [
            "chart",
            "frequency_vector",
            "resonant_modes",
            "coboundary_modes",
            *LOSS_FIELDS,
        ],
        path,
    )
    chart_name = chart["chart"]
    require(isinstance(chart_name, str) and chart_name.strip(), f"{path}.chart must be nonempty")
    raw_frequency = chart["frequency_vector"]
    require(isinstance(raw_frequency, list) and raw_frequency, f"{path}.frequency_vector must be a nonempty list")
    frequency = tuple(parse_fraction(item, f"{path}.frequency_vector[{index}]") for index, item in enumerate(raw_frequency))
    require(any(item != 0 for item in frequency), f"{path}.frequency_vector must not vanish")
    dimension = len(frequency)
    resonant_lower, checked_resonant, resonant_coefficients = validate_resonant_modes(
        chart["resonant_modes"],
        frequency,
        dimension,
        f"{path}.resonant_modes",
    )
    checked_coboundary, _, _ = validate_coboundary_modes(
        chart["coboundary_modes"],
        frequency,
        dimension,
        resonant_coefficients,
        f"{path}.coboundary_modes",
    )
    loss_total = validate_losses(chart, path)
    net_margin = resonant_lower - loss_total
    require(net_margin > 0, f"{path} net margin must be positive")
    return {
        "chart": chart_name,
        "dimension": dimension,
        "frequency_vector": [str(item) for item in frequency],
        "resonant_lower_bound": str(resonant_lower),
        "loss_total": str(loss_total),
        "net_margin": str(net_margin),
        "checked_resonant_modes": checked_resonant,
        "checked_coboundary_modes": checked_coboundary,
    }


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    symbol_class = validate_symbol_class(data)
    required_margin = validate_globalization(data)
    charts = data.get("torus_charts")
    require(isinstance(charts, list) and charts, "torus_charts must be a nonempty list")
    chart_results = [validate_chart(chart, index) for index, chart in enumerate(charts)]
    minimum_margin = min(Fraction(result["net_margin"]) for result in chart_results)
    require(
        minimum_margin >= required_margin,
        f"minimum orbit-cohomology margin {minimum_margin} is below required {required_margin}",
    )
    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "symbol_class": symbol_class,
        "checked_charts": [result["chart"] for result in chart_results],
        "chart_results": chart_results,
        "minimum_orbit_cohomology_margin": str(minimum_margin),
        "required_global_margin": str(required_margin),
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
        print(f"Carter orbit-cohomology certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
