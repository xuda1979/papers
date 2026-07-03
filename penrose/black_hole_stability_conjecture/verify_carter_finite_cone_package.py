#!/usr/bin/env python3
"""Validate a finite Carter cone assembly package.

The checker is a soundness guard for the finite-cover reduction in the paper.
It verifies that every referenced local chart certificate is independently
valid, that the quadratic partition is exact at the recorded level, and that
the certified assembly errors are strictly smaller than the local margin slack.
It does not prove that the local certificates are Kerr certificates; it proves
only that a proposed finite package has enough rational margin to be patched
without losing positivity.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_chart_certificate as chart_certificate
from verify_carter_transport_certificate import Interval, parse_interval


ERROR_FIELDS = [
    "transport_defect_bound",
    "overlap_mismatch_bound",
    "normal_extension_error_bound",
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


def resolve_path(raw_path: Any, base_dir: Path | None) -> Path:
    require(isinstance(raw_path, str) and raw_path.strip(), "certificate_path must be a nonempty string")
    path = Path(raw_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def interval_is_zero(interval: Interval) -> bool:
    return interval.lo == 0 and interval.hi == 0


def format_fraction(value: Fraction) -> str:
    return str(value)


def validate_metadata(data: dict[str, Any]) -> None:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "status", "claim_scope"], "metadata")
    require(
        metadata["claim_scope"] == "finite_carter_cone_assembly",
        "metadata.claim_scope must be finite_carter_cone_assembly",
    )


def validate_partition(data: dict[str, Any]) -> Fraction:
    partition = data.get("quadratic_partition")
    require(isinstance(partition, dict), "quadratic_partition must be an object")
    require_keys(partition, ["type", "sum_square_residual"] + ERROR_FIELDS, "quadratic_partition")
    require(
        partition["type"] == "transport_tame_quadratic",
        "quadratic_partition.type must be transport_tame_quadratic",
    )
    residual = parse_interval(partition["sum_square_residual"], "quadratic_partition.sum_square_residual")
    require(interval_is_zero(residual), "quadratic_partition.sum_square_residual must be [0, 0]")

    total_error = Fraction(0)
    for field in ERROR_FIELDS:
        error = parse_interval(partition[field], f"quadratic_partition.{field}")
        require(error.nonnegative, f"quadratic_partition.{field} must be nonnegative")
        total_error += error.hi
    return total_error


def local_slack(certificate: dict[str, Any], path: str) -> Fraction:
    margins = certificate.get("margins")
    require(isinstance(margins, dict), f"{path}.margins must be an object")
    require_keys(margins, ["remainder_bound", "positive_margin"], f"{path}.margins")
    remainder = parse_interval(margins["remainder_bound"], f"{path}.margins.remainder_bound")
    positive = parse_interval(margins["positive_margin"], f"{path}.margins.positive_margin")
    require(remainder.nonnegative, f"{path}.margins.remainder_bound must be nonnegative")
    require(positive.positive, f"{path}.margins.positive_margin must be positive")
    require(positive.dominates(remainder), f"{path}.margins.positive_margin must dominate remainder_bound")
    return positive.lo - remainder.hi


def validate_local_certificates(data: dict[str, Any], base_dir: Path | None) -> tuple[list[str], Fraction, Fraction | None]:
    locals_raw = data.get("local_cone_certificates")
    require(isinstance(locals_raw, list) and locals_raw, "local_cone_certificates must be a nonempty list")
    checked_charts: list[str] = []
    minimum_slack: Fraction | None = None
    minimum_coercivity_delta: Fraction | None = None
    seen: set[str] = set()

    for index, item in enumerate(locals_raw):
        path = f"local_cone_certificates[{index}]"
        require(isinstance(item, dict), f"{path} must be an object")
        require_keys(item, ["chart", "certificate_path"], path)
        chart = item["chart"]
        require(isinstance(chart, str) and chart.strip(), f"{path}.chart must be a nonempty string")
        require(chart not in seen, f"{path}.chart duplicates an earlier local certificate")
        seen.add(chart)

        certificate_path = resolve_path(item["certificate_path"], base_dir)
        certificate = load_json(certificate_path)
        result = chart_certificate.verify(certificate, base_dir=certificate_path.parent)
        require(result["chart"] == chart, f"{path}.chart must match referenced certificate chart")
        slack = local_slack(certificate, str(certificate_path))
        require(slack > 0, f"{path} has no positive local margin slack")
        minimum_slack = slack if minimum_slack is None else min(minimum_slack, slack)
        if "coercivity_delta" in result:
            delta = Fraction(result["coercivity_delta"])
            minimum_coercivity_delta = delta if minimum_coercivity_delta is None else min(minimum_coercivity_delta, delta)
        checked_charts.append(chart)

    require(minimum_slack is not None, "no local margin slack was computed")
    return checked_charts, minimum_slack, minimum_coercivity_delta


def validate_assembly(data: dict[str, Any]) -> tuple[Fraction, Fraction]:
    assembly = data.get("assembly")
    require(isinstance(assembly, dict), "assembly must be an object")
    require_keys(assembly, ["global_delta", "quantization_loss_bound"], "assembly")
    delta = parse_interval(assembly["global_delta"], "assembly.global_delta")
    require(delta.positive, "assembly.global_delta must be positive")
    quantization = parse_interval(assembly["quantization_loss_bound"], "assembly.quantization_loss_bound")
    require(quantization.nonnegative, "assembly.quantization_loss_bound must be nonnegative")
    return delta.lo, quantization.hi


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "package must be an object")
    validate_metadata(data)
    partition_error = validate_partition(data)
    checked_charts, minimum_slack, minimum_coercivity_delta = validate_local_certificates(data, base_dir)
    delta, quantization_loss = validate_assembly(data)
    if minimum_coercivity_delta is not None:
        require(
            delta <= minimum_coercivity_delta,
            "assembly.global_delta must not exceed minimum local coercivity delta",
        )
    total_error = partition_error + quantization_loss
    require(
        minimum_slack > total_error,
        "finite assembly errors must be strictly smaller than the minimum local margin slack",
    )
    result = {
        "accepted": True,
        "claim_scope": data["metadata"]["claim_scope"],
        "checked_charts": checked_charts,
        "global_delta_lower": format_fraction(delta),
        "minimum_local_slack": format_fraction(minimum_slack),
        "total_assembly_error": format_fraction(total_error),
    }
    if minimum_coercivity_delta is not None:
        result["minimum_coercivity_delta"] = format_fraction(minimum_coercivity_delta)
    return result


def verify_file(path: Path) -> dict[str, Any]:
    return verify(load_json(path), base_dir=path.parent)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.package)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"finite Carter cone package rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
