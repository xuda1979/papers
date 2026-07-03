#!/usr/bin/env python3
"""Validate a master Carter certificate package.

The master checker connects the finite cone assembly checker to the
Carter-Wick finite-margin checker. If it accepts, the finite package already
has valid local chart certificates and enough assembly slack, while the Wick
package supplies a quantization-loss bound small enough to fit inside the
finite package's recorded quantization allowance.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_finite_cone_package as finite_cone
import verify_carter_wick_margin_certificate as wick_margin
from verify_carter_transport_certificate import parse_interval


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


def format_fraction(value: Fraction) -> str:
    return str(value)


def validate_metadata(data: dict[str, Any]) -> None:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "claim_scope", "status"], "metadata")
    require(metadata["claim_scope"] == "carter_master_certificate", "metadata.claim_scope must be carter_master_certificate")


def finite_quantization_bound(finite_package: dict[str, Any]) -> Fraction:
    assembly = finite_package.get("assembly")
    require(isinstance(assembly, dict), "finite package assembly must be an object")
    require("quantization_loss_bound" in assembly, "finite package assembly.quantization_loss_bound is missing")
    interval = parse_interval(assembly["quantization_loss_bound"], "finite_package.assembly.quantization_loss_bound")
    require(interval.nonnegative, "finite package quantization_loss_bound must be nonnegative")
    return interval.hi


def validate_compatibility(data: dict[str, Any], derivative_loss: int) -> int:
    compatibility = data.get("compatibility")
    require(isinstance(compatibility, dict), "compatibility must be an object")
    require_keys(compatibility, ["derivative_loss_budget", "same_symbol_class"], "compatibility")
    require(isinstance(compatibility["derivative_loss_budget"], int), "compatibility.derivative_loss_budget must be an integer")
    require(compatibility["derivative_loss_budget"] >= derivative_loss, "derivative_loss_budget must dominate Wick derivative loss")
    require(compatibility["same_symbol_class"] is True, "compatibility.same_symbol_class must be true")
    return compatibility["derivative_loss_budget"]


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "master certificate must be an object")
    validate_metadata(data)
    require_keys(data, ["finite_cone_package_path", "wick_margin_certificate_path", "compatibility"], "master certificate")

    finite_path = resolve_path(data["finite_cone_package_path"], base_dir, "finite_cone_package_path")
    wick_path = resolve_path(data["wick_margin_certificate_path"], base_dir, "wick_margin_certificate_path")

    finite_package = load_json(finite_path)
    wick_certificate = load_json(wick_path)
    finite_result = finite_cone.verify(finite_package, base_dir=finite_path.parent)
    wick_result = wick_margin.verify(wick_certificate)

    finite_q_bound = finite_quantization_bound(finite_package)
    wick_q_loss = Fraction(wick_result["exported_quantization_loss_upper"])
    require(finite_q_bound >= wick_q_loss, "finite package quantization_loss_bound must contain Wick exported loss")

    derivative_budget = validate_compatibility(data, wick_result["derivative_loss"])
    return {
        "accepted": True,
        "claim_scope": data["metadata"]["claim_scope"],
        "finite_checked_charts": finite_result["checked_charts"],
        "finite_total_assembly_error": finite_result["total_assembly_error"],
        "wick_exported_quantization_loss": format_fraction(wick_q_loss),
        "finite_quantization_loss_bound": format_fraction(finite_q_bound),
        "derivative_loss": wick_result["derivative_loss"],
        "derivative_loss_budget": derivative_budget,
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
        print(f"master Carter certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
