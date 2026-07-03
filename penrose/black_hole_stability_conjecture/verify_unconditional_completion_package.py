#!/usr/bin/env python3
"""Validate an unconditional Kerr-stability completion package.

The checker is a guardrail for the roadmap in the paper. It verifies that a
claimed unconditional proof package supplies every remaining theorem/certificate
slot, that no slot is merely assumed or conjectural, and that all slots use the
same gauge, foliation, Sobolev hierarchy, frequency decomposition, and symbol
class. It cannot decide whether a cited theorem is true.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from verify_kerr_certificate import parse_interval


NORMALIZATION_KEYS = [
    "background",
    "gauge",
    "foliation",
    "sobolev_index",
    "weight",
    "frequency_decomposition",
    "symbol_class",
]

REQUIRED_BLOCKS = {
    "linear_to_nonlinear_estimates",
    "trapped_superradiant_morawetz",
    "curvature_to_metric_reconstruction",
    "global_gauge_modulation",
    "nonlinear_weak_null_hierarchy",
    "near_extremal_parameter_control",
    "compatibility_assembly",
}

FORBIDDEN_REFERENCE_WORDS = {
    "assume",
    "assumption",
    "conjecture",
    "open",
    "tbd",
    "todo",
    "unknown",
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


def validate_metadata(data: dict[str, Any]) -> None:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "spin", "claim_scope", "status"], "metadata")
    require(metadata["claim_scope"] == "fixed_subextremal_exterior", "claim_scope must be fixed_subextremal_exterior")
    require("proof protocol" in str(metadata["status"]).lower(), "metadata.status must describe a proof protocol")


def validate_normalization(data: dict[str, Any]) -> dict[str, Any]:
    normalization = data.get("normalization")
    require(isinstance(normalization, dict), "normalization must be an object")
    require_keys(normalization, NORMALIZATION_KEYS + ["exterior_region_only"], "normalization")
    require(normalization["exterior_region_only"] is True, "normalization.exterior_region_only must be true")
    require(isinstance(normalization["sobolev_index"], int) and normalization["sobolev_index"] >= 8, "normalization.sobolev_index must be an integer >= 8")
    for key in NORMALIZATION_KEYS:
        if key != "sobolev_index":
            require(isinstance(normalization[key], str) and normalization[key].strip(), f"normalization.{key} must be a nonempty string")
    return normalization


def validate_reference(text: Any, path: str) -> None:
    require(isinstance(text, str) and text.strip(), f"{path} must be a nonempty string")
    lowered = text.lower()
    for word in FORBIDDEN_REFERENCE_WORDS:
        require(word not in lowered, f"{path} must not cite an unresolved {word!r} item")


def validate_block(block: dict[str, Any], normalization: dict[str, Any], path: str) -> str:
    require_keys(
        block,
        [
            "name",
            "status",
            "reference",
            "theorem_type",
            "gauge",
            "foliation",
            "sobolev_index",
            "weight",
            "frequency_decomposition",
            "symbol_class",
            "derivative_loss",
            "positive_margin",
            "remainder_bound",
        ],
        path,
    )
    name = block["name"]
    require(isinstance(name, str) and name in REQUIRED_BLOCKS, f"{path}.name must be one of {sorted(REQUIRED_BLOCKS)}")
    require(block["status"] == "proved", f"{path}.status must be proved")
    validate_reference(block["reference"], f"{path}.reference")
    require(block["theorem_type"] in {"published_theorem", "preprint_theorem", "machine_certificate", "internal_theorem"}, f"{path}.theorem_type is unsupported")

    for key in ["gauge", "foliation", "sobolev_index", "weight", "frequency_decomposition", "symbol_class"]:
        require(block[key] == normalization[key], f"{path}.{key} must match normalization.{key}")

    require(isinstance(block["derivative_loss"], int) and block["derivative_loss"] >= 0, f"{path}.derivative_loss must be a nonnegative integer")
    margin = parse_interval(block["positive_margin"], f"{path}.positive_margin")
    remainder = parse_interval(block["remainder_bound"], f"{path}.remainder_bound")
    require(margin.positive, f"{path}.positive_margin must be positive")
    require(remainder.nonnegative, f"{path}.remainder_bound must be nonnegative")
    require(margin.strictly_dominates(remainder), f"{path}.positive_margin must dominate remainder_bound")
    return name


def validate_proof_blocks(data: dict[str, Any], normalization: dict[str, Any]) -> int:
    blocks = data.get("proof_blocks")
    require(isinstance(blocks, list) and blocks, "proof_blocks must be a nonempty list")
    seen: set[str] = set()
    total_loss = 0
    for index, block in enumerate(blocks):
        path = f"proof_blocks[{index}]"
        require(isinstance(block, dict), f"{path} must be an object")
        name = validate_block(block, normalization, path)
        require(name not in seen, f"{path}.name duplicates an earlier block")
        seen.add(name)
        total_loss += block["derivative_loss"]
    missing = REQUIRED_BLOCKS - seen
    require(not missing, f"proof_blocks missing required blocks {sorted(missing)}")
    return total_loss


def validate_near_extremal_policy(data: dict[str, Any]) -> None:
    policy = data.get("near_extremal_policy")
    require(isinstance(policy, dict), "near_extremal_policy must be an object")
    require_keys(policy, ["uniform_to_extremality_claimed", "constant_policy"], "near_extremal_policy")
    require(policy["uniform_to_extremality_claimed"] is False, "uniform-to-extremality requires a separate extremal-degeneration proof")
    require(
        policy["constant_policy"] == "constants may depend on fixed spin distance_to_extremality",
        "near_extremal_policy.constant_policy must allow fixed-spin constants",
    )


def validate_compatibility(data: dict[str, Any], total_loss: int) -> None:
    compatibility = data.get("compatibility")
    require(isinstance(compatibility, dict), "compatibility must be an object")
    require_keys(
        compatibility,
        [
            "all_blocks_same_normalization",
            "all_remainders_absorbed",
            "available_derivative_buffer",
            "max_total_derivative_loss",
            "final_state_identification",
            "continuation_criterion",
        ],
        "compatibility",
    )
    require(compatibility["all_blocks_same_normalization"] is True, "all proof blocks must use the same normalization")
    require(compatibility["all_remainders_absorbed"] is True, "all remainders must be absorbed")
    require(isinstance(compatibility["available_derivative_buffer"], int), "available_derivative_buffer must be an integer")
    require(isinstance(compatibility["max_total_derivative_loss"], int), "max_total_derivative_loss must be an integer")
    require(total_loss <= compatibility["max_total_derivative_loss"], "proof block derivative losses exceed max_total_derivative_loss")
    require(compatibility["available_derivative_buffer"] >= total_loss, "available_derivative_buffer must cover total derivative loss")
    require(compatibility["final_state_identification"] is True, "final_state_identification must be true")
    require(compatibility["continuation_criterion"] is True, "continuation_criterion must be true")


def verify(data: dict[str, Any]) -> dict[str, Any]:
    validate_metadata(data)
    normalization = validate_normalization(data)
    total_loss = validate_proof_blocks(data, normalization)
    validate_near_extremal_policy(data)
    validate_compatibility(data, total_loss)
    return {
        "accepted": True,
        "claim_scope": data["metadata"]["claim_scope"],
        "checked_blocks": sorted(REQUIRED_BLOCKS),
        "total_derivative_loss": total_loss,
    }


def audit(data: dict[str, Any]) -> dict[str, Any]:
    """Return all detectable completion-package failures without short-circuiting."""
    errors: list[str] = []

    try:
        validate_metadata(data)
    except ValueError as exc:
        errors.append(str(exc))

    try:
        normalization = validate_normalization(data)
    except ValueError as exc:
        errors.append(str(exc))
        normalization = None

    total_loss = 0
    seen: set[str] = set()
    if normalization is not None:
        blocks = data.get("proof_blocks")
        if not isinstance(blocks, list) or not blocks:
            errors.append("proof_blocks must be a nonempty list")
        else:
            for index, block in enumerate(blocks):
                path = f"proof_blocks[{index}]"
                if not isinstance(block, dict):
                    errors.append(f"{path} must be an object")
                    continue
                name = block.get("name")
                if isinstance(block.get("derivative_loss"), int):
                    total_loss += block["derivative_loss"]
                try:
                    validated_name = validate_block(block, normalization, path)
                except ValueError as exc:
                    errors.append(str(exc))
                    if isinstance(name, str) and name in REQUIRED_BLOCKS:
                        validated_name = name
                    else:
                        continue
                if validated_name in seen:
                    errors.append(f"{path}.name duplicates an earlier block")
                seen.add(validated_name)
            missing = REQUIRED_BLOCKS - seen
            if missing:
                errors.append(f"proof_blocks missing required blocks {sorted(missing)}")

    try:
        validate_near_extremal_policy(data)
    except ValueError as exc:
        errors.append(str(exc))

    try:
        validate_compatibility(data, total_loss)
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        return {
            "accepted": False,
            "claim_scope": data.get("metadata", {}).get("claim_scope", "unknown"),
            "checked_blocks": sorted(REQUIRED_BLOCKS),
            "errors": errors,
            "total_derivative_loss": total_loss,
        }
    return {
        "accepted": True,
        "claim_scope": data["metadata"]["claim_scope"],
        "checked_blocks": sorted(REQUIRED_BLOCKS),
        "errors": [],
        "total_derivative_loss": total_loss,
    }


def verify_file(path: Path) -> dict[str, Any]:
    return verify(load_json(path))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.package)
    except ValueError as exc:
        print(f"completion package rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
