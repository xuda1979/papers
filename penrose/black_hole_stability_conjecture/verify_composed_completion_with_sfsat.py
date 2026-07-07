#!/usr/bin/env python3
"""Validate the composition of an unconditional completion package with a SFSAT certificate.

This composed verifier checks that:
  1. the completion package `verify_unconditional_completion_package.py` accepts `P`;
  2. the SFSAT verifier `verify_sobolev_frequency_symbol_admissibility.py` accepts `C`;
  3. the cross-certificate agreement holds:
     - header fields (claim_scope, background, spin) match;
     - sobolev_hierarchy.N_shared matches normalization.sobolev_index;
     - frequency_decomposition.name matches normalization.frequency_decomposition;
     - symbol_class.name matches normalization.symbol_class;
     - each block_margins entry matches the corresponding proof_blocks entry
       (derivative_loss, positive_margin, remainder_bound);
     - composition_order is a permutation of the seven block names appearing in proof_blocks;
     - admissibility_budget.total_derivative_loss matches the total derivative loss
       computed from proof_blocks;
     - admissibility_budget.max_total_derivative_loss matches compatibility.max_total_derivative_loss;
     - admissibility_budget.available_derivative_buffer matches compatibility.available_derivative_buffer.

If any cross-check fails, the composed verifier rejects.

Usage:
    python3 verify_composed_completion_with_sfsat.py <package.json> <sfsat_certificate.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Import the upstream verifiers from the same directory.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import verify_unconditional_completion_package as vcp  # noqa: E402
import verify_sobolev_frequency_symbol_admissibility as vsfsat  # noqa: E402


REQUIRED_BLOCKS = vcp.REQUIRED_BLOCKS


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def cross_validate(package: dict[str, Any], sfsat: dict[str, Any], errors: list[str]) -> None:
    """Check that the SFSAT certificate matches the completion package."""

    # --- Header agreement ---
    pkg_meta = package.get("metadata", {})
    pkg_norm = package.get("normalization", {})

    _check(sfsat.get("claim_scope") == pkg_meta.get("claim_scope"),
           f"claim_scope mismatch: package={pkg_meta.get('claim_scope')!r}, "
           f"sfsat={sfsat.get('claim_scope')!r}",
           errors)
    _check(sfsat.get("background") == pkg_norm.get("background"),
           f"background mismatch: package={pkg_norm.get('background')!r}, "
           f"sfsat={sfsat.get('background')!r}",
           errors)
    _check(sfsat.get("spin") == pkg_meta.get("spin"),
           f"spin mismatch: package={pkg_meta.get('spin')!r}, "
           f"sfsat={sfsat.get('spin')!r}",
           errors)

    # --- Sobolev index agreement ---
    pkg_N = pkg_norm.get("sobolev_index")
    sfsat_N = sfsat.get("sobolev_hierarchy", {}).get("N_shared")
    _check(pkg_N == sfsat_N,
           f"sobolev index mismatch: package.normalization.sobolev_index={pkg_N!r}, "
           f"sfsat.sobolev_hierarchy.N_shared={sfsat_N!r}",
           errors)

    # --- Frequency decomposition name agreement ---
    pkg_fd = pkg_norm.get("frequency_decomposition")
    sfsat_fd = sfsat.get("frequency_decomposition", {}).get("name")
    _check(pkg_fd == sfsat_fd,
           f"frequency_decomposition mismatch: package={pkg_fd!r}, sfsat={sfsat_fd!r}",
           errors)

    # --- Symbol class name agreement ---
    pkg_sc = pkg_norm.get("symbol_class")
    sfsat_sc = sfsat.get("symbol_class", {}).get("name")
    _check(pkg_sc == sfsat_sc,
           f"symbol_class mismatch: package={pkg_sc!r}, sfsat={sfsat_sc!r}",
           errors)

    # --- Block agreement ---
    pkg_blocks = {b["name"]: b for b in package.get("proof_blocks", [])}
    sfsat_blocks = {b["name"]: b for b in sfsat.get("block_margins", [])}
    _check(set(pkg_blocks.keys()) == set(sfsat_blocks.keys()),
           f"block name sets differ: package={sorted(pkg_blocks.keys())}, "
           f"sfsat={sorted(sfsat_blocks.keys())}",
           errors)

    for name in pkg_blocks:
        if name not in sfsat_blocks:
            continue
        pb = pkg_blocks[name]
        sb = sfsat_blocks[name]
        _check(pb.get("derivative_loss") == sb.get("derivative_loss"),
                f"block {name!r}: derivative_loss mismatch "
                f"(package={pb.get('derivative_loss')}, sfsat={sb.get('derivative_loss')})",
                errors)
        _check(pb.get("positive_margin") == sb.get("positive_margin"),
                f"block {name!r}: positive_margin mismatch "
                f"(package={pb.get('positive_margin')!r}, "
                f"sfsat={sb.get('positive_margin')!r})",
                errors)
        _check(pb.get("remainder_bound") == sb.get("remainder_bound"),
                f"block {name!r}: remainder_bound mismatch "
                f"(package={pb.get('remainder_bound')!r}, "
                f"sfsat={sb.get('remainder_bound')!r})",
                errors)

    # --- Composition order is a permutation of the seven required blocks ---
    co = sfsat.get("composition_order", [])
    _check(isinstance(co, list) and len(co) == len(REQUIRED_BLOCKS),
           f"composition_order must be a list of length {len(REQUIRED_BLOCKS)}",
           errors)
    if isinstance(co, list):
        _check(set(co) == REQUIRED_BLOCKS,
               f"composition_order must be a permutation of {sorted(REQUIRED_BLOCKS)}, "
               f"got {sorted(co)}",
               errors)

    # --- Admissibility budget agreement ---
    pkg_compat = package.get("compatibility", {})
    sfsat_budget = sfsat.get("admissibility_budget", {})

    pkg_total_loss = sum(pb.get("derivative_loss", 0) for pb in pkg_blocks.values())
    _check(sfsat_budget.get("total_derivative_loss") == pkg_total_loss,
           f"total_derivative_loss mismatch: package computes {pkg_total_loss}, "
           f"sfsat declares {sfsat_budget.get('total_derivative_loss')}",
           errors)
    _check(sfsat_budget.get("max_total_derivative_loss") == pkg_compat.get("max_total_derivative_loss"),
           f"max_total_derivative_loss mismatch: package={pkg_compat.get('max_total_derivative_loss')}, "
           f"sfsat={sfsat_budget.get('max_total_derivative_loss')}",
           errors)
    _check(sfsat_budget.get("available_derivative_buffer") == pkg_compat.get("available_derivative_buffer"),
           f"available_derivative_buffer mismatch: "
           f"package={pkg_compat.get('available_derivative_buffer')}, "
           f"sfsat={sfsat_budget.get('available_derivative_buffer')}",
           errors)


def verify_composed(package_path: Path, sfsat_path: Path) -> dict[str, Any]:
    errors: list[str] = []

    package = _load(package_path)
    sfsat = _load(sfsat_path)

    # Run each upstream verifier independently.
    try:
        pkg_result = vcp.verify(package)
    except ValueError as exc:
        errors.append(f"completion package rejected: {exc}")
        pkg_result = None

    try:
        sfsat_result = vsfsat.verify(sfsat)
    except ValueError as exc:
        errors.append(f"SFSAT certificate rejected: {exc}")
        sfsat_result = None

    # Cross-validate only if both upstream checks passed.
    if pkg_result is not None and sfsat_result is not None:
        cross_validate(package, sfsat, errors)

    if errors:
        return {
            "accepted": False,
            "composition": "completion_package + SFSAT",
            "errors": errors,
            "package_accepted": pkg_result is not None,
            "sfsat_accepted": sfsat_result is not None,
        }
    return {
        "accepted": True,
        "composition": "completion_package + SFSAT",
        "claim_scope": pkg_result["claim_scope"],
        "checked_blocks": sorted(REQUIRED_BLOCKS),
        "total_derivative_loss": pkg_result["total_derivative_loss"],
        "sfsat_total_derivative_loss": sfsat_result["total_derivative_loss"],
        "sfsat_final_prefix_product": sfsat_result["final_prefix_product"],
        "sfsat_product_lower_bound": sfsat_result["product_lower_bound"],
        "sfsat_N_shared": sfsat_result["N_shared"],
        "sfsat_N_min": sfsat_result["N_min"],
        "sfsat_N_symbol_class": sfsat_result["N_symbol_class"],
        "sfsat_layer_count": sfsat_result["layer_count"],
        "warning": ("Composed combinatorial, arithmetic, internal-coherence, and "
                    "cross-certificate checks pass. This does NOT prove the black "
                    "hole stability conjecture: the analytic content of each block "
                    "and the SFSAT analytic inputs (regularity scales, frequency "
                    "minimum, composition order) remain open."),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("sfsat", type=Path)
    args = parser.parse_args(argv[1:])
    result = verify_composed(args.package, args.sfsat)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
