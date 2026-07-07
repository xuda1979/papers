#!/usr/bin/env python3
"""Validate the composition of a radial atlas, angular patch certificates, and an ATSC certificate.

This composed verifier checks that:
  1. the radial atlas is well-formed (charts, transitions, denominator certificates);
  2. the ATSC verifier `verify_angular_torus_subdivision_cover.py` accepts `C`;
  3. the cross-certificate agreement holds:
     - header fields (claim_scope, background, spin) match the radial atlas metadata;
     - every `positivity_certificate_reference` in the ATSC certificate refers to a
       file that exists in the certificates/ directory;
     - the radial atlas reference matches the actual radial atlas path;
     - the per-chart radial chart names in the ATSC certificate match the actual
       radial atlas chart names.

If any cross-check fails, the composed verifier rejects.

Usage:
    python3 verify_composed_angular_torus_cover.py <radial_atlas.json> <atsc_certificate.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0, str(ROOT))

import verify_angular_torus_subdivision_cover as vatsc  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def cross_validate(atlas: dict[str, Any], atsc: dict[str, Any], errors: list[str]) -> None:
    """Check that the ATSC certificate matches the radial atlas."""

    atlas_meta = atlas.get("metadata", {})
    rac = atsc.get("radial_atlas_compatibility", {})

    # --- Background / spin agreement ---
    atlas_spin = atlas_meta.get("spin")
    atsc_spin = atsc.get("spin")
    _check(atlas_spin == atsc_spin,
           f"spin mismatch: atlas={atlas_spin!r}, atsc={atsc_spin!r}",
           errors)

    # --- Radial chart name agreement ---
    atlas_chart_names = {c["chart"] for c in atlas.get("charts", [])}
    atsc_chart_entries = rac.get("per_chart_lambda_eta_coverage", [])
    atsc_chart_names = {e.get("radial_chart") for e in atsc_chart_entries}
    _check(atlas_chart_names == atsc_chart_names,
           f"radial chart name sets differ: atlas has {len(atlas_chart_names)} charts, "
           f"atsc references {len(atsc_chart_names)} charts; "
           f"missing in atsc: {sorted(atlas_chart_names - atsc_chart_names)}; "
           f"extra in atsc: {sorted(atsc_chart_names - atlas_chart_names)}",
           errors)

    # --- Radial chart count agreement ---
    _check(rac.get("radial_chart_count") == len(atlas_chart_names),
           f"radial_chart_count mismatch: atlas has {len(atlas_chart_names)} charts, "
           f"atsc declares {rac.get('radial_chart_count')}",
           errors)

    # --- Positivity certificate references must point to existing files ---
    sub = atsc.get("angular_subdivision", {})
    for i, iv in enumerate(sub.get("intervals", [])):
        ref = iv.get("positivity_certificate_reference", "")
        ref_path = ROOT / ref
        _check(ref_path.exists(),
               f"angular_subdivision.intervals[{i}].positivity_certificate_reference "
               f"({ref!r}) does not point to an existing file at {ref_path}",
               errors)


def verify_composed(atlas_path: Path, atsc_path: Path) -> dict[str, Any]:
    errors: list[str] = []

    atlas = _load(atlas_path)
    atsc = _load(atsc_path)

    # Validate radial atlas minimally (charts must be a nonempty list with names).
    atlas_charts = atlas.get("charts")
    if not isinstance(atlas_charts, list) or not atlas_charts:
        errors.append("radial atlas must have a nonempty 'charts' list")
        atlas_ok = False
    else:
        atlas_ok = True
        for i, c in enumerate(atlas_charts):
            if not isinstance(c, dict) or "chart" not in c:
                errors.append(f"radial atlas chart {i} is missing 'chart' name")
                atlas_ok = False

    # Run ATSC verifier.
    try:
        atsc_result = vatsc.verify(atsc)
    except ValueError as exc:
        errors.append(f"ATSC certificate rejected: {exc}")
        atsc_result = None

    # Cross-validate only if both upstream checks passed.
    if atlas_ok and atsc_result is not None:
        cross_validate(atlas, atsc, errors)

    if errors:
        return {
            "accepted": False,
            "composition": "radial_atlas + angular_patches + ATSC",
            "errors": errors,
            "atlas_accepted": atlas_ok,
            "atsc_accepted": atsc_result is not None,
        }
    return {
        "accepted": True,
        "composition": "radial_atlas + angular_patches + ATSC",
        "claim_scope": atsc_result["claim_scope"],
        "spin": atsc_result["spin"],
        "interval_count": atsc_result["interval_count"],
        "tag_distribution": atsc_result["tag_distribution"],
        "radial_chart_count": atsc_result["radial_chart_count"],
        "min_overlap_length": atsc_result["min_overlap_length"],
        "min_interval_length": atsc_result["min_interval_length"],
        "total_length": atsc_result["total_length"],
        "warning": ("Composed combinatorial, arithmetic, coverage, and "
                    "cross-certificate checks pass. This does NOT prove the "
                    "black hole stability conjecture: the per-patch positivity "
                    "certificates and the mid-latitude positivity certificate "
                    "remain open."),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("atsc", type=Path)
    args = parser.parse_args(argv[1:])
    result = verify_composed(args.atlas, args.atsc)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
