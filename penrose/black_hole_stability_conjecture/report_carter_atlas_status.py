#!/usr/bin/env python3
"""Render a status report for Carter atlas completeness."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_atlas_coverage as coverage


VERIFIED_STATUS = "verified"
CERTIFIED_LOCAL_STATUSES = {
    VERIFIED_STATUS,
    "seed_angular_overlap_policy_verified",
    "interval_transition_seed_verified",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def escape_cell(value: object) -> str:
    return str(value).replace("|", r"\|")


def render_list(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""
    return "<br>".join(f"- {escape_cell(item)}" for item in items)


def resolve_path(raw_path: Any, base_dir: Path | None) -> Path:
    require(isinstance(raw_path, str) and raw_path.strip(), "atlas_path must be a nonempty string")
    path = Path(raw_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def expected_endpoint(data: dict[str, Any], key: str) -> Fraction | None:
    expected = data.get("expected_radial_cover", {})
    if not isinstance(expected, dict) or key not in expected:
        return None
    return Fraction(str(expected[key]))


def family_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    families = data.get("chart_families")
    require(isinstance(families, list) and families, "chart_families must be a nonempty list")
    result: dict[str, dict[str, Any]] = {}
    for index, family in enumerate(families):
        require(isinstance(family, dict), f"chart_families[{index}] must be an object")
        name = family.get("name")
        require(isinstance(name, str) and name, f"chart_families[{index}].name must be a nonempty string")
        require(name not in result, f"duplicate chart family {name!r}")
        status = family.get("status")
        require(isinstance(status, str) and status, f"chart_families[{index}].status must be a nonempty string")
        result[name] = family
    return result


def audit_atlas(data: dict[str, Any], base_dir: Path | None) -> dict[str, Any]:
    atlas = coverage.load_json(resolve_path(data.get("atlas_path"), base_dir))
    return coverage.verify(
        atlas,
        expected_start=expected_endpoint(data, "start"),
        expected_end=expected_endpoint(data, "end"),
    )


def audit(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    atlas_result: dict[str, Any] | None = None
    try:
        atlas_result = audit_atlas(data, base_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))

    blocking = []
    try:
        families = family_map(data)
        for name, family in families.items():
            if family.get("status") != VERIFIED_STATUS:
                blocking.append(name)
    except ValueError as exc:
        errors.append(str(exc))
        families = {}

    if errors:
        return {
            "accepted": False,
            "errors": errors,
            "blocking_families": sorted(set(blocking)),
            "atlas": atlas_result,
        }
    return {
        "accepted": not blocking,
        "errors": [],
        "blocking_families": sorted(set(blocking)),
        "atlas": atlas_result,
        "checked_families": sorted(families),
    }


def local_certification_audit(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    base_audit = audit(data, base_dir)
    errors = list(base_audit["errors"])
    locally_open = []
    try:
        families = family_map(data)
        for name, family in families.items():
            status = family.get("status")
            if status not in CERTIFIED_LOCAL_STATUSES:
                locally_open.append(name)
    except ValueError as exc:
        errors.append(str(exc))
        families = {}
    return {
        "accepted": not errors and not locally_open,
        "errors": errors,
        "locally_open_families": sorted(set(locally_open)),
        "certified_families": sorted(
            name
            for name, family in families.items()
            if family.get("status") in CERTIFIED_LOCAL_STATUSES
        ),
        "atlas": base_audit["atlas"],
    }


def render_markdown(data: dict[str, Any], base_dir: Path | None = None) -> str:
    metadata = data.get("metadata", {})
    require(isinstance(metadata, dict), "metadata must be an object")
    families = family_map(data)
    atlas_audit = audit(data, base_dir)

    lines = [
        "# Carter Atlas Completeness Status",
        "",
        f"Package: {metadata.get('name', 'unknown')}",
        f"Spin scope: {metadata.get('spin_scope', 'unknown')}",
        "",
        metadata.get("warning", "This report is not a proof certificate."),
        "",
        "## Radial Atlas Audit",
        "",
    ]

    if atlas_audit["atlas"] is None:
        lines.append("- Atlas coverage and transition audit: failed before producing a result.")
    else:
        atlas = atlas_audit["atlas"]
        lines.append(
            "- Atlas coverage and transition audit: accepted "
            f"({atlas['chart_count']} charts, {len(atlas['transitions'])} transitions, "
            f"coverage {atlas['coverage'][0]} to {atlas['coverage'][1]})."
        )

    if atlas_audit["errors"]:
        for error in atlas_audit["errors"]:
            lines.append(f"- `{escape_cell(error)}`")

    lines.extend(
        [
            "",
            "## Chart Families",
            "",
            "| Family | Status | Evidence | Remaining obstruction |",
            "| --- | --- | --- | --- |",
        ]
    )

    for name in sorted(families):
        family = families[name]
        lines.append(
            "| {name} | {status} | {evidence} | {obstructions} |".format(
                name=escape_cell(name),
                status=escape_cell(family.get("status", "")),
                evidence=render_list(family.get("evidence", [])),
                obstructions=render_list(family.get("remaining_obstructions", [])),
            )
        )

    lines.extend(["", "## Blocking Items", ""])
    blocking = atlas_audit["blocking_families"]
    if blocking:
        for name in blocking:
            lines.append(f"- `{name}` is not yet fully verified.")
    else:
        lines.append("- No blocking chart families recorded.")
    local_audit = local_certification_audit(data, base_dir)
    lines.extend(["", "## Local Certification Audit", ""])
    if local_audit["errors"]:
        for error in local_audit["errors"]:
            lines.append(f"- `{escape_cell(error)}`")
    if local_audit["locally_open_families"]:
        for name in local_audit["locally_open_families"]:
            lines.append(f"- `{name}` has no accepted local seed/transition certificate yet.")
    else:
        lines.append("- Every recorded spin-half chart family has at least local certified evidence.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status_package", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        data = load_json(args.status_package)
        print(render_markdown(data, base_dir=args.status_package.parent))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter atlas status failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
