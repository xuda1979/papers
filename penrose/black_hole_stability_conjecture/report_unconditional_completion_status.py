#!/usr/bin/env python3
"""Render status for a not-yet-proved unconditional completion package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_unconditional_completion_package as completion


DEFAULT_STATUS = Path("certificates/completion_packages/current_public_status_package.json")
PROVED_STATUS = "proved"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def escape_cell(value: object) -> str:
    return str(value).replace("|", r"\|")


def render_list(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""
    return "<br>".join(f"- {escape_cell(item)}" for item in items)


def block_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    blocks = data.get("proof_blocks")
    if not isinstance(blocks, list):
        raise ValueError("status package must contain a proof_blocks list")
    result: dict[str, dict[str, Any]] = {}
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError(f"proof_blocks[{index}] must be an object")
        name = block.get("name")
        if not isinstance(name, str):
            raise ValueError(f"proof_blocks[{index}].name must be a string")
        if name in result:
            raise ValueError(f"duplicate proof block {name!r}")
        result[name] = block
    return result


def render_markdown(data: dict[str, Any]) -> str:
    metadata = data.get("metadata", {})
    blocks = block_map(data)
    missing = sorted(completion.REQUIRED_BLOCKS - set(blocks))
    blocking = []

    lines = [
        "# Unconditional Kerr Completion Status",
        "",
        f"Package: {metadata.get('name', 'unknown')}",
        f"Generated on: {metadata.get('generated_on', 'unknown')}",
        "",
        metadata.get("warning", "This status report is not a proof certificate."),
        "",
        "| Proof block | Status | Best current input | Current coverage | Remaining obstruction |",
        "| --- | --- | --- | --- | --- |",
    ]

    for name in sorted(completion.REQUIRED_BLOCKS):
        block = blocks.get(name, {})
        status = block.get("status", "missing")
        if status != PROVED_STATUS:
            blocking.append(name)
        lines.append(
            "| {name} | {status} | {reference} | {coverage} | {gaps} |".format(
                name=escape_cell(name),
                status=escape_cell(status),
                reference=escape_cell(block.get("reference", "")),
                coverage=render_list(block.get("coverage", [])),
                gaps=render_list(block.get("remaining_obstructions", block.get("does_not_cover", []))),
            )
        )

    if missing:
        blocking.extend(missing)

    lines.extend(["", "## Verifier Audit", ""])
    audit = completion.audit(data)
    if audit["accepted"]:
        lines.append(f"- Accepted by proof-package verifier: `{escape_cell(audit)}`.")
    else:
        lines.append("- Rejected by proof-package verifier with these blockers:")
        for error in audit["errors"]:
            lines.append(f"- `{escape_cell(error)}`")

    lines.extend(["", "## Blocking Items", ""])
    if blocking:
        for name in sorted(set(blocking)):
            lines.append(f"- `{name}` is not yet a proved compatible input.")
    else:
        lines.append("- No blocking items recorded.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status_package", nargs="?", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args(argv[1:])
    try:
        data = load_json(args.status_package)
        print(render_markdown(data))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"completion status failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
