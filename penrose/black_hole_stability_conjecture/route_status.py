#!/usr/bin/env python3
"""Render the current Kerr certificate route status as Markdown."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LEDGER = Path("certificates/candidate_inputs/current_literature.json")


def load_ledger(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "inputs" not in data or not isinstance(data["inputs"], list):
        raise ValueError("ledger must contain an inputs list")
    return data


def render_list(items: list[str]) -> str:
    return "<br>".join(f"- {escape_cell(item)}" for item in items)


def escape_cell(value: object) -> str:
    return str(value).replace("|", r"\|")


def render_markdown(data: dict[str, Any]) -> str:
    metadata = data.get("metadata", {})
    lines = [
        "# Kerr Certificate Route Status",
        "",
        f"Generated on: {metadata.get('generated_on', 'unknown')}",
        "",
        metadata.get("warning", "This ledger is not a proof certificate."),
        "",
        "| ID | Component | Status | Coverage | Remaining gaps |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in data["inputs"]:
        lines.append(
            "| {id} | {component} | {status} | {coverage} | {gaps} |".format(
                id=entry.get("id", ""),
                component=escape_cell(entry.get("component", "")),
                status=escape_cell(entry.get("status", "")),
                coverage=render_list(entry.get("coverage", [])),
                gaps=render_list(entry.get("does_not_cover", [])),
            )
        )

    missing = [entry for entry in data["inputs"] if entry.get("status") in {"missing", "candidate_external_input", "model_external_input"}]
    lines.extend(["", "## Blocking Items", ""])
    if missing:
        for entry in missing:
            lines.append(f"- `{entry.get('id')}` remains `{entry.get('status')}`: {entry.get('component')}.")
    else:
        lines.append("- No blocking items recorded.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ledger = Path(argv[1]) if len(argv) > 1 else DEFAULT_LEDGER
    try:
        data = load_ledger(ledger)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"route status failed: {exc}", file=sys.stderr)
        return 1
    print(render_markdown(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
