#!/usr/bin/env python3
"""Validate the fine-grained Kerr stability completion roadmap.

This checker keeps the roadmap honest as a dependency graph: every node has a
status, phase, acceptance criterion, and explicit blockers; dependencies must
refer to existing nodes and must be acyclic.  It does not prove the analytic
claims listed in the roadmap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CLAIM_SCOPE = "fine_grained_completion_roadmap"
ALLOWED_STATUSES = {
    "proved",
    "partial",
    "reduction",
    "gap",
    "speculative",
}
ALLOWED_PHASES = {
    "first_chart_qgrav",
    "atlas_propagation",
    "kerr_orbit_cohomology",
    "wick_quantization",
    "linear_package",
    "near_extremal",
    "nonlinear_closure",
    "completion_assembly",
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


def validate_string_list(raw: Any, path: str, allow_empty: bool = True) -> list[str]:
    require(isinstance(raw, list), f"{path} must be a list")
    require(allow_empty or bool(raw), f"{path} must be nonempty")
    require(all(isinstance(item, str) and item for item in raw), f"{path} entries must be nonempty strings")
    require(len(set(raw)) == len(raw), f"{path} entries must be distinct")
    return list(raw)


def validate_metadata(data: dict[str, Any]) -> None:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "claim_scope", "status"], "metadata")
    require(metadata["claim_scope"] == CLAIM_SCOPE, f"metadata.claim_scope must be {CLAIM_SCOPE}")
    require(isinstance(metadata["name"], str) and metadata["name"], "metadata.name must be nonempty")
    require(isinstance(metadata["status"], str) and metadata["status"], "metadata.status must be nonempty")


def validate_nodes(raw_nodes: Any) -> dict[str, dict[str, Any]]:
    require(isinstance(raw_nodes, list) and raw_nodes, "nodes must be a nonempty list")
    nodes: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(raw_nodes):
        path = f"nodes[{index}]"
        require(isinstance(node, dict), f"{path} must be an object")
        require_keys(
            node,
            ["id", "phase", "status", "statement", "dependencies", "acceptance_criteria", "artifacts", "blockers"],
            path,
        )
        node_id = node["id"]
        require(isinstance(node_id, str) and node_id, f"{path}.id must be nonempty")
        require(node_id not in nodes, f"duplicate node id {node_id!r}")
        require(node["phase"] in ALLOWED_PHASES, f"{path}.phase must be one of {sorted(ALLOWED_PHASES)}")
        require(node["status"] in ALLOWED_STATUSES, f"{path}.status must be one of {sorted(ALLOWED_STATUSES)}")
        validate_string_list(node["dependencies"], f"{path}.dependencies")
        validate_string_list(node["acceptance_criteria"], f"{path}.acceptance_criteria", allow_empty=False)
        validate_string_list(node["artifacts"], f"{path}.artifacts")
        validate_string_list(node["blockers"], f"{path}.blockers")
        require(isinstance(node["statement"], str) and node["statement"].strip(), f"{path}.statement must be nonempty")
        if node["status"] == "proved":
            require(not node["blockers"], f"{path} is proved but still lists blockers")
        if node["status"] in {"gap", "speculative"}:
            require(node["blockers"], f"{path} is {node['status']} but lists no blockers")
        nodes[node_id] = node
    return nodes


def validate_dependencies(nodes: dict[str, dict[str, Any]]) -> None:
    for node_id, node in nodes.items():
        for dep in node["dependencies"]:
            require(dep in nodes, f"node {node_id!r} depends on unknown node {dep!r}")
            require(dep != node_id, f"node {node_id!r} depends on itself")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, stack: tuple[str, ...]) -> None:
        if node_id in visited:
            return
        require(node_id not in visiting, f"cycle detected: {' -> '.join(stack + (node_id,))}")
        visiting.add(node_id)
        for dep in nodes[node_id]["dependencies"]:
            visit(dep, stack + (node_id,))
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id, ())


def frontier(nodes: dict[str, dict[str, Any]]) -> list[str]:
    open_statuses = {"partial", "reduction", "gap", "speculative"}
    proved = {node_id for node_id, node in nodes.items() if node["status"] == "proved"}
    result: list[str] = []
    for node_id, node in nodes.items():
        if node["status"] in open_statuses and set(node["dependencies"]).issubset(proved):
            result.append(node_id)
    return sorted(result)


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(data, dict), "roadmap must be an object")
    validate_metadata(data)
    nodes = validate_nodes(data.get("nodes"))
    validate_dependencies(nodes)
    counts = {status: 0 for status in sorted(ALLOWED_STATUSES)}
    phases = {phase: 0 for phase in sorted(ALLOWED_PHASES)}
    for node in nodes.values():
        counts[node["status"]] += 1
        phases[node["phase"]] += 1
    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "node_count": len(nodes),
        "status_counts": counts,
        "phase_counts": phases,
        "frontier": frontier(nodes),
    }


def verify_file(path: Path) -> dict[str, Any]:
    return verify(load_json(path))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roadmap", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.roadmap)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Fine-grained roadmap rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
