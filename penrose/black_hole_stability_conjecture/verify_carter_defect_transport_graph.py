#!/usr/bin/env python3
"""Validate finite graphs of Carter defect-transport morphisms.

The checker starts from one accepted trace-defect absorber and then verifies a
finite ordered transport graph.  Each edge may use any chart already certified
by earlier edges as its source.  The target inherits the source margin multiplied
by the edge phase-norm lower factor.  This gives an exact-arithmetic certificate
for iterating local defect transports through an atlas graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_defect_transport_morphism as transport
import verify_carter_sos_identity as sos
import verify_carter_trace_defect_absorber as absorber


CLAIM_SCOPE = "finite_defect_transport_graph"
ALLOWED_DERIVATION_STATUSES = {
    "candidate_transport_graph",
    "kerr_gravitational_proved",
}


@dataclass(frozen=True)
class CertifiedNode:
    chart: str
    phase_variables: tuple[str, ...]
    density_expression: Any
    nonphase_variables: tuple[str, ...]
    local_delta: Fraction
    path: tuple[str, ...]


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
            "metadata.derivation_status kerr_gravitational_proved requires exact "
            "Kerr derivations on every graph node; this checker verifies only "
            "finite defect-transport graph propagation"
        )
    return str(derivation_status)


def validate_edge_metadata(edge: dict[str, Any], path: str) -> None:
    metadata = edge.get("metadata")
    if metadata is None:
        return
    require(isinstance(metadata, dict), f"{path}.metadata must be an object")
    if "claim_scope" in metadata:
        require(
            metadata["claim_scope"] == transport.CLAIM_SCOPE,
            f"{path}.metadata.claim_scope must be {transport.CLAIM_SCOPE}",
        )
    derivation_status = metadata.get("derivation_status")
    if derivation_status == "kerr_gravitational_proved":
        raise ValueError(
            f"{path}.metadata.derivation_status kerr_gravitational_proved requires "
            "an exact Kerr derivation on the edge target chart"
        )


def validate_string_list(raw: Any, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    require(isinstance(raw, list), f"{name} must be a list")
    require(allow_empty or bool(raw), f"{name} must be nonempty")
    require(all(isinstance(item, str) and item for item in raw), f"{name} entries must be nonempty strings")
    require(len(set(raw)) == len(raw), f"{name} entries must be distinct")
    return tuple(raw)


def validate_phase_pullback_map(
    raw_map: Any,
    source_phase_variables: tuple[str, ...],
    target_phase_count: int,
    variables: tuple[str, ...],
) -> dict[str, Any]:
    require(isinstance(raw_map, dict), "phase_pullback_map must be an object")
    require(set(raw_map) == set(source_phase_variables), f"phase_pullback_map must contain exactly {list(source_phase_variables)}")
    pullback: dict[str, Any] = dict(raw_map)
    for source_name in source_phase_variables:
        poly = transport.eval_expr_with_phase_pullback(pullback[source_name], variables, pullback)
        transport.validate_phase_homogeneous(poly, target_phase_count, 1, f"phase pullback {source_name!r}")
    return pullback


def load_seed_node(data: dict[str, Any], base_dir: Path | None) -> tuple[CertifiedNode, str, list[str]]:
    require_keys(data, ["seed_chart", "seed_defect_absorber_path"], "certificate")
    seed_path = resolve_path(data["seed_defect_absorber_path"], base_dir, "seed_defect_absorber_path")
    seed_data = load_json(seed_path)
    seed_result = absorber.verify(seed_data, base_dir=seed_path.parent)
    seed_chart = data["seed_chart"]
    require(isinstance(seed_chart, str) and seed_chart, "seed_chart must be a nonempty string")
    require(seed_chart == seed_result["chart"], "seed_chart must match seed defect absorber chart")
    require("candidate_density_expression" in seed_data, "seed absorber candidate_density_expression is missing")

    phase_variables, symbolic_variables, source_auxiliary_variables, defect_auxiliary_variables = (
        transport.source_variable_families_from_paths(seed_data, seed_path)
    )
    nonphase_variables = symbolic_variables + source_auxiliary_variables + defect_auxiliary_variables
    require(len(set(phase_variables + nonphase_variables)) == len(phase_variables + nonphase_variables), "seed phase and nonphase variables must be disjoint")
    node = CertifiedNode(
        chart=seed_chart,
        phase_variables=phase_variables,
        density_expression=seed_data["candidate_density_expression"],
        nonphase_variables=nonphase_variables,
        local_delta=sos.parse_fraction(seed_result["local_coercivity_delta"]),
        path=(seed_chart,),
    )
    return node, str(seed_path), list(seed_result["checked_identities"])


def edge_name(edge: dict[str, Any], index: int) -> str:
    metadata = edge.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("name"), str) and metadata["name"]:
        return metadata["name"]
    if isinstance(edge.get("name"), str) and edge["name"]:
        return edge["name"]
    if isinstance(edge.get("target_chart"), str) and edge["target_chart"]:
        return edge["target_chart"]
    return f"transport_edge_{index}"


def load_edge(raw_edge: Any, base_dir: Path | None, path: str) -> dict[str, Any]:
    require(isinstance(raw_edge, dict), f"{path} must be an object")
    if "certificate_path" not in raw_edge:
        return raw_edge
    require(set(raw_edge) == {"certificate_path"}, f"{path} with certificate_path must not contain inline edge fields")
    edge_path = resolve_path(raw_edge["certificate_path"], base_dir, f"{path}.certificate_path")
    return load_json(edge_path)


def verify_edge(
    edge: dict[str, Any],
    source: CertifiedNode,
    index: int,
) -> tuple[CertifiedNode, dict[str, Any]]:
    path = f"transport_edges[{index}]"
    validate_edge_metadata(edge, path)
    require_keys(
        edge,
        [
            "source_chart",
            "target_chart",
            "variables",
            "phase_pullback_map",
            "target_density_expression",
            "nonnegative_transport_terms",
            "constraint_ideal_terms",
            "phase_norm_lower_bound",
        ],
        path,
    )
    require(edge["source_chart"] == source.chart, f"{path}.source_chart is not the certified source chart")
    target_chart = edge["target_chart"]
    require(isinstance(target_chart, str) and target_chart, f"{path}.target_chart must be a nonempty string")
    target_phase_variables = validate_string_list(edge["variables"], f"{path}.variables")
    require(
        len(target_phase_variables) == len(source.phase_variables),
        f"{path}.variables must have the same phase dimension as the source chart",
    )
    transport_auxiliary_variables = validate_string_list(
        edge.get("transport_auxiliary_variables", []),
        f"{path}.transport_auxiliary_variables",
        allow_empty=True,
    )
    all_variables = target_phase_variables + source.nonphase_variables + transport_auxiliary_variables
    require(len(set(all_variables)) == len(all_variables), f"{path} target, source, and transport variables must be disjoint")

    phase_pullback_map = validate_phase_pullback_map(
        edge["phase_pullback_map"],
        source.phase_variables,
        len(target_phase_variables),
        all_variables,
    )
    checked_identities, checked_constraints, checked_nonnegative = transport.verify_density_transport(
        edge,
        {"candidate_density_expression": source.density_expression},
        all_variables,
        len(target_phase_variables),
        phase_pullback_map,
    )
    norm_lower_bound, checked_norm_terms = transport.verify_phase_norm_lower_bound(
        edge,
        source.phase_variables,
        target_phase_variables,
        all_variables,
        phase_pullback_map,
    )
    target_delta = source.local_delta * norm_lower_bound
    target = CertifiedNode(
        chart=target_chart,
        phase_variables=target_phase_variables,
        density_expression=edge["target_density_expression"],
        nonphase_variables=source.nonphase_variables + transport_auxiliary_variables,
        local_delta=target_delta,
        path=source.path + (target_chart,),
    )
    summary = {
        "name": edge_name(edge, index),
        "source_chart": source.chart,
        "target_chart": target_chart,
        "phase_norm_lower_bound": str(norm_lower_bound),
        "source_delta": str(source.local_delta),
        "target_delta": str(target_delta),
        "checked_identities": checked_identities,
        "checked_constraint_terms": checked_constraints,
        "checked_nonnegative_transport_terms": checked_nonnegative,
        "checked_phase_norm_defects": checked_norm_terms,
    }
    return target, summary


def verify_required_targets(data: dict[str, Any], nodes: dict[str, CertifiedNode]) -> None:
    required_targets = data.get("required_target_charts", [])
    required = validate_string_list(required_targets, "required_target_charts", allow_empty=True)
    for chart in required:
        require(chart in nodes, f"required target chart {chart!r} was not certified by the graph")


def verify_minimum_delta(data: dict[str, Any], nodes: dict[str, CertifiedNode]) -> Fraction | None:
    if "minimum_required_delta" not in data:
        return None
    threshold = sos.parse_fraction(data["minimum_required_delta"])
    require(threshold > 0, "minimum_required_delta must be positive")
    for chart, node in nodes.items():
        require(node.local_delta >= threshold, f"chart {chart!r} has local delta {node.local_delta} below required {threshold}")
    return threshold


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    seed_node, seed_path, seed_identities = load_seed_node(data, base_dir)
    edges_raw = data.get("transport_edges")
    require(isinstance(edges_raw, list) and edges_raw, "transport_edges must be a nonempty list")

    nodes: dict[str, CertifiedNode] = {seed_node.chart: seed_node}
    edge_summaries: list[dict[str, Any]] = []
    checked_edge_names: list[str] = []
    for index, raw_edge in enumerate(edges_raw):
        edge = load_edge(raw_edge, base_dir, f"transport_edges[{index}]")
        source_chart = edge.get("source_chart")
        require(isinstance(source_chart, str) and source_chart, f"transport_edges[{index}].source_chart must be a nonempty string")
        require(source_chart in nodes, f"transport_edges[{index}].source_chart has not been certified yet")
        target_chart = edge.get("target_chart")
        require(isinstance(target_chart, str) and target_chart, f"transport_edges[{index}].target_chart must be a nonempty string")
        require(target_chart not in nodes, f"transport_edges[{index}].target_chart duplicates an existing certified chart")
        target, summary = verify_edge(edge, nodes[source_chart], index)
        nodes[target.chart] = target
        edge_summaries.append(summary)
        checked_edge_names.append(summary["name"])

    verify_required_targets(data, nodes)
    required_delta = verify_minimum_delta(data, nodes)
    chart_deltas = {chart: str(node.local_delta) for chart, node in nodes.items()}
    chart_paths = {chart: list(node.path) for chart, node in nodes.items()}
    minimum_delta = min(node.local_delta for node in nodes.values())

    result: dict[str, Any] = {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "seed_chart": seed_node.chart,
        "seed_defect_absorber": seed_path,
        "certified_charts": list(nodes),
        "checked_edges": checked_edge_names,
        "chart_deltas": chart_deltas,
        "chart_paths": chart_paths,
        "minimum_local_coercivity_delta": str(minimum_delta),
        "edge_summaries": edge_summaries,
        "seed_checked_identities": seed_identities,
    }
    if required_delta is not None:
        result["minimum_required_delta"] = str(required_delta)
    return result


def verify_file(path: Path) -> dict[str, Any]:
    return verify(load_json(path), base_dir=path.parent)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter defect transport graph rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
