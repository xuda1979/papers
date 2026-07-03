#!/usr/bin/env python3
"""Audit exact-derivation obligations for a Carter principal-symbol trace.

This checker is a finite bookkeeping gate between a candidate trace certificate
and a future exact Kerr principal-symbol derivation.  It does not differentiate
the Einstein operator.  Instead, it requires every trace primitive and every
backend derivation obligation to be registered with a status, source target,
and replacement criterion.  The global proof status remains open unless every
entry is marked exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_carter_principal_symbol_trace as trace


CLAIM_SCOPE = "chart_exact_derivation_audit"
ALLOWED_AUDIT_STATUSES = {
    "candidate_audit",
    "exact_derivation_complete",
}
ALLOWED_ENTRY_STATUSES = {
    "candidate_placeholder",
    "formula_target",
    "exact_formula",
}
REQUIRED_SOURCE_TARGETS = {
    "linearized_kerr_principal_symbol",
    "trapped_characteristic_constraint",
    "normalization_or_interval_extraction",
    "quotient_identity",
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


def resolve_path(raw_path: Any, base_dir: Path | None, path_name: str) -> Path:
    require(isinstance(raw_path, str) and raw_path.strip(), f"{path_name} must be a nonempty string")
    path = Path(raw_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def validate_metadata(data: dict[str, Any]) -> str:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "claim_scope", "status", "audit_status"], "metadata")
    require(metadata["claim_scope"] == CLAIM_SCOPE, f"metadata.claim_scope must be {CLAIM_SCOPE}")
    audit_status = metadata["audit_status"]
    require(audit_status in ALLOWED_AUDIT_STATUSES, f"metadata.audit_status must be one of {sorted(ALLOWED_AUDIT_STATUSES)}")
    return str(audit_status)


def validate_string_list(raw: Any, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    require(isinstance(raw, list), f"{name} must be a list")
    require(allow_empty or bool(raw), f"{name} must be nonempty")
    require(all(isinstance(item, str) and item for item in raw), f"{name} entries must be nonempty strings")
    require(len(set(raw)) == len(raw), f"{name} entries must be distinct")
    return tuple(raw)


def validate_registry_entries(
    raw_entries: Any,
    expected_names: tuple[str, ...],
    registry_name: str,
) -> tuple[list[str], list[str]]:
    require(isinstance(raw_entries, list), f"{registry_name} must be a list")
    require(len(raw_entries) == len(expected_names), f"{registry_name} must contain exactly the expected entries")
    entries_by_name: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(raw_entries):
        require(isinstance(entry, dict), f"{registry_name}[{index}] must be an object")
        require_keys(
            entry,
            ["name", "status", "source_target", "verification_interface", "replacement_criterion"],
            f"{registry_name}[{index}]",
        )
        name = entry["name"]
        status = entry["status"]
        source_target = entry["source_target"]
        require(isinstance(name, str) and name, f"{registry_name}[{index}].name must be nonempty")
        require(name not in entries_by_name, f"duplicate {registry_name} entry {name!r}")
        require(status in ALLOWED_ENTRY_STATUSES, f"{registry_name}[{index}].status must be one of {sorted(ALLOWED_ENTRY_STATUSES)}")
        require(source_target in REQUIRED_SOURCE_TARGETS, f"{registry_name}[{index}].source_target must be one of {sorted(REQUIRED_SOURCE_TARGETS)}")
        require(
            isinstance(entry["verification_interface"], str) and entry["verification_interface"].strip(),
            f"{registry_name}[{index}].verification_interface must be nonempty",
        )
        require(
            isinstance(entry["replacement_criterion"], str) and entry["replacement_criterion"].strip(),
            f"{registry_name}[{index}].replacement_criterion must be nonempty",
        )
        entries_by_name[name] = entry

    require(set(entries_by_name) == set(expected_names), f"{registry_name} names must match expected entries")
    exact_entries = [name for name, entry in entries_by_name.items() if entry["status"] == "exact_formula"]
    open_entries = [name for name, entry in entries_by_name.items() if entry["status"] != "exact_formula"]
    return exact_entries, open_entries


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    audit_status = validate_metadata(data)
    require_keys(data, ["chart", "variables", "principal_symbol_trace_certificate_path"], "certificate")

    chart = data["chart"]
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")
    variables = validate_string_list(data["variables"], "variables")

    trace_path = resolve_path(
        data["principal_symbol_trace_certificate_path"],
        base_dir,
        "principal_symbol_trace_certificate_path",
    )
    trace_result = trace.verify_file(trace_path)
    require(trace_result["chart"] == chart, "audit chart must match trace chart")
    require(set(trace_result["variables"]) == set(variables), "audit variables must match trace variables")
    require(not trace_result["proves_kerr_gravitational_density"], "trace result unexpectedly proves q_grav")

    primitive_exact, primitive_open = validate_registry_entries(
        data.get("primitive_derivation_registry"),
        tuple(trace_result["checked_primitives"]),
        "primitive_derivation_registry",
    )
    obligation_exact, obligation_open = validate_registry_entries(
        data.get("backend_obligation_registry"),
        tuple(trace_result["remaining_exact_derivation_obligations"]),
        "backend_obligation_registry",
    )

    exact_derivation_complete = not primitive_open and not obligation_open
    if audit_status == "exact_derivation_complete":
        require(
            exact_derivation_complete,
            "metadata.audit_status exact_derivation_complete requires all primitive and obligation entries to be exact_formula",
        )

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "audit_status": audit_status,
        "exact_derivation_complete": exact_derivation_complete,
        "proves_kerr_gravitational_density": False,
        "chart": chart,
        "variables": list(variables),
        "principal_symbol_trace_certificate": str(trace_path),
        "local_coercivity_delta": trace_result["local_coercivity_delta"],
        "primitive_entries_exact": primitive_exact,
        "primitive_entries_open": primitive_open,
        "backend_obligations_exact": obligation_exact,
        "backend_obligations_open": obligation_open,
        "checked_identities": [
            "all_trace_primitives_registered_for_exact_derivation",
            "all_backend_obligations_registered_for_exact_derivation",
        ]
        + trace_result["checked_identities"],
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
        print(f"Carter exact-derivation audit rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
