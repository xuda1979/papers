"""schwinger_limit_evidence.py

Artifact schema + verifier for continuum-limit Schwinger function evidence.

Goal
----
A Clay-standard proof needs an auditable, explicit chain that:

1) defines a family of lattice Schwinger functions (or measures) indexed by lattice spacing a,
2) proves uniform bounds (tightness/compactness) to extract a subsequential continuum limit,
3) verifies OS-relevant structural properties on the limit (as needed).

This module defines a machine-verifiable artifact interface for recording those steps.
It does NOT *prove* them. It prevents silent drift by demanding:
- action-spec pinning,
- explicit flags for what has been verified,
- provenance linking to a derivation or formal artifact.

Schema (yangmills.schwinger_limit_evidence.v1)
----------------------------------------------
Required keys:
- schema
- action_spec.sha256: must match sha256 of `verification/action_spec.json`
- family:
    index: string (e.g., "lattice_spacing")
    parameter: string (e.g., "a")
    description: string
- bounds:
    uniform_moment_bounds: bool
    tightness: bool
    subsequence_extraction: bool
    uniqueness: bool
    details: string
- invariances:
    lattice_symmetries: bool
    euclidean_invariance_in_limit: bool
    details: string
- rp_and_os:
    rp_passes_to_limit: bool
    clustering: bool
    regularity: bool
    details: string
- provenance.source: non-empty string

Status logic
------------
- Missing => CONDITIONAL.
- Present but action_spec sha is missing/TBD/UNKNOWN => CONDITIONAL.
- Present but sha mismatch / malformed => FAIL.
- Present and well-formed => PASS.

NOTE
----
PASS here only means "artifact exists and is pinned + well-formed".
Strict/clay mode should still require the boolean flags to be true for the
corresponding obligations to flip to PASS.

Clay-level gating
-----------------
Even if the JSON flags are set to true, this module reports CONDITIONAL unless
an explicit pinned proof artifact block is present. This prevents a purely
declarative file from upgrading theorem-boundary items.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple


def _maybe_sha256(path: str) -> Optional[str]:
    try:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def default_schwinger_limit_evidence_path() -> str:
    return os.path.join(os.path.dirname(__file__), "schwinger_limit_evidence.json")


def _action_spec_sha256() -> Optional[str]:
    path = os.path.join(os.path.dirname(__file__), "action_spec.json")
    return _maybe_sha256(path)


def load_schwinger_limit_evidence(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as e:
        return None, f"read_error:{e}"


def verify_schwinger_limit_evidence(doc: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(doc, dict):
        return False, "not_a_dict"

    if doc.get("schema") != "yangmills.schwinger_limit_evidence.v1":
        return False, "bad_schema"

    action = doc.get("action_spec")
    if not isinstance(action, dict):
        return False, "missing_action_spec_block"

    claimed = action.get("sha256")
    actual = _action_spec_sha256()

    if not claimed or not isinstance(claimed, str) or claimed.strip().upper() in {"MISSING", "TBD", "UNKNOWN"}:
        return False, "theorem_boundary_missing_action_spec_sha256"

    if actual and claimed != actual:
        return False, "action_spec_sha256_mismatch"

    fam = doc.get("family")
    if not isinstance(fam, dict):
        return False, "missing_family"

    for k in ["index", "parameter", "description"]:
        if not (isinstance(fam.get(k), str) and fam.get(k).strip()):
            return False, f"missing_family_field:{k}"

    bounds = doc.get("bounds")
    inv = doc.get("invariances")
    rp = doc.get("rp_and_os")
    for blk, name in [(bounds, "bounds"), (inv, "invariances"), (rp, "rp_and_os")]:
        if not isinstance(blk, dict):
            return False, f"missing_block:{name}"
        if not (isinstance(blk.get("details"), str) and blk.get("details").strip()):
            return False, f"missing_details:{name}"

    # Validate required bool flags exist (but do not require them to be True here).
    required_bool_paths = [
        ("bounds", "uniform_moment_bounds"),
        ("bounds", "tightness"),
        ("bounds", "subsequence_extraction"),
        ("bounds", "uniqueness"),
        ("invariances", "lattice_symmetries"),
        ("invariances", "euclidean_invariance_in_limit"),
        ("rp_and_os", "rp_passes_to_limit"),
        ("rp_and_os", "clustering"),
        ("rp_and_os", "regularity"),
    ]

    for block_key, flag_key in required_bool_paths:
        blk = doc.get(block_key)
        if not isinstance(blk, dict) or flag_key not in blk:
            return False, f"missing_flag:{block_key}.{flag_key}"
        if not isinstance(blk.get(flag_key), bool):
            return False, f"flag_not_bool:{block_key}.{flag_key}"

    prov = doc.get("provenance")
    if not isinstance(prov, dict):
        return False, "missing_provenance"
    if not (isinstance(prov.get("source"), str) and prov.get("source").strip()):
        return False, "missing_provenance_source"

    proof = doc.get("proof")
    if not isinstance(proof, dict):
        return False, "theorem_boundary_missing_proof_block"

    if proof.get("schema") not in {"yangmills.schwinger_limit_proof_artifact.v1"}:
        return False, "theorem_boundary_bad_or_missing_proof_schema"

    proof_sha = proof.get("sha256")
    if not (isinstance(proof_sha, str) and proof_sha.strip() and proof_sha.strip().upper() not in {"MISSING", "TBD", "UNKNOWN"}):
        return False, "theorem_boundary_missing_proof_sha256"

    return True, "ok"


def audit_schwinger_limit_evidence(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = default_schwinger_limit_evidence_path()

    doc, err = load_schwinger_limit_evidence(path)
    sha = _maybe_sha256(path) if doc is not None else None

    if doc is None:
        return {
            "key": "schwinger_limit_evidence_present",
            "title": "Continuum limit Schwinger-function evidence artifact",
            "status": "CONDITIONAL",
            "detail": f"Missing schwinger limit evidence artifact at {path} ({err})",
            "artifact": {
                "path": os.path.abspath(path),
                "sha256": sha,
                "schema": "yangmills.schwinger_limit_evidence.v1",
            },
        }

    ok, reason = verify_schwinger_limit_evidence(doc)

    status = "PASS" if ok else "FAIL"
    if not ok and reason.startswith("theorem_boundary_"):
        status = "CONDITIONAL"

    return {
        "key": "schwinger_limit_evidence_present",
        "title": "Continuum limit Schwinger-function evidence artifact",
        "status": status,
        "detail": reason,
        "artifact": {
            "path": os.path.abspath(path),
            "sha256": sha,
            "schema": doc.get("schema"),
        },
        "evidence": {
            "action_spec_sha256": ((doc.get("action_spec") or {}) if isinstance(doc.get("action_spec"), dict) else {}).get("sha256"),
            "family": doc.get("family"),
            "bounds": doc.get("bounds"),
            "invariances": doc.get("invariances"),
            "rp_and_os": doc.get("rp_and_os"),
            "provenance": doc.get("provenance"),
        },
    }
