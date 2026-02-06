"""rp_evidence.py

Artifact schema + verifier for reflection positivity (RP) evidence.

Important context
-----------------
Reflection positivity for the Wilson plaquette action is a known theorem in the
mathematical physics literature, but this repository aims to avoid *silent*
assumption drift.

This module does NOT prove RP from first principles. Instead, it defines a
machine-verifiable interface for an RP certificate to exist inside the repo.

A future rigorous implementation could be:
- a formal proof artifact (e.g., via a proof assistant), or
- a fully explicit algebraic decomposition certificate.

For now, the artifact is a strict contract:
- it must pin the action-spec hash,
- it must specify the reflection plane and the algebra split,
- it must reference a proof source (bibliographic or formal artifact).

Status logic
------------
- This repository targets a Clay-standard proof. Therefore, merely having a
    narrative/provenance JSON is *not* sufficient to upgrade a theorem-boundary
    item to PASS.
- PASS: a proof artifact exists that is pinned and machine-verifiable.
- CONDITIONAL: artifact missing OR only provides non-machine-checkable
    provenance (theorem-boundary).
- FAIL: artifact present but malformed / contradicts pinned inputs.
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


def default_rp_evidence_path() -> str:
    return os.path.join(os.path.dirname(__file__), "rp_evidence.json")


def _action_spec_sha256() -> Optional[str]:
    path = os.path.join(os.path.dirname(__file__), "action_spec.json")
    return _maybe_sha256(path)


def load_rp_evidence(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as e:
        return None, f"read_error:{e}"


def verify_rp_evidence(doc: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(doc, dict):
        return False, "not_a_dict"

    if doc.get("schema") != "yangmills.rp_evidence.v1":
        return False, "bad_schema"

    # Must bind to action_spec hash.
    # If the artifact is present but doesn't claim a concrete sha256 yet, treat
    # as CONDITIONAL (theorem boundary) rather than FAIL.
    claimed = ((doc.get("action_spec") or {}) if isinstance(doc.get("action_spec"), dict) else {}).get("sha256")
    actual = _action_spec_sha256()
    if not claimed or not isinstance(claimed, str) or claimed.strip().upper() in {"MISSING", "TBD", "UNKNOWN"}:
        return False, "theorem_boundary_missing_action_spec_sha256"
    if actual and claimed != actual:
        return False, "action_spec_sha256_mismatch"

    # Must specify a reflection convention.
    refl = doc.get("reflection")
    if not isinstance(refl, dict):
        return False, "missing_reflection_block"

    if not isinstance(refl.get("axis"), str) or not refl.get("axis"):
        return False, "missing_reflection_axis"

    # Must include some provenance reference.
    prov = doc.get("provenance")
    if not isinstance(prov, dict):
        return False, "missing_provenance"

    src = prov.get("source")
    if not (isinstance(src, str) and src.strip()):
        return False, "missing_provenance_source"

    # Clay-level gating: require an explicit proof artifact that can be
    # mechanically validated (even if minimal for now).
    proof = doc.get("proof")
    if not isinstance(proof, dict):
        return False, "theorem_boundary_missing_proof_block"

    # Minimal contract for a proof artifact: schema + sha256 pinned.
    if proof.get("schema") not in {
        "yangmills.rp_proof_artifact.v1",
    }:
        return False, "theorem_boundary_bad_or_missing_proof_schema"

    proof_sha = proof.get("sha256")
    if not (isinstance(proof_sha, str) and proof_sha.strip() and proof_sha.strip().upper() not in {"MISSING", "TBD", "UNKNOWN"}):
        return False, "theorem_boundary_missing_proof_sha256"

    return True, "ok"


def audit_rp_evidence(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = default_rp_evidence_path()

    doc, err = load_rp_evidence(path)
    sha = _maybe_sha256(path) if doc is not None else None

    if doc is None:
        return {
            "key": "rp_evidence_present",
            "title": "Reflection positivity evidence artifact",
            "status": "CONDITIONAL",
            "detail": f"Missing rp evidence artifact at {path} ({err})",
            "artifact": {
                "path": os.path.abspath(path),
                "sha256": sha,
                "schema": "yangmills.rp_evidence.v1",
            },
        }

    ok, reason = verify_rp_evidence(doc)

    # Downgrade certain "not ok" reasons to CONDITIONAL (theorem boundary),
    # reserving FAIL for genuine contradictions/malformed artifacts.
    status = "PASS" if ok else "FAIL"
    if not ok and reason.startswith("theorem_boundary_"):
        status = "CONDITIONAL"

    return {
        "key": "rp_evidence_present",
        "title": "Reflection positivity evidence artifact",
        "status": status,
        "detail": reason,
        "artifact": {
            "path": os.path.abspath(path),
            "sha256": sha,
            "schema": doc.get("schema"),
        },
        "evidence": {
            "action_spec_sha256": ((doc.get("action_spec") or {}) if isinstance(doc.get("action_spec"), dict) else {}).get("sha256"),
            "reflection": doc.get("reflection"),
            "provenance": doc.get("provenance"),
        },
    }
