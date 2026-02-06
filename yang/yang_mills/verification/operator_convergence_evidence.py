"""operator_convergence_evidence.py

Artifact schema + verifier for Hamiltonian identification / operator convergence.

Goal
----
A Clay-standard proof needs to identify a continuum Hamiltonian H as a limit of
lattice objects (transfer matrices / Hamiltonians) as a -> 0.

This module defines a machine-verifiable artifact interface for *evidence* of
operator convergence, without claiming to prove it today.

We focus on an auditable, minimal sufficient pattern:
- either resolvent convergence at a fixed spectral parameter z,
- or semigroup convergence at one time t0,
with an explicit quantitative bound that can be used downstream.

Contract
--------
- `audit_operator_convergence_evidence()` returns a single check record.
- PASS only if the artifact exists and has a valid shape + inequalities.
- CONDITIONAL if missing (default).

NOTE
----
The numeric entries here must ultimately be backed by rigorous derivations.
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


def default_operator_convergence_evidence_path() -> str:
    return os.path.join(os.path.dirname(__file__), "operator_convergence_evidence.json")


def load_operator_convergence_evidence(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as e:
        return None, f"read_error:{e}"


def verify_operator_convergence_evidence(doc: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(doc, dict):
        return False, "not_a_dict"

    if doc.get("schema") != "yangmills.operator_convergence_evidence.v1":
        return False, "bad_schema"

    kind = doc.get("kind")
    if kind not in {"semigroup", "resolvent"}:
        return False, "kind_must_be_semigroup_or_resolvent"

    bound = doc.get("bound")
    try:
        bound = float(bound)
    except Exception:
        return False, "bound_not_numeric"

    if not (bound >= 0.0):
        return False, "bound_negative"

    if kind == "semigroup":
        try:
            t0 = float(doc.get("t0"))
        except Exception:
            return False, "t0_not_numeric"
        if not (t0 > 0.0):
            return False, "t0_not_positive"
        # No further inequality beyond nonnegativity: this artifact is an interface.
        return True, "ok"

    # resolvent
    z = doc.get("z")
    if not (isinstance(z, str) and z.strip()):
        return False, "z_must_be_nonempty_string"

    return True, "ok"


def audit_operator_convergence_evidence(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = default_operator_convergence_evidence_path()

    doc, err = load_operator_convergence_evidence(path)
    sha = _maybe_sha256(path) if doc is not None else None

    if doc is None:
        return {
            "key": "operator_convergence_evidence_present",
            "title": "Operator/semigroup convergence evidence artifact",
            "status": "CONDITIONAL",
            "detail": f"Missing operator convergence evidence artifact at {path} ({err})",
            "artifact": {
                "path": os.path.abspath(path),
                "sha256": sha,
                "schema": "yangmills.operator_convergence_evidence.v1",
            },
        }

    ok, reason = verify_operator_convergence_evidence(doc)
    return {
        "key": "operator_convergence_evidence_present",
        "title": "Operator/semigroup convergence evidence artifact",
        "status": "PASS" if ok else "FAIL",
        "detail": reason,
        "artifact": {
            "path": os.path.abspath(path),
            "sha256": sha,
            "schema": doc.get("schema"),
        },
        "evidence": {
            "kind": doc.get("kind"),
            "t0": doc.get("t0"),
            "z": doc.get("z"),
            "bound": doc.get("bound"),
        },
    }
