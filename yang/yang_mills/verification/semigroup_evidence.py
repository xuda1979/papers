"""semigroup_evidence.py

Artifact schema + verifier for semigroup convergence evidence.

Why this exists
---------------
To apply the (math-only) gap-transfer lemma in `functional_analysis_gap_transfer.py`,
we ultimately need *Yang--Mills-specific* estimates of the form:

(A) Uniform decay for approximants on the vacuum complement: ||T_a(t)v|| <= exp(-m t)||v||.
(B) Uniform closeness at some t0: sup_{v \perp \Omega, ||v||=1} ||(T_a(t0)-T(t0))v|| <= delta.

This module does NOT produce those estimates. It provides a machine-verifiable
place to record them as an artifact, including hashes for provenance.

Contract
--------
- `load_semigroup_evidence(path)` loads JSON.
- `verify_semigroup_evidence(doc)` validates shape and basic inequalities.
- `audit_semigroup_evidence(path)` returns an audit record suitable for
  `verify_continuum_limit.py`.

Status logic
------------
- PASS only if the artifact exists and passes validation.
- CONDITIONAL if missing (default today).
  Rationale: missing evidence is a theorem-boundary / not-yet-implemented item,
  not a logical contradiction.
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


def default_semigroup_evidence_path() -> str:
    return os.path.join(os.path.dirname(__file__), "semigroup_evidence.json")


def load_semigroup_evidence(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as e:
        return None, f"read_error:{e}"


def verify_semigroup_evidence(doc: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(doc, dict):
        return False, "not_a_dict"

    if doc.get("schema") != "yangmills.semigroup_evidence.v1":
        return False, "bad_schema"

    m = doc.get("m_approx")
    t0 = doc.get("t0")
    delta = doc.get("delta")

    try:
        m = float(m)
        t0 = float(t0)
        delta = float(delta)
    except Exception:
        return False, "m_t0_delta_not_numeric"

    if not (m > 0.0):
        return False, "m_not_positive"
    if not (t0 > 0.0):
        return False, "t0_not_positive"
    if not (delta >= 0.0):
        return False, "delta_negative"

    # Minimal feasibility check: must have q < 1.
    # This exactly matches the sufficient condition used in the gap-transfer lemma.
    import math

    q = delta + math.exp(-m * t0)
    if not (q < 1.0):
        return False, "insufficient_decay_or_convergence"

    return True, "ok"


def audit_semigroup_evidence(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = default_semigroup_evidence_path()

    doc, err = load_semigroup_evidence(path)
    sha = _maybe_sha256(path) if doc is not None else None

    if doc is None:
        return {
            "key": "semigroup_evidence_present",
            "title": "Semigroup convergence evidence artifact",
            "status": "CONDITIONAL",
            "detail": f"Missing semigroup evidence artifact at {path} ({err})",
            "artifact": {"path": os.path.abspath(path), "sha256": sha, "schema": "yangmills.semigroup_evidence.v1"},
        }

    ok, reason = verify_semigroup_evidence(doc)
    return {
        "key": "semigroup_evidence_present",
        "title": "Semigroup convergence evidence artifact",
        "status": "PASS" if ok else "FAIL",
        "detail": reason,
        "artifact": {"path": os.path.abspath(path), "sha256": sha, "schema": doc.get("schema")},
        "evidence": {"m_approx": doc.get("m_approx"), "t0": doc.get("t0"), "delta": doc.get("delta")},
    }
