"""semigroup_hypotheses.py

Loader utilities for the semigroup/transfer-matrix bridge obligations.

Why this exists
---------------
We want continuum-bridge assumptions to be *hash-pinned artifacts* so that
certificates/LaTeX outputs can't drift silently.

Contract
--------
- `load_semigroup_hypotheses_artifact()` returns (artifact, path).
- `semigroup_hypotheses_checks()` converts artifact items into the standard
  checklist shape consumed by audits.

Notes
-----
This does *not* prove anything; it is an auditing/traceability mechanism.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


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


def load_semigroup_hypotheses_artifact() -> Tuple[Optional[Dict[str, Any]], str]:
    path = os.path.join(os.path.dirname(__file__), "semigroup_hypotheses.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), path
    except Exception:
        return None, path


def semigroup_hypotheses_checks() -> List[Dict[str, Any]]:
    art, path = load_semigroup_hypotheses_artifact()
    sha256 = _maybe_sha256(path)

    if not art or not isinstance(art, dict):
        return [
            {
                "key": "semigroup_hypotheses_artifact_present",
                "title": "Semigroup hypotheses artifact present",
                "status": "FAIL",
                "detail": f"Missing/unreadable semigroup_hypotheses.json at {path}",
            }
        ]

    items = art.get("items")
    if not isinstance(items, list):
        return [
            {
                "key": "semigroup_hypotheses_artifact_shape",
                "title": "Semigroup hypotheses artifact schema",
                "status": "FAIL",
                "detail": "semigroup_hypotheses.json missing 'items' list",
                "artifact": {"path": path, "sha256": sha256},
            }
        ]

    checks: List[Dict[str, Any]] = [
        {
            "key": "semigroup_hypotheses_artifact_present",
            "title": "Semigroup hypotheses artifact present",
            "status": "PASS",
            "detail": "semigroup_hypotheses.json loaded",
            "artifact": {"path": path, "sha256": sha256, "schema": art.get("$schema")},
        }
    ]

    for it in items:
        if not isinstance(it, dict):
            continue
        chk = {
            "key": str(it.get("key", "")),
            "title": str(it.get("title", "")),
            "status": str(it.get("status", "CONDITIONAL")),
            "detail": str(it.get("detail", "")),
        }
        checks.append(chk)

    return checks
