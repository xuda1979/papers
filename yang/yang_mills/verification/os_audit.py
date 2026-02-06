"""os_audit.py

Machine-auditable OS reconstruction status gate.

This repository currently *assumes* classical facts like:
- reflection positivity (RP) of the Wilson plaquette action,
- the standard OS reconstruction theorem for gauge theories with compact group.

Those are math theorems, but the *repo* should still report clearly what it
constructively checks vs what it assumes.

Contract
--------
- Produces a structured result dict with fields:
    ok: bool
    status: PASS | CONDITIONAL | FAIL
    checks: list of per-check records
- In strict mode (YM_STRICT=1), CONDITIONAL is treated as FAIL.

This mirrors the strict-mode pattern used in verify_perturbative_regime.py.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _maybe_sha256(path: str) -> str | None:
    try:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _load_action_spec() -> Dict[str, Any] | None:
    path = os.path.join(os.path.dirname(__file__), "action_spec.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if isinstance(doc, dict) and doc.get("schema") == "yangmills.action_spec.v1":
            doc = dict(doc)
            doc["_path"] = os.path.abspath(path)
            doc["_sha256"] = _maybe_sha256(path)
            return doc
        return None
    except Exception:
        return None


def _is_strict_mode() -> bool:
    return os.environ.get("YM_STRICT", "0").strip().lower() in {"1", "true", "yes"}


def _load_proof_status() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "proof_status.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"clay_standard": False, "claim": "ASSUMPTION-BASED"}


def _check_action_is_wilson_plaquette() -> Dict[str, Any]:
    # Bind this premise to a small, hashable action spec artifact.
    spec = _load_action_spec()
    if not spec:
        return {
            "key": "action_spec_present",
            "status": "FAIL",
            "detail": "Missing or invalid action_spec.json (expected schema yangmills.action_spec.v1).",
        }

    action = (spec.get("action") or {}) if isinstance(spec, dict) else {}
    name = action.get("name")
    group = action.get("gauge_group")
    dim = ((action.get("lattice") or {}) if isinstance(action.get("lattice"), dict) else {}).get("dimension")

    ok = (name == "wilson_plaquette") and (group == "SU(3)") and (dim == 4)
    return {
        "key": "action_is_wilson_plaquette",
        "status": "PASS" if ok else "FAIL",
        "detail": f"action_spec={spec.get('_path')} sha256={spec.get('_sha256')}",
        "action_spec": {
            "path": spec.get("_path"),
            "sha256": spec.get("_sha256"),
            "name": name,
            "gauge_group": group,
            "dimension": dim,
        },
    }


def _check_reflection_positivity_is_theorem_boundary() -> Dict[str, Any]:
    # We don't mechanically prove reflection positivity here; we declare it as a
    # theorem boundary and keep it visible.
    return {
        "key": "reflection_positivity_theorem_boundary",
        "status": "CONDITIONAL",
        "detail": (
            "Reflection positivity of Wilson plaquette action is treated as an external theorem; "
            "repo does not yet provide a machine-checkable proof artifact for RP."
        ),
    }


def _check_os_reconstruction_is_theorem_boundary() -> Dict[str, Any]:
    return {
        "key": "os_reconstruction_theorem_boundary",
        "status": "CONDITIONAL",
        "detail": (
            "OS reconstruction (Hilbert space + Hamiltonian from Euclidean correlators) is an external theorem; "
            "repo does not yet construct correlators + verify RP/cluster properties to invoke it mechanically."
        ),
    }


def audit_os_reconstruction() -> Dict[str, Any]:
    proof_status = _load_proof_status()
    clay = bool(proof_status.get("clay_standard"))
    strict = _is_strict_mode()

    checks: List[Dict[str, Any]] = []
    checks.append(_check_action_is_wilson_plaquette())

    # Prefer the granular obligation registry when available.
    try:
        from os_obligations import os_obligations
    except ImportError:
        from .os_obligations import os_obligations

    try:
        from rp_evidence import audit_rp_evidence
    except ImportError:
        from .rp_evidence import audit_rp_evidence

    try:
        from os_reconstruction_evidence import audit_os_reconstruction_evidence
    except ImportError:
        from .os_reconstruction_evidence import audit_os_reconstruction_evidence

    # Keep obligations scoped to OS/RP, but do not duplicate the action pinning check.
    for ob in os_obligations():
        if isinstance(ob, dict) and ob.get("key") == "os_action_pinned":
            continue
        checks.append(ob)

    # Add explicit RP evidence record (artifact-verified interface).
    checks.append(dict(audit_rp_evidence()))

    # Add explicit OS reconstruction evidence record (artifact-verified interface).
    checks.append(dict(audit_os_reconstruction_evidence()))

    # Aggregate
    statuses = [c["status"] for c in checks]

    if any(s == "FAIL" for s in statuses):
        status = "FAIL"
        ok = False
        reason = "one_or_more_failed"
    elif any(s == "CONDITIONAL" for s in statuses):
        status = "CONDITIONAL"
        ok = True
        reason = "theorem_boundary"
        if strict or clay:
            ok = False
            status = "FAIL"
            reason = "strict_mode_disallows_conditional"
    else:
        status = "PASS"
        ok = True
        reason = "all_checks_passed"

    return {
        "ok": ok,
        "status": status,
        "reason": reason,
        "claim": proof_status.get("claim", "ASSUMPTION-BASED"),
        "clay_standard": clay,
        "strict": strict,
        "checks": checks,
    }


def main() -> int:
    res = audit_os_reconstruction()
    print("=" * 60)
    print("OS RECONSTRUCTION AUDIT")
    print("=" * 60)
    print(f"status: {res['status']}")
    for chk in res["checks"]:
        print(f"- {chk['key']}: {chk['status']}  ({chk.get('detail','')})")

    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
