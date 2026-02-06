"""verify_continuum_limit.py

Continuum limit / identification audit gate.

Goal
----
The Clay problem is a *continuum* statement: existence of a 4D Yang--Mills QFT
satisfying the axioms and possessing a positive mass gap.

This repository currently proves a number of lattice / RG stability inequalities
and a positive lower bound on a lattice proxy (e.g. LSI constant / transfer
matrix bound), but it does *not* yet implement the functional-analytic bridge
that identifies the lattice theory with a continuum Wightman/OS QFT and
transfers the spectral gap to the continuum Hamiltonian.

This module makes that gap explicit in machine-auditable form.

Contract
--------
- Returns a structured dict with fields: ok, status, reason, checks.
- Writes an artifact `continuum_limit_audit_result.json` when used via
  `audit_artifacts.write_json_artifact`.
- In strict mode (YM_STRICT=1), any theorem-boundary (CONDITIONAL) becomes FAIL.

This is intentionally conservative: it prevents "proved" messaging unless the
continuum bridge is implemented.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _is_strict_mode() -> bool:
    return os.environ.get("YM_STRICT", "0").strip().lower() in {"1", "true", "yes"}


def _load_proof_status() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "proof_status.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"clay_standard": False, "claim": "ASSUMPTION-BASED"}


def audit_continuum_limit() -> Dict[str, Any]:
    proof_status = _load_proof_status()
    clay = bool(proof_status.get("clay_standard"))
    strict = _is_strict_mode()

    try:
        from continuum_hypotheses import continuum_hypotheses
    except ImportError:
        from .continuum_hypotheses import continuum_hypotheses

    try:
        from continuum_obligations import continuum_obligations
    except ImportError:
        from .continuum_obligations import continuum_obligations

    try:
        from semigroup_hypotheses import semigroup_hypotheses_checks
    except ImportError:
        from .semigroup_hypotheses import semigroup_hypotheses_checks

    try:
        from functional_analysis_gap_transfer import audit_gap_transfer_lemma_available
    except ImportError:
        from .functional_analysis_gap_transfer import audit_gap_transfer_lemma_available

    try:
        from semigroup_evidence import audit_semigroup_evidence
    except ImportError:
        from .semigroup_evidence import audit_semigroup_evidence

    try:
        from operator_convergence_evidence import audit_operator_convergence_evidence
    except ImportError:
        from .operator_convergence_evidence import audit_operator_convergence_evidence

    try:
        from schwinger_limit_evidence import audit_schwinger_limit_evidence
    except ImportError:
        from .schwinger_limit_evidence import audit_schwinger_limit_evidence

    checks: List[Dict[str, Any]] = list(continuum_hypotheses())
    checks.extend(list(continuum_obligations()))
    checks.extend(list(semigroup_hypotheses_checks()))
    checks.append(dict(audit_gap_transfer_lemma_available()))
    checks.append(dict(audit_semigroup_evidence()))
    checks.append(dict(audit_operator_convergence_evidence()))
    checks.append(dict(audit_schwinger_limit_evidence()))

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
            status = "FAIL"
            ok = False
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
    res = audit_continuum_limit()
    print("=" * 60)
    print("CONTINUUM LIMIT / IDENTIFICATION AUDIT")
    print("=" * 60)
    print(f"status: {res['status']}")
    for chk in res["checks"]:
        print(f"- {chk['key']}: {chk['status']}  ({chk.get('detail','')})")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
