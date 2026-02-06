"""verify_clay_standard.py

Clay-standard proof gate (strict mode).

This script is intentionally conservative. It is not a substitute for proving the
Yang--Mills mass gap conjecture, but it provides a single hard gate that enforces
what this repository means by "Clay-standard":

- `proof_status.json` must declare `clay_standard=true`.
- strict mode must be enabled (YM_STRICT=1).
- all audit gates must return PASS (no CONDITIONAL allowed).

It exists so automated pipelines (CI, release, paper export) can rely on one
entrypoint.

Exit codes
----------
0: all checks PASS under Clay/strict mode
1: a check failed or Clay prerequisites not met
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple


def _load_proof_status() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "proof_status.json")
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError("proof_status.json must be a JSON object")
    return doc


def _run_gate(fn, name: str) -> Tuple[bool, Dict[str, Any]]:
    res = fn()
    status = (res or {}).get("status")
    ok = bool(status == "PASS" and (res or {}).get("ok") is True)
    if not ok:
        return False, {"gate": name, "status": status, "reason": (res or {}).get("reason"), "result": res}
    return True, {"gate": name, "status": status}


def main() -> int:
    proof_status = _load_proof_status()
    if not bool(proof_status.get("clay_standard")):
        print("[CLAY] FAIL: proof_status.json clay_standard=false")
        return 1

    # Force strict gating in-process.
    os.environ["YM_STRICT"] = "1"

    try:
        from os_audit import audit_os_reconstruction
        from verify_continuum_limit import audit_continuum_limit
        from verify_gap_rigorous import verify_mass_gap
    except ImportError:
        from .os_audit import audit_os_reconstruction
        from .verify_continuum_limit import audit_continuum_limit
        from .verify_gap_rigorous import verify_mass_gap

    failures: List[Dict[str, Any]] = []

    ok, info = _run_gate(audit_os_reconstruction, "os_audit")
    if not ok:
        failures.append(info)

    ok, info = _run_gate(audit_continuum_limit, "continuum_limit_audit")
    if not ok:
        failures.append(info)

    # Mass gap gate: normalize to {ok,status,reason}.
    try:
        mg = verify_mass_gap(strict=True)
    except TypeError:
        mg = verify_mass_gap()
    mg_status = (mg or {}).get("status")
    if not (mg_status == "PASS" and (mg or {}).get("ok") is True):
        failures.append({"gate": "mass_gap", "status": mg_status, "reason": (mg or {}).get("reason"), "result": mg})

    if failures:
        print("[CLAY] FAIL: one or more Clay gates not PASS")
        for f in failures:
            print(f"- {f.get('gate')}: status={f.get('status')} reason={f.get('reason')}")
        return 1

    print("[CLAY] PASS: all Clay gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
