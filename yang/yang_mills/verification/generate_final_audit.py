"""generate_final_audit.py

Generates `certificate_final_audit.json` from actual audits.

Unlike the previous static file that unconditionally claimed PASS, this
generator *computes* the status from:
 - verify_continuum_limit.audit_continuum_limit()
 - os_audit.audit_os_reconstruction()
 - verify_perturbative_regime.verify_asymptotic_freedom_flow_result()

The resulting certificate_final_audit.json will have:
  - status = "PASS" only if all audits are PASS
  - status = "CONDITIONAL" if any audit is CONDITIONAL (and none FAIL)
  - status = "FAIL" otherwise
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def _load_proof_status() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "proof_status.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {"claim": "ASSUMPTION-BASED", "clay_standard": False}


def generate_final_audit() -> Dict[str, Any]:
    """Run all audits and produce a final certificate dict."""
    proof_status = _load_proof_status()
    clay = bool(proof_status.get("clay_standard"))

    # Import auditors lazily so sys.path is correct
    try:
        from verify_continuum_limit import audit_continuum_limit
    except ImportError:
        from .verify_continuum_limit import audit_continuum_limit

    try:
        from os_audit import audit_os_reconstruction
    except ImportError:
        from .os_audit import audit_os_reconstruction

    try:
        from verify_perturbative_regime import verify_asymptotic_freedom_flow_result
    except ImportError:
        from .verify_perturbative_regime import verify_asymptotic_freedom_flow_result

    # Collect individual audits
    continuum = audit_continuum_limit()
    os_rec = audit_os_reconstruction()
    uv_flow = verify_asymptotic_freedom_flow_result()

    # Combine status
    all_statuses = [
        continuum.get("status", "FAIL"),
        os_rec.get("status", "FAIL"),
        uv_flow.get("status", "FAIL"),
    ]

    if any(s == "FAIL" for s in all_statuses):
        overall_status = "FAIL"
    elif any(s == "CONDITIONAL" for s in all_statuses):
        overall_status = "CONDITIONAL"
    else:
        overall_status = "PASS"

    # Axiomatic conditions can only be asserted true if continuum/OS PASS
    mass_gap_positivity = (continuum.get("status") == "PASS") and bool(continuum.get("ok"))
    continuum_limit_exists = (continuum.get("status") == "PASS") and bool(continuum.get("ok"))
    lsi_uniformity = (os_rec.get("status") == "PASS") and bool(os_rec.get("ok"))

    cert = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": overall_status,
        "claim": proof_status.get("claim", "ASSUMPTION-BASED"),
        "clay_standard": clay,
        "phases": {
            "strong_coupling": {
                "method": "Rigorous Character Expansion (Area Law)",
                "status": os_rec.get("status", "FAIL"),
            },
            "intermediate_crossover": {
                "method": "Interval Arithmetic Tube Tracking + Jacobian Intersection",
                "status": continuum.get("status", "FAIL"),
            },
            "weak_coupling": {
                "method": "Perturbative Scaling + Balaban Remainder",
                "status": uv_flow.get("status", "FAIL"),
            },
        },
        "axiomatic_conditions": {
            "mass_gap_positivity": mass_gap_positivity,
            "continuum_limit_existence": continuum_limit_exists,
            "lsi_uniformity": lsi_uniformity,
        },
        "audit_details": {
            "continuum": {
                "status": continuum.get("status"),
                "reason": continuum.get("reason"),
                "ok": continuum.get("ok"),
            },
            "os_reconstruction": {
                "status": os_rec.get("status"),
                "reason": os_rec.get("reason"),
                "ok": os_rec.get("ok"),
            },
            "uv_flow": {
                "status": uv_flow.get("status"),
                "reason": uv_flow.get("reason"),
                "ok": uv_flow.get("ok"),
            },
        },
    }
    return cert


def write_final_audit(output_path: str | None = None) -> Dict[str, Any]:
    """Generate and write the final audit certificate to disk."""
    cert = generate_final_audit()
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "certificate_final_audit.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=4)
    print(f"[generate_final_audit] Wrote {output_path} with status={cert['status']}")
    return cert


def main() -> int:
    cert = write_final_audit()
    return 0 if cert["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
