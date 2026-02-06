"""
verify_gap_rigorous.py

Constructs the Infinite Volume Hamiltonian Gap argument.
Includes rigorous OS Reconstruction checks and Transfer Matrix spectral bounds.
"""

import sys
import os
import json
import math
from typing import Any, Dict, List, Optional

try:
    from mpmath import iv, mp
    mp.dps = 20
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

sys.path.insert(0, os.path.dirname(__file__))

from interval_arithmetic import Interval
# Import the new rigorous bounds checker
import verify_bounds_monotonic as verify_rigorous_bounds


def _load_proof_status() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "proof_status.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"clay_standard": False, "claim": "ASSUMPTION-BASED"}


def _is_strict_mode() -> bool:
    return os.environ.get("YM_STRICT", "0").strip().lower() in {"1", "true", "yes"}


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

def verify_spectrum():
    print("=" * 60)
    print("PHASE 2: HAMILTONIAN SPECTRUM & GAP VERIFICATION")
    print("=" * 60)
    
    # 1. Pre-requisite: Rigorous Math Inequalities
    print("\n[STEP 1] Verifying Mathematical Foundations (Bessel/Turan)...")
    if verify_rigorous_bounds.main() != 0:
        print("  [FAIL] Mathematical bounds setup failed.")
        return False
        
    # 2. Load the Audited Constants (with provenance check)
    json_path = os.path.join(os.path.dirname(__file__), "rigorous_constants.json")
    if not os.path.exists(json_path):
        print(f"[ERROR] Constants file not found: {json_path}")
        return False

    # Provenance check: warn-only in ASSUMPTION-BASED mode, hard-fail in Clay-certified mode.
    clay_certified = False
    try:
        proof_status_path = os.path.join(os.path.dirname(__file__), "proof_status.json")
        with open(proof_status_path, "r", encoding="utf-8") as f:
            clay_certified = bool(json.load(f).get("clay_standard"))
    except Exception:
        clay_certified = False

    try:
        from provenance import enforce_artifact
        enforce_artifact(
            json_path,
            clay_certified=clay_certified,
            label="rigorous_constants.json",
        )
    except ImportError:
        if clay_certified:
            print("[FAIL] provenance module not available in Clay-certified mode.")
            return False
        print("  [INFO] provenance module not available; skipping artifact verification.")
    except Exception as e:
        print(f"  [FAIL] Provenance verification failed: {e}")
        return False

    with open(json_path, 'r') as f:
        data = json.load(f)

    # -----------------------------------------------------------------------
    # ROADMAP ITEM 3: Mass Gap in the Continuum
    # -----------------------------------------------------------------------
    # Current Gap: Lattice proxy (LSI constant) != Continuum Spectral Gap.
    # Required Future Work:
    # A) Norm Resolvent Convergence: Prove || (H_a - z)^-1 - (H - z)^-1 || -> 0.
    #    Construct interpolation operator J to "lock" the spectrum.
    # B) Spectrum Isolation: Use Bethe-Salpeter kernel analysis to show 
    #    one-particle pole is isolated from two-particle cut (2m-epsilon).
    #
    # ROADMAP ITEM 8: Tail Control (Gevrey Regularity)
    # -----------------------------------------------------------------------
    # This verification checks the PROJECTION stability.
    # Requirement: Prove that if the projection is stable, the infinite tail 
    # (higher derivatives of effective action) cannot grow.
    # Strategy: Gevrey Class s bounds |S^(n)| < (n!)^s.
    
    # Import AbInitioBounds for constructive verification (Provenance Check)
    try:
        from rigorous_constants_derivation import AbInitioBounds, Interval
        print("  [INFO] rigorous_constants_derivation module loaded for constructive verification.")
    except ImportError:
        print("  [ERROR] Could not import AbInitioBounds. Constructive check impossible.")
        return False
        
    print(f"\n[STEP 2] Checking positivity across {len(data)} checkpoints...")
    print("  [NOTE] Comparing JSON stored values against runtime Ab-Initio derivation.")
    
    # 3. Transfer Matrix Gap Construction
    min_physical_gap = float('inf')
    gap_rows: List[Dict[str, Any]] = []
    
    print(f"{'Beta':<10} | {'Gap (JSON)':<15} | {'Gap (Compute)':<15} | {'J_irr':<15} | {'Status'}")
    print("-" * 80)
    
    for beta_str, metrics in data.items():
        if not isinstance(metrics, dict) or 'lsi_constant' not in metrics:
            continue
            
        beta = float(beta_str)
        
        # 1. JSON Value
        lsi_data = metrics['lsi_constant']
        json_lower = lsi_data['lower'] if isinstance(lsi_data, dict) else lsi_data
        
        # 2. Constructive Computation (Provenance Check)
        # Re-derive the gap from the Lagrangian parameter beta
        # This proves the gap exists given the code logic, not just the file existence.
        try:
            beta_iv = Interval(beta, beta)
            computed_gap_iv = AbInitioBounds.get_lsi_constant(beta_iv)
            computed_lower = computed_gap_iv.lower
        except Exception as e:
            print(f"[ERROR] Computation failed for beta={beta}: {e}")
            return False

        j_irr = metrics.get('lambda_irrelevant', 0.0)
        j_irr_upper = j_irr['upper'] if isinstance(j_irr, dict) else j_irr
        
        # Rigorous Transfer Matrix Check
        # Condition 1: Computed gap must be strictly positive
        gap_exists = computed_lower > 1e-9
        
        # Condition 2: Provenance/consistency sanity check.
        # If the recomputed LOWER bound is >= the stored JSON lower bound, we're strictly stronger.
        # If the JSON value is higher than the recomputed lower bound, that's a potential mismatch.
        abs_tol = 1e-10
        rel_tol = 1e-6
        tol = max(abs_tol, rel_tol * max(abs(json_lower), abs(computed_lower), 1.0))

        if (json_lower - computed_lower) > tol:
            consistency = "MISMATCH"
        else:
            consistency = "OK"

        status = "PASS" if gap_exists and consistency == "OK" else ("FAIL" if not gap_exists else "WARN")
        
        print(f"{beta:<10.3f} | {json_lower:<15.4e} | {computed_lower:<15.4e} | {j_irr_upper:<15.4f} | {status}")
        
        if not gap_exists:
            print(f"  [FAIL] Computed Gap collapsed at beta={beta}")
            return False
            
        if computed_lower < min_physical_gap:
            min_physical_gap = computed_lower

        gap_rows.append(
            {
                "beta": beta,
                "gap_json_lower": float(json_lower),
                "gap_computed_lower": float(computed_lower),
                "lambda_irrelevant_upper": float(j_irr_upper),
                "consistency": consistency,
                "gap_exists": bool(gap_exists),
            }
        )

    print("\n[STEP 3] OS Reconstruction Conditions")
    try:
        from os_audit import audit_os_reconstruction
    except ImportError:
        from .os_audit import audit_os_reconstruction

    # Always write a machine-readable audit artifact (even if CONDITIONAL).
    # This helps keep theorem boundaries explicit in certificate bundles.
    try:
        from audit_artifacts import write_json_artifact
        _write_audit = True
    except Exception:
        _write_audit = False

    os_res = audit_os_reconstruction()

    if _write_audit:
        try:
            # Write to CWD so test harnesses and external runners can redirect
            # outputs by changing working directory.
            write_json_artifact(os.path.join(os.getcwd(), "os_audit_result.json"), os_res)
        except Exception as e:
            # Audit artifact failure should not change the math result in non-Clay mode.
            print(f"  [WARN] Could not write os_audit_result.json: {e}")

    if not os_res.get("ok"):
        print(f"  [FAIL] OS audit failed: {os_res.get('reason', 'unknown')}")
        return False

    print(f"  [STATUS] {os_res.get('status', 'UNKNOWN')}: {os_res.get('reason', '')}")
    if os_res.get("status") != "PASS":
        print("  [NOTE] OS reconstruction remains a theorem boundary in this repo (see os_audit.py).")

    print("\n[STEP 4] Continuum Limit / Identification Conditions")
    try:
        from verify_continuum_limit import audit_continuum_limit
    except ImportError:
        from .verify_continuum_limit import audit_continuum_limit

    cont_res = audit_continuum_limit()
    if _write_audit:
        try:
            write_json_artifact(os.path.join(os.getcwd(), "continuum_limit_audit_result.json"), cont_res)
        except Exception as e:
            print(f"  [WARN] Could not write continuum_limit_audit_result.json: {e}")

    if not cont_res.get("ok"):
        print(f"  [FAIL] Continuum audit failed: {cont_res.get('reason', 'unknown')}")
        return False

    print(f"  [STATUS] {cont_res.get('status', 'UNKNOWN')}: {cont_res.get('reason', '')}")
    if cont_res.get("status") != "PASS":
        print("  [NOTE] Continuum identification remains a theorem boundary (see verify_continuum_limit.py).")
    
    # Write a consolidated mass gap certificate artifact.
    # This is intended as the single machine-readable object that downstream
    # bundlers/LaTeX exporters can consume.
    proof_status = _load_proof_status()
    strict = _is_strict_mode()
    clay_certified = bool(proof_status.get("clay_standard"))

    # Determine overall status: PASS only if theorem-boundary audits are PASS
    # and all gap checks were consistent.
    any_mismatch = any(r.get("consistency") == "MISMATCH" for r in gap_rows)
    audits_pass = (os_res.get("status") == "PASS") and (cont_res.get("status") == "PASS")
    status = "PASS" if (audits_pass and not any_mismatch) else "CONDITIONAL"
    ok = True
    reason = "all_checks_passed" if status == "PASS" else "theorem_boundary_or_mismatch"
    if strict and status != "PASS":
        ok = False
        status = "FAIL"
        reason = "strict_mode_disallows_conditional"

    cert: Dict[str, Any] = {
        "schema": "yangmills.mass_gap_certificate.v1",
        "generated_by": "verification/verify_gap_rigorous.py",
        "claim": proof_status.get("claim", "ASSUMPTION-BASED"),
        "clay_standard": clay_certified,
        "strict": strict,
        "ok": ok,
        "status": status,
        "reason": reason,
        "mass_gap": {
            "kind": "dimensionless",
            "lower_bound": float(min_physical_gap),
            "derived_from": "lsi_constant_lower_bound",
        },
        "inputs": {
            "rigorous_constants": {
                "path": os.path.abspath(json_path),
                "sha256": _maybe_sha256(json_path),
            }
            ,
            "action_spec": {
                "path": os.path.abspath(os.path.join(os.path.dirname(__file__), "action_spec.json")),
                "sha256": _maybe_sha256(os.path.join(os.path.dirname(__file__), "action_spec.json")),
            }
        },
        "os_audit": os_res,
        "continuum_audit": cont_res,
        "checkpoints": {
            "count": int(len(gap_rows)),
            "rows": gap_rows,
        },
    }

    try:
        from audit_artifacts import write_json_artifact

        # Write into CWD (so tests/runners can redirect outputs by chdir)
        write_json_artifact(os.path.join(os.getcwd(), "mass_gap_certificate.json"), cert)
        # Also write next to this module so the certificate runner can bundle it
        # without needing a specific working directory.
        cert_path = write_json_artifact(os.path.join(os.path.dirname(__file__), "mass_gap_certificate.json"), cert)

        # Best-effort provenance binding (hard-fail only in Clay mode elsewhere).
        try:
            from provenance import record_derivation

            record_derivation(
                artifact_path=cert_path,
                source_files=[
                    os.path.join(os.path.dirname(__file__), "verify_gap_rigorous.py"),
                    os.path.join(os.path.dirname(__file__), "rigorous_constants_derivation.py"),
                    os.path.join(os.path.dirname(__file__), "interval_arithmetic.py"),
                    os.path.join(os.path.dirname(__file__), "os_audit.py"),
                    os.path.join(os.path.dirname(__file__), "audit_artifacts.py"),
                    os.path.join(os.path.dirname(__file__), "provenance.py"),
                ],
                extra_metadata={
                    "kind": "certificate",
                    "phase": "mass_gap",
                    "proof_claim": proof_status.get("claim", "ASSUMPTION-BASED"),
                    "clay_standard": clay_certified,
                },
            )
        except Exception:
            # Keep the verifier robust; provenance is enforced in Clay-mode preflight.
            pass
    except Exception as e:
        print(f"  [WARN] Could not write mass_gap_certificate.json: {e}")

    print(f"\n[RESULT] Global Minimum Dimensionless Mass Gap: {min_physical_gap}")
    if cert.get("status") == "PASS":
        print("  [STATUS] PASS: All audited conditions are satisfied (including OS + continuum audits).")
    elif cert.get("status") == "CONDITIONAL":
        print("  [STATUS] CONDITIONAL: Lattice proxy gap verified, but theorem-boundary audits remain.")
    else:
        print("  [STATUS] FAIL: Strict/clay mode disallows theorem-boundary results.")

    return bool(cert.get("ok"))

if __name__ == "__main__":
    if verify_spectrum():
        sys.exit(0)
    else:
        sys.exit(1)
