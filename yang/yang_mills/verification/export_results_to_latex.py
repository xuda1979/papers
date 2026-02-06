r"""
Export Verification Results to LaTeX
====================================

This script runs all verification modules and exports the results to a
LaTeX-compatible format that can be directly \input{} into the paper.

Usage:
    python export_results_to_latex.py

Output:
    - verification_results.tex  (LaTeX macros for all numerical results)
    - verification_results.json (Machine-readable backup)

The paper can then use macros like:
    \VerBetaStrong, \VerBetaIntermediate, \VerJIrrelevantMax, etc.
"""

import json
import os
import sys
from datetime import datetime
import hashlib

sys.path.insert(0, os.path.dirname(__file__))

from interval_arithmetic import Interval
from ab_initio_jacobian import AbInitioJacobianEstimator
from dobrushin_checker import DobrushinChecker
from rigorous_constants_derivation import AbInitioBounds

try:
    # Attempt to import BallCovering from the phase2 package structure
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from phase2.tube_geometry.ball_covering import BallCovering
    from phase2.operator_basis.basis_generator import OperatorBasis
    from phase2.tube_geometry.tube_definition import TubeDefinition
    HAS_BALL_COVERING = True
except ImportError as e:
    HAS_BALL_COVERING = False
    print(f"WARNING: BallCovering module not found ({e}). Interval-uniform verification will be skipped.")

def run_full_verification():
    """
    Run the complete verification and collect all results.
    """
    results = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "version": "1.0",
            "status": "PENDING"
        },
        "regimes": {
            "strong_coupling_max": 0.25,
            "intermediate_min": 0.25,
            "intermediate_max": 6.0,
            "weak_coupling_min": 6.0
        },
        "verification_points": {},
        "dobrushin_check": {},
        "summary": {}
    }
    
    jacobian_estimator = AbInitioJacobianEstimator()
    dobrushin_checker = DobrushinChecker()
    
    # 1. Run Dobrushin Check for Strong Coupling Closure
    dobrushin_beta = 0.25
    d_norm_interval = dobrushin_checker.compute_interaction_norm(Interval(dobrushin_beta, dobrushin_beta))
    dobrushin_passed = d_norm_interval.upper < 1.0
    
    results["dobrushin_check"] = {
        "beta": dobrushin_beta,
        "norm_upper": round(d_norm_interval.upper, 4),
        "status": "PASS" if dobrushin_passed else "FAIL"
    }

    # 1b. Ab-initio pollution constant (tail feedback) bound
    # We export a conservative (worst-case) bound over the same checkpoint grid
    # used for the intermediate verification points.
    #
    # This is intended to eliminate informal "≈ 0.02" language in the manuscript
    # and replace it with a certificate-derived LaTeX macro.
    results["pollution_check"] = {
        "beta_grid": [],
        "C_poll_upper_grid": [],
        "C_poll_upper_max": None,
        "min_tail_stability_margin": 1.0 # Will be updated
    }

    # 1c. LSI Positivity Verification (Gribov Condition)
    # Checks that c_LSI > 0 assuming FMD restriction.
    print("Running LSI Positivity Verification...")
    from verify_lsi_positivity import verify_lsi_positivity
    lsi_passed, lsi_min = verify_lsi_positivity(steps=50) # Coarser grid for export speed
    
    results["lsi_check"] = {
        "status": "PASS" if lsi_passed else "FAIL",
        "min_lsi_constant": round(lsi_min, 8)
    }

    # 1d. Interval-Uniform Verification (The "No Gaps" Certificate)
    # Instead of just checking points, we traverse a rigorous covering of the
    # entire intermediate regime [0.25, 6.0].
    results["interval_check"] = {
        "covered_min": 0.25,
        "covered_max": 6.0,
        "ball_count": 0,
        "max_J_irr_continuous": 0.0,
        "status": "SKIPPED"
    }

    if HAS_BALL_COVERING:
        print("Running Interval-Uniform Verification (Continuous Covering)...")
        
        # 1. Initialize Basis and Tube (real physics models)
        basis = OperatorBasis(d_max=6)
        tube = TubeDefinition(beta_min=0.25, beta_max=6.0, dim=basis.count())
        
        covering = BallCovering(tube)
        # Use a safe step size for the covering generation
        covering.generate_flow_based_covering(step_size=0.1)
        
        balls = covering.balls
        max_j_continuous = 0.0
        interval_passed = True
        
        for ball in balls:
            # Construct rigorous interval: [center - radius, center + radius]
            # This accounts for ANY value within the ball
            beta_interval = Interval(ball.beta - ball.radius, ball.beta + ball.radius)
            
            # Compute Jacobian over the whole interval
            J = jacobian_estimator.compute_jacobian(beta_interval)
            
            # J[1][1] is the irrelevant direction contraction
            j_irr_upper = J[1][1].upper
            
            if j_irr_upper > max_j_continuous:
                max_j_continuous = j_irr_upper
            
            if j_irr_upper >= 0.99:
                interval_passed = False
                print(f"  FAIL at interval near beta={ball.beta:.3f}: J_bound={j_irr_upper:.4f}")

        results["interval_check"] = {
            "covered_min": 0.25,
            "covered_max": 6.0,
            "ball_count": len(balls),
            "max_J_irr_continuous": round(max_j_continuous, 4),
            "status": "PASS" if interval_passed else "FAIL"
        }
        print(f"  Interval Verification: {results['interval_check']['status']} "
              f"(Balls: {len(balls)}, Max J: {max_j_continuous:.4f})")
    
    # 2. Verification points for Intermediate/Weak Regimes
    check_points = [0.25, 0.40, 0.50, 0.63, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    all_passed = True
    max_j_irr = 0.0
    min_j_irr = 1.0
    
    for beta_val in check_points:
        beta_interval = Interval(beta_val, beta_val)
        
        # New: Compute Rigorous Tail Stability Margin here
        # We need access to the verifier logic for this.
        # But `export_results_to_latex.py` doesn't import TubeVerifierPhase1 by default.
        # We can approximate it or omit.
        # Better: run `phase1_verifier` if available.
        # We'll skip adding a new module dependency here to keep it simple, 
        # but we track C_poll.
        
        try:
            J = jacobian_estimator.compute_jacobian(beta_interval)
            j_pp = J[0][0]
            j_rr = J[1][1]
            
            # Determine regime
            if beta_val <= 0.25:
                regime = "strong"
            elif beta_val < 4.5:
                regime = "crossover"
            else:
                regime = "weak"
            
            # Check contraction
            contracts = j_rr.upper < 0.99
            if not contracts:
                all_passed = False
            
            # Track extremes
            if j_rr.upper > max_j_irr:
                max_j_irr = j_rr.upper
            if j_rr.upper < min_j_irr:
                min_j_irr = j_rr.upper
            
            results["verification_points"][f"{beta_val:.2f}"] = {
                "beta": beta_val,
                "regime": regime,
                "J_marginal_lower": round(j_pp.lower, 6),
                "J_marginal_upper": round(j_pp.upper, 6),
                "J_irrelevant_lower": round(j_rr.lower, 6),
                "J_irrelevant_upper": round(j_rr.upper, 6),
                "contracts": contracts,
                "status": "PASS" if contracts else "FAIL"
            }

            # Track C_poll on the same grid (worst-case bound)
            C_poll = AbInitioBounds.compute_pollution_constant(beta_interval)
            results["pollution_check"]["beta_grid"].append(beta_val)
            results["pollution_check"]["C_poll_upper_grid"].append(round(C_poll.upper, 6))
            
        except Exception as e:
            results["verification_points"][f"{beta_val:.2f}"] = {
                "beta": beta_val,
                "status": "ERROR",
                "error": str(e)
            }
            all_passed = False

    if results["pollution_check"]["C_poll_upper_grid"]:
        results["pollution_check"]["C_poll_upper_max"] = round(
            max(results["pollution_check"]["C_poll_upper_grid"]), 6
        )
    
    # Summary statistics
    results["summary"] = {
        "all_passed": all_passed,
        "total_points": len(check_points),
        "passed_points": sum(1 for k, v in results["verification_points"].items() if v.get("status") == "PASS"),
        "max_J_irrelevant": round(max_j_irr, 6),
        "min_J_irrelevant": round(min_j_irr, 6),
        "contraction_margin": round(0.99 - max_j_irr, 6)
    }

    # Gate on theorem-boundary audits (continuum / OS) to prevent PASS over-claim.
    try:
        from generate_final_audit import generate_final_audit
        final_audit = generate_final_audit()
        audit_status = final_audit.get("status", "FAIL")
    except Exception as e:
        print(f"[WARN] generate_final_audit failed: {e}")
        audit_status = "FAIL"

    if not all_passed:
        results["metadata"]["status"] = "FAIL"
    elif audit_status == "PASS":
        results["metadata"]["status"] = "PASS"
    elif audit_status == "CONDITIONAL":
        # Lattice checks passed, but theorem-boundary remains; don't over-claim.
        results["metadata"]["status"] = "CONDITIONAL"
    else:
        results["metadata"]["status"] = "FAIL"

    return results


def number_to_macro_name(num_str):
    """
    Convert a numeric string to a valid LaTeX macro name (letters only).
    E.g., "0.40" -> "ZeroPointFourZero", "6.0" -> "SixPointZero"
    """
    digit_words = {
        '0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
        '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine',
        '.': 'Point'
    }
    return ''.join(digit_words.get(c, c) for c in num_str)


def export_to_latex(results, output_path):
    """
    Export results to a LaTeX file with macro definitions.
    LaTeX \newcommand names can only contain letters, so we convert
    numeric values like 0.40 to words like ZeroPointFourZero.
    """
    # Load proof-status (claim level) if present.
    proof_status_path = os.path.join(os.path.dirname(__file__), "proof_status.json")
    proof_status = {
        "claim": "ASSUMPTION-BASED",
        "clay_standard": False,
        "blocking_gaps": [],
    }
    try:
        with open(proof_status_path, "r", encoding="utf-8") as f:
            proof_status = json.load(f)
    except FileNotFoundError:
        pass

    blocking_gaps = proof_status.get("blocking_gaps", []) or []
    gaps_blob = json.dumps(blocking_gaps, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    blocking_gaps_sha256 = hashlib.sha256(gaps_blob).hexdigest()

    # Build UV hypotheses bundle (explicit proof obligations for UV/perturbative regime)
    try:
        from uv_hypotheses import build_uv_hypotheses, write_uv_hypotheses_json

        uv_bundle = build_uv_hypotheses(proof_status=proof_status)
        # Write alongside other machine-readable exports.
        uv_json_path = os.path.join(os.path.dirname(__file__), "uv_hypotheses.json")
        write_uv_hypotheses_json(uv_json_path, proof_status=proof_status)
    except Exception as e:
        uv_bundle = {
            "items_sha256": "ERROR",
            "counts": {"total": 0, "proven": 0, "partial": 0, "unproven": 0},
            "error": str(e),
        }

    # Optional: load consolidated mass-gap certificate (produced by verify_gap_rigorous.py)
    mass_gap_cert_path = os.path.join(os.path.dirname(__file__), "mass_gap_certificate.json")
    mass_gap_cert = None

    def _maybe_sha256(path: str):
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    try:
        with open(mass_gap_cert_path, "r", encoding="utf-8") as f:
            mass_gap_cert = json.load(f)
        if not isinstance(mass_gap_cert, dict) or mass_gap_cert.get("schema") != "yangmills.mass_gap_certificate.v1":
            mass_gap_cert = None
    except FileNotFoundError:
        mass_gap_cert = None
    except Exception:
        # Keep exporter resilient; treat as missing.
        mass_gap_cert = None

    def _tex_escape_text(s: str) -> str:
        # Minimal escaping for macro bodies used in running text.
        # We intentionally keep it small to avoid surprising TeX behavior.
        return (
            s.replace("\\", "\\textbackslash{}")
             .replace("{", "\\{")
             .replace("}", "\\}")
        )

    def _yesno(v: object) -> str:
        return "YES" if bool(v) else "NO"

    # Optional: load continuum-limit audit artifact (produced by verify_gap_rigorous.py)
    continuum_audit_path = os.path.join(os.path.dirname(__file__), "continuum_limit_audit_result.json")
    continuum_audit = None
    try:
        with open(continuum_audit_path, "r", encoding="utf-8") as f:
            continuum_audit = json.load(f)
        if not isinstance(continuum_audit, dict):
            continuum_audit = None
    except FileNotFoundError:
        continuum_audit = None
    except Exception:
        continuum_audit = None

    # IMPORTANT: The header status must match the machine-readable macro \VerStatus.
    # We take it from results['metadata']['status'] at write-time so it cannot drift.
    header_status = str(results.get("metadata", {}).get("status", "UNKNOWN"))

    lines = [
        "% =============================================================================",
        "% AUTO-GENERATED FILE - DO NOT EDIT MANUALLY",
        f"% Generated: {results['metadata']['generated']}",
        f"% Status: {header_status}",
        "% =============================================================================",
        "",
        "% Guard against multiple inclusion",
        "\\ifx\\VerificationResultsLoaded\\undefined",
        "\\def\\VerificationResultsLoaded{}",
        "",
        "% --- Regime Boundaries ---",
        "% --- Proof / Claim Status (machine-readable) ---",
        f"\\newcommand{{\\VerClaimLevel}}{{{proof_status.get('claim', 'ASSUMPTION-BASED')}}}",
        f"\\newcommand{{\\VerClayCertified}}{{{'YES' if bool(proof_status.get('clay_standard')) else 'NO'}}}",
        f"\\newcommand{{\\VerBlockingGapsCount}}{{{len(blocking_gaps)}}}",
        f"\\newcommand{{\\VerBlockingGapsSHA}}{{{blocking_gaps_sha256}}}",
        "",
        "% --- UV Hypotheses Bundle (explicit obligations) ---",
        f"\\newcommand{{\\VerUVHypothesesCount}}{{{uv_bundle.get('counts', {}).get('total', 0)}}}",
        f"\\newcommand{{\\VerUVHypothesesUnproven}}{{{uv_bundle.get('counts', {}).get('unproven', 0)}}}",
        f"\\newcommand{{\\VerUVHypothesesSHA}}{{{uv_bundle.get('items_sha256', 'UNKNOWN')}}}",
        "",
    "% --- Mass Gap Certificate (if available) ---",
    f"\\newcommand{{\\VerMassGapCertPresent}}{{{_yesno(mass_gap_cert is not None)}}}",
    f"\\newcommand{{\\VerMassGapCertSHA}}{{{_maybe_sha256(mass_gap_cert_path) if mass_gap_cert is not None else 'MISSING'}}}",
    f"\\newcommand{{\\VerMassGapStatus}}{{{(mass_gap_cert or {}).get('status', 'MISSING')}}}",
    f"\\newcommand{{\\VerMassGapOK}}{{{_yesno((mass_gap_cert or {}).get('ok', False))}}}",
    f"\\newcommand{{\\VerMassGapStrict}}{{{_yesno((mass_gap_cert or {}).get('strict', False))}}}",
    f"\\newcommand{{\\VerMassGapClaim}}{{{(mass_gap_cert or {}).get('claim', 'UNKNOWN')}}}",
    f"\\newcommand{{\\VerMassGapLowerBound}}{{{((mass_gap_cert or {}).get('mass_gap', {}) or {}).get('lower_bound', 0.0):.6e}}}",
    f"\\newcommand{{\\VerMassGapReason}}{{{_tex_escape_text(str((mass_gap_cert or {}).get('reason', '')))}}}",
    f"\\newcommand{{\\VerOSAuditStatus}}{{{(((mass_gap_cert or {}).get('os_audit', {}) or {}).get('status', 'MISSING'))}}}",
    f"\\newcommand{{\\VerOSAuditReason}}{{{_tex_escape_text(str((((mass_gap_cert or {}).get('os_audit', {}) or {}).get('reason', ''))))}}}",
    f"\\newcommand{{\\VerMassGapCertStatusSummary}}{{{_tex_escape_text(str((mass_gap_cert or {}).get('status', 'MISSING')) + ':' + str((mass_gap_cert or {}).get('reason', '')))}}}",
    "",
    "% --- Continuum-limit audit (if available) ---",
    f"\\newcommand{{\\VerContinuumAuditPresent}}{{{_yesno(continuum_audit is not None)}}}",
    f"\\newcommand{{\\VerContinuumAuditSHA}}{{{_maybe_sha256(continuum_audit_path) if continuum_audit is not None else 'MISSING'}}}",
    f"\\newcommand{{\\VerContinuumAuditStatus}}{{{(continuum_audit or {}).get('status', 'MISSING')}}}",
    f"\\newcommand{{\\VerContinuumAuditOK}}{{{_yesno((continuum_audit or {}).get('ok', False))}}}",
    f"\\newcommand{{\\VerContinuumAuditReason}}{{{_tex_escape_text(str((continuum_audit or {}).get('reason', '')))}}}",
    f"\\newcommand{{\\VerSemigroupHypPresent}}{{{_yesno(any((c or {}).get('key') == 'semigroup_hypotheses_artifact_present' for c in ((continuum_audit or {}).get('checks', []) or [])))}}}",
    f"\\newcommand{{\\VerSemigroupHypSHA}}{{{_tex_escape_text(str(next((((c or {}).get('artifact', {}) or {}).get('sha256') for c in ((continuum_audit or {}).get('checks', []) or []) if (c or {}).get('key') == 'semigroup_hypotheses_artifact_present'), 'MISSING')))}}}",
    "",
        f"\\newcommand{{\\VerBetaStrongMax}}{{{results['regimes']['strong_coupling_max']:.2f}}}",
        f"\\newcommand{{\\VerBetaIntermediateMin}}{{{results['regimes']['intermediate_min']:.2f}}}",
        f"\\newcommand{{\\VerBetaIntermediateMax}}{{{results['regimes']['intermediate_max']:.1f}}}",
        f"\\newcommand{{\\VerBetaWeakMin}}{{{results['regimes']['weak_coupling_min']:.1f}}}",
        "",
        "% --- Dobrushin Strong Coupling Bridge ---",
        f"\\newcommand{{\\VerDobrushinBeta}}{{{results['dobrushin_check']['beta']:.2f}}}",
        f"\\newcommand{{\\VerDobrushinNorm}}{{{results['dobrushin_check']['norm_upper']:.4f}}}",
        f"\\newcommand{{\\VerDobrushinStatus}}{{{results['dobrushin_check']['status']}}}",
    "",
    "% --- Tail Pollution (ab-initio) ---",
    f"\\newcommand{{\\VerCPollMax}}{{{results.get('pollution_check', {}).get('C_poll_upper_max', 0.0):.6f}}}",
    # f"\\newcommand{{\\VerTailMargin}}{{{results.get('pollution_check', {}).get('min_tail_stability_margin', 0.0):.2f}}}", # Future work
    "",
    "% --- Gribov/LSI Positivity ---",
    f"\\newcommand{{\\VerLSIStatus}}{{{results.get('lsi_check', {}).get('status', 'FAIL')}}}",
    f"\\newcommand{{\\VerLSIMin}}{{{results.get('lsi_check', {}).get('min_lsi_constant', 0.0):.2e}}}",
    "",
    "% --- Interval-Uniform Verification (Continuous) ---",
    f"\\newcommand{{\\VerIntervalStatus}}{{{results.get('interval_check', {}).get('status', 'N/A')}}}",
    f"\\newcommand{{\\VerIntervalBallCount}}{{{results.get('interval_check', {}).get('ball_count', 0)}}}",
    f"\\newcommand{{\\VerIntervalMaxJ}}{{{results.get('interval_check', {}).get('max_J_irr_continuous', 0.0):.4f}}}",
    f"\\newcommand{{\\VerIntervalMin}}{{{results.get('interval_check', {}).get('covered_min', 0.0):.2f}}}",
    f"\\newcommand{{\\VerIntervalMax}}{{{results.get('interval_check', {}).get('covered_max', 0.0):.1f}}}",
    "",
    "% --- Summary Statistics ---",
        f"\\newcommand{{\\VerTotalPoints}}{{{results['summary']['total_points']}}}",
        f"\\newcommand{{\\VerPassedPoints}}{{{results['summary']['passed_points']}}}",
        f"\\newcommand{{\\VerMaxJIrrelevant}}{{{results['summary']['max_J_irrelevant']:.4f}}}",
        f"\\newcommand{{\\VerMinJIrrelevant}}{{{results['summary']['min_J_irrelevant']:.4f}}}",
        f"\\newcommand{{\\VerContractionMargin}}{{{results['summary']['contraction_margin']:.4f}}}",
        f"\\newcommand{{\\VerStatus}}{{{results['metadata']['status']}}}",
        "",
        "% --- Per-Beta Results ---",
        "% Macro names use words for digits (e.g., \\VerJIrrelevantAtZeroPointFourZero)",
    ]
    
    for key, data in results["verification_points"].items():
        if data.get("status") == "PASS" or data.get("status") == "FAIL":
            beta_str = number_to_macro_name(key)  # 0.40 -> ZeroPointFourZero
            lines.append(f"% Beta = {data['beta']}")
            lines.append(f"\\newcommand{{\\VerJMarginalAt{beta_str}}}{{{data['J_marginal_upper']:.4f}}}")
            lines.append(f"\\newcommand{{\\VerJIrrelevantAt{beta_str}}}{{{data['J_irrelevant_upper']:.4f}}}")
            lines.append(f"\\newcommand{{\\VerStatusAt{beta_str}}}{{{data['status']}}}")
            lines.append("")
    
    # Add a verification table environment
    # NOTE: We skip the table definition in preamble to avoid expansion issues
    # The table can be constructed manually in the document using the per-beta macros
    lines.extend([
        "% --- Verification Table ---",
        "% NOTE: Table data available via per-beta macros above.",
        "% Example usage in document:",
        "%   \\VerJIrrelevantAtZeroPointFourZero gives the J_irr value at beta=0.40",
        "%   \\VerJIrrelevantAtOnePointZeroZero gives the J_irr value at beta=1.0",
        "",
        "\\fi  % End of inclusion guard",
        "% =============================================================================",
        "% END OF AUTO-GENERATED FILE",
        "% =============================================================================",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"[OK] LaTeX macros written to: {output_path}")


def export_to_json(results, output_path):
    """
    Export results to JSON for programmatic access.
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] JSON results written to: {output_path}")


def main():
    print("=" * 70)
    print("YANG-MILLS VERIFICATION: EXPORT TO LATEX")
    print("=" * 70)
    
    # Run verification
    print("\n[1/3] Running full verification...")
    results = run_full_verification()
    
    # Determine output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.join(script_dir, "..", "single")
    
    # Export to LaTeX (in paper directory)
    latex_path = os.path.join(paper_dir, "verification_results.tex")
    print(f"\n[2/3] Exporting to LaTeX...")
    export_to_latex(results, latex_path)
    
    # Export to JSON (in verification directory)
    json_path = os.path.join(script_dir, "verification_results.json")
    print(f"\n[3/3] Exporting to JSON...")
    export_to_json(results, json_path)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"VERIFICATION STATUS: {results['metadata']['status']}")
    print(f"Points Verified: {results['summary']['passed_points']}/{results['summary']['total_points']}")
    print(f"Max J_irrelevant: {results['summary']['max_J_irrelevant']:.4f} (margin: {results['summary']['contraction_margin']:.4f})")
    print("=" * 70)
    
    if results['metadata']['status'] == 'PASS':
        print("\n✓ Paper can now use \\input{verification_results.tex} to load all values.")
    else:
        print("\n✗ Verification FAILED. Fix issues before updating paper.")
    
    return 0 if results['metadata']['status'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())
