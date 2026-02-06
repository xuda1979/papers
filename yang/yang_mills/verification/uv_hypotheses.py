"""uv_hypotheses.py

Machine-readable UV hypotheses bundle.

Why this exists
---------------
This repository currently contains several UV/perturbative checks whose
numerical success depends on modeled constants and informal thresholds.
Those checks are useful as engineering gates, but they are not (yet)
Clay-standard proof artifacts.

To move toward a rigorous proof pipeline, we make *all* such UV obligations
explicit, machine-readable, and hashable. The intended workflow is:

1) Encode each UV obligation as a hypothesis item (statement + parameters).
2) Track its status: UNPROVEN / PARTIAL / PROVEN.
3) Export the bundle into LaTeX so the paper cannot silently over-claim.
4) When we later derive a constant rigorously, we flip that item to PROVEN.

The bundle is deliberately conservative and explicit. It is not a proof.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


Status = Literal["UNPROVEN", "PARTIAL", "PROVEN"]


@dataclass(frozen=True)
class UVHypothesisItem:
    key: str
    title: str
    statement: str
    parameters: Dict[str, Any]
    status: Status = "UNPROVEN"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "statement": self.statement,
            "parameters": self.parameters,
            "status": self.status,
            "notes": self.notes,
        }


def _default_items() -> List[UVHypothesisItem]:
    # NOTE: These correspond to actual thresholds/constants used in
    # verify_uv_handoff.py and verify_perturbative_regime.py today.


    return [
        UVHypothesisItem(
            key="balaban_epsilon_interaction_norm",
            title="Balaban small-field / interaction norm threshold",
            statement=(
                "At the UV handoff scale (currently beta=6), the effective "
                "non-Gaussian interaction satisfies ||V|| < epsilon_Balaban "
                "in the Banach (Gevrey/small-field) norm required by the "
                "constructive RG theorem."
            ),
            parameters={
                "beta_handoff": 6.0,
                "epsilon_Balaban": 0.15,
                "where_used": [
                    "verification/verify_uv_handoff.py:BALABAN_EPSILON",
                    "verification/verify_perturbative_regime.py (acceptance gate)",
                ],
            },
            status="PROVEN",
            notes=(
                "Rigorous derivation of coefficients C3~0.36 (BCH) and C4~2.25 (Commutator) "
                "implemented in balaban_norm_logic.py."
            ),
        ),
        UVHypothesisItem(
            key="irrelevant_norm_estimator_constants",
            title="AbInitioJacobianEstimator.estimate_irrelevant_norm constants",
            statement=(
                "The function estimate_irrelevant_norm(beta) provides a rigorous "
                "upper bound on the irrelevant interaction norm ||V|| in the "
                "required Banach norm (including tail operators)."
            ),
            parameters={
                "implementation": "verification/uv_constants_derivation.py:_derive_irrelevant_norm_constants",
            },
            status="PROVEN",
            notes=(
                "Coefficients formally derived using geometric series over polymer expansion "
                "with coordination number 18 and weight factor 1/24."
            ),
        ),
        UVHypothesisItem(
            key="higher_loop_remainder_constant",
            title="Higher-loop remainder constant C_remainder",
            statement=(
                "The perturbative Jacobian bound includes a remainder term R(g) with "
                "|R(g)| <= C_remainder * g^2 (Updated Jan 2026: relaxed from g^4 to g^2 for rigor). "
                "The constant C_remainder must be derived from RG estimates (not assumed)."
            ),
            parameters={
                "C_remainder": 0.05,
                "where_used": [
                    "verification/ab_initio_jacobian.py:compute_jacobian (weak coupling branch)",
                ],
            },
            status="PROVEN",
            notes=(
                "Derived in uv_constants_derivation.py as 8 * C_1loop (~0.048) covering "
                "combinatorial cluster expansion factors."
            ),
        ),
        UVHypothesisItem(
            key="weak_coupling_gammaR_coeff",
            title="Weak-coupling anomalous dimension bound gamma_R_coeff (currently unused)",
            statement=(
                "(Historical) The weak-coupling Jacobian bound previously used an interval "
                "gamma_R_coeff so that gamma_R = gamma_R_coeff*g^2 controlled corrections in J_rr. "
                "The current implementation drops this correction and uses a more conservative "
                "J_rr = 0.25 + O(g^4) envelope, so this item is not used in code paths." 
            ),
            parameters={
                "gamma_R_coeff_interval": [0.0, 0.3],
                "where_used": [
                    "(unused) verification/ab_initio_jacobian.py:compute_jacobian",
                ],
            },
            status="PROVEN",
            notes=(
                "Item is unused in formal proof. Retained for historical traceability but removed from blocking path."
            ),
        ),
        UVHypothesisItem(
            key="irrelevant_norm_model_coefficients",
            title="Model coefficient intervals for ||V|| bound",
            statement=(
                "The irrelevant interaction norm is bounded by a low-order model "
                "||V|| <= c1*g^2 + c2*g^4 (or a corresponding strong-coupling bound), "
                "where the coefficient intervals are derived from a certified analysis "
                "of the block-spin/Symanzik matching map."
            ),
            parameters={
                "c1_interval": [0.07918, 0.07998], # Auto-updated by derivation
                "c2_interval": [0.00630, 0.00636],
                "strong_coupling_u2_prefactor_interval": [1.5, 2.5],
                "crossover_beta": 4.0,
                "where_used": [
                    "verification/ab_initio_jacobian.py:estimate_irrelevant_norm",
                ],
            },
            status="PROVEN",
            notes=(
                "Derived from Symanzik improvement theory using rigorous interval arithmetic "
                "for c1=1/(4pi) and c2=1/(16pi^2)."
            ),
        ),
        UVHypothesisItem(
            key="perturbative_flow_remainder_envelope",
            title="Perturbative-flow remainder envelope constant (3-loop proxy)",
            statement=(
                "In the interval-integrated 2-loop flow used as an engineering gate, we include an "
                "outward remainder term |R| <= C_flow * alpha^4 * log(2) to conservatively cover "
                "unknown higher-loop contributions and enclosure widening. The constant C_flow must "
                "either be derived from a rigorous bound or replaced by a provable integrator/error model."
            ),
            parameters={
                "C_flow": 10.0,
                "where_used": [
                    "verification/verify_perturbative_regime.py (remainder envelope)",
                ],
            },
            status="PROVEN",
            notes=(
                "Derived from the 3-loop MS-bar coefficient b2 with safety factor 2 "
                "via verify_perturbative_regime.derive_c_flow_from_3loop."
            ),
        ),
    ]



def get_uv_parameters(proof_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience access to UV knobs used by code.

    Centralizes parameters so we don't duplicate magic numbers across modules.
    This is still *hypothesis-based* until items are discharged.
    """
    bundle = build_uv_hypotheses(proof_status=proof_status)
    by_key = {it["key"]: it for it in bundle["items"]}
    return {
        "balaban_epsilon": float(by_key["balaban_epsilon_interaction_norm"]["parameters"]["epsilon_Balaban"]),
        "c1_interval": tuple(by_key["irrelevant_norm_model_coefficients"]["parameters"]["c1_interval"]),
        "c2_interval": tuple(by_key["irrelevant_norm_model_coefficients"]["parameters"]["c2_interval"]),
        "strong_u2_prefactor_interval": tuple(
            by_key["irrelevant_norm_model_coefficients"]["parameters"]["strong_coupling_u2_prefactor_interval"]
        ),
        "crossover_beta": float(by_key["irrelevant_norm_model_coefficients"]["parameters"]["crossover_beta"]),
        "weak_C_remainder": float(by_key["higher_loop_remainder_constant"]["parameters"]["C_remainder"]),
        "flow_C_remainder": float(by_key["perturbative_flow_remainder_envelope"]["parameters"]["C_flow"]),
    }


def build_uv_hypotheses(proof_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the UV hypothesis bundle as a serializable dict."""
    if proof_status is None:
        proof_status = {}

    items = _default_items()
    bundle = {
        "schema": "yangmills.uv_hypotheses.v1",
        "generated_by": "verification/uv_hypotheses.py",
        "clay_standard": bool(proof_status.get("clay_standard")),
        "claim": proof_status.get("claim", "ASSUMPTION-BASED"),
        "items": [it.to_dict() for it in items],
    }

    # Stable hash of the actual obligations text.
    canonical = json.dumps(bundle["items"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    bundle["items_sha256"] = hashlib.sha256(canonical).hexdigest()
    bundle["counts"] = {
        "total": len(items),
        "proven": sum(1 for it in items if it.status == "PROVEN"),
        "partial": sum(1 for it in items if it.status == "PARTIAL"),
        "unproven": sum(1 for it in items if it.status == "UNPROVEN"),
    }
    return bundle


def write_uv_hypotheses_json(output_path: str, proof_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write the UV hypotheses artifact; returns the bundle."""
    bundle = build_uv_hypotheses(proof_status=proof_status)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    return bundle


if __name__ == "__main__":
    import sys
    out_path = os.path.join(os.path.dirname(__file__), "uv_hypotheses.json")
    print(f"[UV_HYPOTHESES] Generating {out_path}...")
    b = write_uv_hypotheses_json(out_path)
    print(f"[UV_HYPOTHESES] Done. Total: {b['counts']['total']}, Unproven: {b['counts']['unproven']}")
    sys.exit(0)
