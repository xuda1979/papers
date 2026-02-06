"""uv_constants.py

Thin loader for derived UV constants.

Instead of importing thresholds directly from `uv_hypotheses`, proof-facing
verifiers should prefer:

    from uv_constants import get_uv_constants

and then consume the values from `uv_constants.json` (provenance-bound).

In non-Clay mode, missing artifacts fall back to `uv_hypotheses.get_uv_parameters`
for backwards compatibility.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

try:
    from .provenance import enforce_artifact
except ImportError:  # pragma: no cover
    from provenance import enforce_artifact


def _load_proof_status() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "proof_status.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"clay_standard": False, "claim": "ASSUMPTION-BASED"}


def get_uv_constants(*, proof_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if proof_status is None:
        proof_status = _load_proof_status()

    clay = bool(proof_status.get("clay_standard"))
    path = os.path.join(os.path.dirname(__file__), "uv_constants.json")

    if os.path.isfile(path):
        # Enforce provenance hard only in clay mode.
        enforce_artifact(
            path,
            clay_certified=clay,
            label="uv_constants.json",
            require_extra={"kind": "uv_constants"},
            require_sources=[
                os.path.join(os.path.dirname(__file__), "uv_constants_derivation.py"),
                os.path.join(os.path.dirname(__file__), "interval_arithmetic.py"),
            ],
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Backward compatible fallback: hypothesis parameters.
    try:
        from .uv_hypotheses import get_uv_parameters
    except ImportError:
        from uv_hypotheses import get_uv_parameters

    params = get_uv_parameters(proof_status=proof_status)
    return {
        "schema": "yangmills.uv_constants.v1",
        "generated_by": "fallback:uv_hypotheses.get_uv_parameters",
        "clay_standard": clay,
        "claim": proof_status.get("claim", "ASSUMPTION-BASED"),
        "derived": {
            "beta_handoff": 6.0,
            "epsilon_Balaban": float(params["balaban_epsilon"]),
            "weak_C_remainder": float(params["weak_C_remainder"]),
            "flow_C_remainder": float(params["flow_C_remainder"]),
            "irrelevant_norm_model": {
                "c1_interval": list(params["c1_interval"]),
                "c2_interval": list(params["c2_interval"]),
                "strong_u2_prefactor_interval": list(params["strong_u2_prefactor_interval"]),
                "crossover_beta": float(params["crossover_beta"]),
            },
        },
    }


def get_uv_parameters_derived(*, proof_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compatibility shim returning the same shape as uv_hypotheses.get_uv_parameters."""
    c = get_uv_constants(proof_status=proof_status)
    model = c["derived"]["irrelevant_norm_model"]
    beta_coeffs = c["derived"].get("beta_coefficients", {"b0": None, "b1": None})
    
    return {
        "balaban_epsilon": float(c["derived"]["epsilon_Balaban"]),
        "c1_interval": tuple(model["c1_interval"]),
        "c2_interval": tuple(model["c2_interval"]),
        "strong_u2_prefactor_interval": tuple(model["strong_u2_prefactor_interval"]),
        "crossover_beta": float(model["crossover_beta"]),
        "weak_C_remainder": float(c["derived"]["weak_C_remainder"]),
        "flow_C_remainder": float(c["derived"]["flow_C_remainder"]),
        "b0_interval": (float(beta_coeffs.get("b0_lower", 0.0)), float(beta_coeffs.get("b0_upper", 0.0))),
        "b1_interval": (float(beta_coeffs.get("b1_lower", 0.0)), float(beta_coeffs.get("b1_upper", 0.0))),
    }
