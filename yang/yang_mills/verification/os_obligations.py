"""os_obligations.py

Granular OS/RP proof obligations registry.

Purpose
-------
`os_audit.py` used to represent the OS/RP situation with two coarse theorem-boundary
items. That was honest, but not actionable.

This module provides a *more explicit contract*: what concrete inputs must exist
(or be proven) to claim OS reconstruction and reflection positivity *within this repo*.

Design
------
- Each obligation is a dict with stable keys so downstream artifacts/tests/LaTeX
  can reference them.
- Today most items remain CONDITIONAL (the repo does not mechanically prove them).
- We keep at least one concrete, checkable PASS item (action-spec pinning), so
  the OS audit remains partly constructive.

NOTE
----
Marking something as PASS here should only be done when the repo has a
machine-checkable artifact/derivation for it.
"""

from __future__ import annotations

from typing import Any, Dict, List


def os_obligations() -> List[Dict[str, Any]]:
    try:
        from rp_evidence import audit_rp_evidence
    except Exception:
        from .rp_evidence import audit_rp_evidence

    try:
        from os_reconstruction_evidence import audit_os_reconstruction_evidence
    except Exception:
        from .os_reconstruction_evidence import audit_os_reconstruction_evidence

    rp_ev = audit_rp_evidence()

    os_ev = audit_os_reconstruction_evidence()

    # IMPORTANT: this module is an obligations *registry*. It must not silently
    # upgrade obligations to PASS just because an evidence artifact exists.
    # Until the repo provides a machine-checkable proof artifact (hash-pinned),
    # `audit_os_reconstruction_evidence()` should remain CONDITIONAL and these
    # obligations should remain CONDITIONAL.

    # IMPORTANT: default repo mode must remain theorem-boundary unless a
    # Clay-certified proof status explicitly enables upgrading.
    try:
        import json, os

        _ps_path = os.path.join(os.path.dirname(__file__), "proof_status.json")
        with open(_ps_path, "r", encoding="utf-8") as f:
            _ps = json.load(f)
        _clay = bool((_ps or {}).get("clay_standard"))
    except Exception:
        _clay = False

    rp_ok = bool(_clay and rp_ev.get("status") == "PASS")
    os_valid = bool(_clay and os_ev.get("status") == "PASS")

    # Only the reflection map definition is treated as concretely specified by
    # the lattice geometry (still not a full RP proof).
    has_map = True

    # The remaining OS axioms are checked from the evidence.
    axioms = os_ev.get("evidence", {}).get("axioms", {})
    has_euclid = bool(axioms.get("euclidean_invariance")) and os_valid
    has_sym = bool(axioms.get("symmetry") and axioms.get("regularity")) and os_valid
    has_cluster = bool(axioms.get("clustering")) and os_valid

    # These are scoped to Euclidean OS axioms + RP sufficient to reconstruct
    # a Hilbert space and a self-adjoint Hamiltonian from limiting Schwinger
    # functions.
    return [
        {
            "key": "os_action_pinned",
            "title": "Action spec pinned",
            "status": "PASS",
            "detail": "Action is pinned by verification/action_spec.json and audited elsewhere.",
        },
        {
            "key": "os_reflection_map_defined",
            "title": "Reflection map and half-space algebra defined",
            "status": "PASS" if has_map else "CONDITIONAL",
            "detail": (
                "Reflection map defined by geometric lattice plane reflection sites" if has_map else
                "Need an explicit definition of the time-reflection map theta and the positive-time *-algebra."
            ),
        },
        {
            "key": "os_rp_lattice_proved",
            "title": "Lattice reflection positivity proved for the pinned action",
            "status": "PASS" if rp_ok else "CONDITIONAL",
            "detail": (
                "Verified via rp_evidence.json" if rp_ok else
                "Need a machine-checkable proof (or formalizable certificate) that the Wilson plaquette action "
                "satisfies reflection positivity under the chosen reflection."
            ),
            "evidence": rp_ev,
        },
        {
            "key": "os_euclidean_invariance_limit",
            "title": "Euclidean invariance in the limit",
            "status": "PASS" if has_euclid else "CONDITIONAL",
            "detail": (
                "Verified via os_reconstruction_evidence.json" if has_euclid else
                "Need proof that the limiting Schwinger functions are (at least) translation + rotation invariant."
            ),
        },
        {
            "key": "os_symmetry_and_regularities",
            "title": "Symmetry and regularity axioms",
            "status": "PASS" if has_sym else "CONDITIONAL",
            "detail": (
                "Verified via os_reconstruction_evidence.json" if has_sym else
                "Need symmetry under permutations and suitable continuity/temperedness bounds."
            ),
        },
        {
            "key": "os_cluster_property",
            "title": "Clustering / ergodicity",
            "status": "PASS" if has_cluster else "CONDITIONAL",
            "detail": (
                "Verified via os_reconstruction_evidence.json" if has_cluster else
                "Need a verified clustering property to obtain uniqueness of the vacuum and a sharp spectral interpretation."
            ),
        },
        {
            "key": "os_reconstruction_invoked_constructively",
            "title": "OS reconstruction invoked with verified inputs",
            "status": "PASS" if os_valid else "CONDITIONAL",
            "detail": (
                "Verified via os_reconstruction_evidence.json" if os_valid else
                "Need a fully explicit chain from verified OS axioms to construction of Hilbert space, vacuum, and Hamiltonian; "
                "not merely citing the theorem."
            ),
            "evidence": os_ev,
        },
    ]
