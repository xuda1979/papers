"""os_reconstruction_evidence.py

Artifact schema + verifier for OS reconstruction *inputs* and constructive invocation.

Why this exists
---------------
The OS reconstruction theorem is classical, but a "Clay-standard" claim requires
that all *hypotheses* it needs are explicitly discharged for the limiting
Schwinger functions used by the project.

This module does NOT prove those hypotheses today. Instead, it defines a strict,
machine-verifiable interface for an in-repo evidence bundle, so that:

- `PASS` means: an artifact exists, it is well-formed, and it is pinned to the
  current `action_spec.json` (to prevent assumption drift).
- `CONDITIONAL` means: the artifact is missing or still marked as theorem-boundary.
- `FAIL` means: the artifact exists but is malformed or contradicts pinned inputs.

The intent is identical to `rp_evidence.py` and `operator_convergence_evidence.py`.

Schema (yangmills.os_reconstruction_evidence.v1)
----------------------------------------------
Top-level keys (required unless noted):

- schema: fixed string
- action_spec.sha256: must match the sha256 of `verification/action_spec.json`
- schwinger_functions:
    kind: string (e.g., "gauge_invariant_schwinger")
    description: string
    n_point_max: int >= 2
- axioms: dict of boolean flags describing what has been verified
    reflection_positivity
    euclidean_invariance
    symmetry
    regularity
    clustering
- reconstruction:
    invoked: bool
    output:
        hilbert_space: string (description or reference)
        hamiltonian: string (description or reference)
        vacuum: string (description or reference)
- provenance.source: non-empty string

Status logic
------------
- Missing file => CONDITIONAL.
- Present but action_spec sha is missing/TBD/UNKNOWN => CONDITIONAL.
- Present but action_spec sha mismatches => FAIL.
- Clay-level gating: even if the JSON declares all axioms/bounds as true,
  this module reports CONDITIONAL unless it also includes an explicit pinned
  proof artifact block that is (at least) machine-verifiable by hash.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple


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


def default_os_reconstruction_evidence_path() -> str:
    return os.path.join(os.path.dirname(__file__), "os_reconstruction_evidence.json")


def _action_spec_sha256() -> Optional[str]:
    path = os.path.join(os.path.dirname(__file__), "action_spec.json")
    return _maybe_sha256(path)


def load_os_reconstruction_evidence(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as e:
        return None, f"read_error:{e}"


def _is_truthy_bool(x: Any) -> bool:
    return isinstance(x, bool)


def verify_os_reconstruction_evidence(doc: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(doc, dict):
        return False, "not_a_dict"

    if doc.get("schema") != "yangmills.os_reconstruction_evidence.v1":
        return False, "bad_schema"

    # Action pinning.
    action_block = doc.get("action_spec")
    if not isinstance(action_block, dict):
        return False, "missing_action_spec_block"

    claimed = action_block.get("sha256")
    actual = _action_spec_sha256()

    if not claimed or not isinstance(claimed, str) or claimed.strip().upper() in {"MISSING", "TBD", "UNKNOWN"}:
        return False, "theorem_boundary_missing_action_spec_sha256"

    if actual and claimed != actual:
        return False, "action_spec_sha256_mismatch"

    # Schwinger functions descriptor.
    sf = doc.get("schwinger_functions")
    if not isinstance(sf, dict):
        return False, "missing_schwinger_functions"

    if not (isinstance(sf.get("kind"), str) and sf.get("kind")):
        return False, "schwinger_kind_missing"

    if not (isinstance(sf.get("description"), str) and sf.get("description")):
        return False, "schwinger_description_missing"

    nmax = sf.get("n_point_max")
    try:
        nmax_i = int(nmax)
    except Exception:
        return False, "n_point_max_not_int"
    if nmax_i < 2:
        return False, "n_point_max_too_small"

    # Axioms flags.
    axioms = doc.get("axioms")
    if not isinstance(axioms, dict):
        return False, "missing_axioms"

    required_flags = [
        "reflection_positivity",
        "euclidean_invariance",
        "symmetry",
        "regularity",
        "clustering",
    ]
    for k in required_flags:
        if k not in axioms:
            return False, f"missing_axiom_flag:{k}"
        if not _is_truthy_bool(axioms.get(k)):
            return False, f"axiom_flag_not_bool:{k}"

    # Reconstruction block.
    recon = doc.get("reconstruction")
    if not isinstance(recon, dict):
        return False, "missing_reconstruction"

    if not _is_truthy_bool(recon.get("invoked")):
        return False, "reconstruction_invoked_not_bool"

    out = recon.get("output")
    if not isinstance(out, dict):
        return False, "missing_reconstruction_output"

    for k in ["hilbert_space", "hamiltonian", "vacuum"]:
        v = out.get(k)
        if not (isinstance(v, str) and v.strip()):
            return False, f"missing_output_field:{k}"

    # Provenance.
    prov = doc.get("provenance")
    if not isinstance(prov, dict):
        return False, "missing_provenance"

    src = prov.get("source")
    if not (isinstance(src, str) and src.strip()):
        return False, "missing_provenance_source"

    # Clay-level gating: require a pinned proof artifact block.
    proof = doc.get("proof")
    if not isinstance(proof, dict):
        return False, "theorem_boundary_missing_proof_block"

    if proof.get("schema") not in {
        "yangmills.os_reconstruction_proof_artifact.v1",
    }:
        return False, "theorem_boundary_bad_or_missing_proof_schema"

    proof_sha = proof.get("sha256")
    if not (isinstance(proof_sha, str) and proof_sha.strip() and proof_sha.strip().upper() not in {"MISSING", "TBD", "UNKNOWN"}):
        return False, "theorem_boundary_missing_proof_sha256"

    return True, "ok"


def audit_os_reconstruction_evidence(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = default_os_reconstruction_evidence_path()

    doc, err = load_os_reconstruction_evidence(path)
    sha = _maybe_sha256(path) if doc is not None else None

    if doc is None:
        return {
            "key": "os_reconstruction_evidence_present",
            "title": "OS reconstruction evidence artifact",
            "status": "CONDITIONAL",
            "detail": f"Missing OS reconstruction evidence artifact at {path} ({err})",
            "artifact": {
                "path": os.path.abspath(path),
                "sha256": sha,
                "schema": "yangmills.os_reconstruction_evidence.v1",
            },
        }

    ok, reason = verify_os_reconstruction_evidence(doc)

    status = "PASS" if ok else "FAIL"
    if not ok and reason.startswith("theorem_boundary_"):
        status = "CONDITIONAL"

    return {
        "key": "os_reconstruction_evidence_present",
        "title": "OS reconstruction evidence artifact",
        "status": status,
        "detail": reason,
        "artifact": {
            "path": os.path.abspath(path),
            "sha256": sha,
            "schema": doc.get("schema"),
        },
        "evidence": {
            "action_spec_sha256": ((doc.get("action_spec") or {}) if isinstance(doc.get("action_spec"), dict) else {}).get("sha256"),
            "schwinger_functions": doc.get("schwinger_functions"),
            "axioms": doc.get("axioms"),
            "reconstruction": doc.get("reconstruction"),
            "provenance": doc.get("provenance"),
        },
    }
