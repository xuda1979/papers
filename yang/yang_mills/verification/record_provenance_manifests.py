"""record_provenance_manifests.py

Create/update provenance manifests for claim-critical artifacts.

Design goals:
  - Pure Python; deterministic file hashing.
  - In ASSUMPTION-BASED mode: convenience script to generate manifests.
  - In Clay-certified mode: artifacts must already be provenance-bound and
    validated by the runner; this script exists to help prepare a bundle, not to
    silently "fix" Clay mode.

Usage:
  python verification/record_provenance_manifests.py

This writes `<artifact>.provenance.json` next to each artifact.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from provenance import record_derivation


def _load_proof_status() -> Dict[str, object]:
    path = os.path.join(os.path.dirname(__file__), "proof_status.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"clay_standard": False, "claim": "ASSUMPTION-BASED"}


def _artifact_specs() -> List[Dict[str, object]]:
    """Return claim-critical artifacts to provenance-bind (if they exist)."""
    base = os.path.dirname(__file__)

    # NOTE: We keep this list aligned with the runner's Clay-mode preflight list.
    # Source files are "best effort"—they are informational in manifests and used
    # for change detection, not enforcement.
    common_sources = [
        os.path.join(base, "interval_arithmetic.py"),
        os.path.join(base, "uv_hypotheses.py"),
        os.path.join(base, "export_results_to_latex.py"),
    ]

    return [
        {
            "path": os.path.join(base, "rigorous_constants.json"),
            "sources": common_sources + [os.path.join(base, "rigorous_constants_derivation.py")],
            "extra": {"kind": "constants"},
        },
        {
            "path": os.path.join(base, "uv_hypotheses.json"),
            "sources": common_sources,
            "extra": {"kind": "uv_obligations"},
        },
        {
            "path": os.path.join(base, "verification_results.json"),
            "sources": common_sources,
            "extra": {"kind": "results"},
        },
        {
            "path": os.path.join(base, "mass_gap_certificate.json"),
            "sources": common_sources + [os.path.join(base, "verify_gap_rigorous.py")],
            "extra": {"kind": "certificate", "phase": "mass_gap"},
        },
        {
            "path": os.path.join(base, "certificate_phase1.json"),
            "sources": common_sources + [os.path.join(base, "tube_verifier_phase1.py")],
            "extra": {"kind": "certificate", "phase": "phase1"},
        },
        {
            "path": os.path.join(base, "certificate_phase2_hardened.json"),
            "sources": common_sources,
            "extra": {"kind": "certificate", "phase": "phase2"},
        },
        {
            "path": os.path.join(base, "certificate_lorentz.json"),
            "sources": common_sources,
            "extra": {"kind": "certificate", "phase": "lorentz"},
        },
        {
            "path": os.path.join(base, "certificate_anisotropy.json"),
            "sources": common_sources,
            "extra": {"kind": "certificate", "phase": "anisotropy"},
        },
        {
            "path": os.path.join(base, "certificate_final_audit.json"),
            "sources": common_sources,
            "extra": {"kind": "certificate", "phase": "final_audit"},
        },
    ]


def main() -> int:
    proof_status = _load_proof_status()
    clay = bool(proof_status.get("clay_standard"))
    if clay:
        print("[PROVENANCE] NOTE: proof_status marks clay_standard=true.")
        print("[PROVENANCE] This script can record manifests, but Clay-mode runs should not rely on auto-generation.")

    wrote = 0
    skipped = 0
    missing = 0

    for spec in _artifact_specs():
        path = str(spec["path"])
        if not os.path.isfile(path):
            missing += 1
            continue

        try:
            out = record_derivation(
                artifact_path=path,
                source_files=[p for p in spec.get("sources", []) if os.path.isfile(p)],
                extra_metadata={
                    **(spec.get("extra", {}) or {}),
                    "proof_claim": proof_status.get("claim", "ASSUMPTION-BASED"),
                    "clay_standard": clay,
                },
            )
            wrote += 1
            print(f"[PROVENANCE] wrote {os.path.basename(out)}")
        except Exception as e:
            skipped += 1
            print(f"[PROVENANCE][WARN] could not record for {os.path.basename(path)}: {e}")

    print("[PROVENANCE] summary:")
    print(f"  wrote:   {wrote}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
