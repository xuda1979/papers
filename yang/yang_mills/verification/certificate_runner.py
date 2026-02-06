"""certificate_runner.py

Deterministic certificate-grade entrypoint.

This script is designed to be the *single* reproducible runner for the CAP suite.
It:
  1) Runs the verification exporter (which itself runs the main checks).
  2) Emits an artifact bundle with hashes + environment metadata.

Artifacts are written under: verification/artifacts/<UTC timestamp>/

Why a wrapper when export_results_to_latex.py already exists?
- It makes the "certificate contract" explicit.
- It collects reproducibility metadata in one place.
- It pins what files are considered the certificate outputs.

Usage:
    python certificate_runner.py

Exit codes:
    0  PASS
    1  FAIL (verification failed)
    2  ERROR (exception / cannot produce artifacts)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from typing import Dict, List


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _copy_into_artifacts(src: str, dst_dir: str) -> str:
    _ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


def _collect_environment() -> Dict[str, object]:
    return {
        "utc_generated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
    }


def _hash_source_files(script_dir: str) -> Dict[str, str]:
    """
    Hash all Python source files to ensure code integrity.
    
    Implements Gap #7: Provenance Hashing.
    Returns a dictionary of relative paths to SHA256 hashes.
    """
    hashes = {}
    print("  [PROVENANCE] Hashing source files for integrity...")
    
    # Walk typical code directories
    valid_extensions = {".py", ".json", ".tex", ".bib"}
    
    for root, _, files in os.walk(script_dir):
        # Exclude generated artifacts, cache, and experimental folders
        if any(x in root for x in ["items", "artifacts", "__pycache__", ".git", ".vscode"]):
            continue
            
        for name in files:
            _, ext = os.path.splitext(name)
            if ext not in valid_extensions:
                continue
                
            path = os.path.join(root, name)
            
            # Skip the manifest itself if it exists (loop prevention)
            if "certificate_manifest.json" in name:
                continue

            # Store relative path for cleanliness
            rel_path = os.path.relpath(path, script_dir)
            hashes[rel_path] = _sha256_file(path)
            
    return hashes


def _generate_bundle(script_dir: str) -> int:
    """Create an artifact bundle for the primary outputs.

    Returns:
        0 on success, 2 on bundle-generation error.
    """
    paper_dir = os.path.join(script_dir, "..", "single")
    latex_out = os.path.abspath(os.path.join(paper_dir, "verification_results.tex"))
    json_out = os.path.abspath(os.path.join(script_dir, "verification_results.json"))

    expected: List[str] = [latex_out, json_out]
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        print("[CERT] ERROR: expected outputs missing:")
        for p in missing:
            print(f"  - {p}")
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts_root = os.path.join(script_dir, "artifacts", ts)
    _ensure_dir(artifacts_root)

    for p in expected:
        _copy_into_artifacts(p, artifacts_root)

    manifest = {
        "environment": _collect_environment(),
        "status": "PASS",
        "outputs": {
            os.path.basename(p): {
                "source": os.path.abspath(p),
                "sha256": _sha256_file(os.path.join(artifacts_root, os.path.basename(p))),
            }
            for p in expected
        },
        "provenance": _hash_source_files(script_dir)
    }

    manifest_path = os.path.join(artifacts_root, "certificate_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[CERT] Bundle written to: {artifacts_root}")
    return 0



def check_result(res):
    """Normalize return values: True/0 -> Success (0), False/!0 -> Fail (1)"""
    if res is True: return 0
    if res is False: return 1
    if isinstance(res, int): return res
    return 1 # Default fail

def main() -> int:
    # Ensure imports resolve relative to verification/ directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    # Import late so sys.path is set.
    import export_results_to_latex  # noqa: WPS433
    import verify_uv_handoff
    import verify_ir_limit
    import verify_axioms
    import verify_gap_rigorous  # New rigorous gap check
    import verify_perturbative_regime # New UV completion check
    import verify_lorentz_restoration_strict # New rotation check

    # Run exporter (returns 0/1)
    status_code = 0
    try:
        # 1. Main Verification Phase (Tube Contraction)
        print("\n[CERT] Running Phase 1: Main Verification (Tube Contraction)...")
        if check_result(export_results_to_latex.main()) != 0:
            status_code = 1
        
        # 2. UV Handoff Check
        print("\n[CERT] Running Phase 2: UV Handoff Verification...")
        if check_result(verify_uv_handoff.verify_uv_condition()) != 0:
            status_code = 1
            
        # 3. IR Limit Check
        print("\n[CERT] Running Phase 3: IR Limit Verification...")
        if check_result(verify_ir_limit.verify_ir_condition()) != 0:
            status_code = 1

        # 4. Rigorous Gap Spectrum & OS Reconstruction
        print("\n[CERT] Running Phase 4: Hamiltonian Gap & OS Reconstruction...")
        if check_result(verify_gap_rigorous.verify_spectrum()) != 0:
            status_code = 1
            
        # 5. UV Completion (Perturbative Flow)
        print("\n[CERT] Running Phase 5: UV Completion (Perturbative Regime)...")
        if check_result(verify_perturbative_regime.verify_asymptotic_freedom_flow()) != 0:
            status_code = 1
            
        # 6. Axioms
        print("\n[CERT] Running Phase 6: Axiom Compliance Audit...")
        if check_result(verify_axioms.verify_axiom_compliance()) != 0:
            status_code = 1

        # 7. Rotation Invariance (Euclidean Symmetry Restoration)
        print("\n[CERT] Running Phase 7: Rotation Invariance Restoration...")
        if check_result(verify_lorentz_restoration_strict.verify_rotation_restoration()) != 0:
            status_code = 1
            
    except Exception as e:
        print(f"[CERT] ERROR: Verification suite crashed: {e}")
        status_code = 2

    # Emit Artifact Bundle
    if status_code == 0:
        print("\n[CERT] SUCCESS: All certificates valid. Generating Audit Bundle...")
        bundle_rc = _generate_bundle(script_dir)
        if bundle_rc != 0:
            return bundle_rc
    else:
        print("\n[CERT] FAILURE: One or more verification phases failed.")

    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
