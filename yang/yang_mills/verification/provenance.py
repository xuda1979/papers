"""
provenance.py - Certificate-chain / provenance utilities

This module provides tools to:
1. Hash the inputs (code + data) to a derivation run (e.g. rigorous_constants_derivation.py).
2. Record the hash + timestamp + version into a "provenance manifest".
3. Verify loaded artifacts against the recorded manifest.

The goal is to ensure that any numeric constants used by the verification pipeline
are bound to a reproducible derivation run, reducing "constants came from somewhere" gaps.

Usage:
    from provenance import record_derivation, verify_artifact

    # After generating rigorous_constants.json:
    record_derivation(
        artifact_path="verification/rigorous_constants.json",
        source_files=["verification/rigorous_constants_derivation.py", "verification/interval_arithmetic.py"],
    )

    # Before loading rigorous_constants.json:
    if not verify_artifact("verification/rigorous_constants.json"):
        raise RuntimeError("Artifact provenance check failed!")
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

MANIFEST_SUFFIX = ".provenance.json"


def _sha256_file(path: str) -> str:
    """Return hex sha256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _python_version_string() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def record_derivation(
    artifact_path: str,
    source_files: Optional[List[str]] = None,
    extra_metadata: Optional[dict] = None,
) -> str:
    """
    Record a provenance manifest for an artifact (e.g. rigorous_constants.json).

    The manifest includes:
        - SHA256 of the artifact itself.
        - SHA256 of each source file that contributed to the derivation.
        - Timestamp (UTC).
        - Python version.
        - Optional extra_metadata dict.

    Returns: path to the manifest file.
    """
    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    artifact_hash = _sha256_file(artifact_path)

    source_hashes = {}
    if source_files:
        for sf in source_files:
            if os.path.isfile(sf):
                source_hashes[sf] = _sha256_file(sf)
            else:
                source_hashes[sf] = "FILE_NOT_FOUND"

    manifest = {
        "artifact": os.path.basename(artifact_path),
        "artifact_sha256": artifact_hash,
        "source_files_sha256": source_hashes,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": _python_version_string(),
    }
    if extra_metadata:
        manifest["extra"] = extra_metadata

    manifest_path = artifact_path + MANIFEST_SUFFIX
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def _validate_required_extra(
    manifest: Dict[str, object],
    *,
    require_extra: Optional[Dict[str, object]],
    warn_only: bool,
) -> bool:
    """Validate semantic requirements on manifest metadata.

    Only enforces requirements when `require_extra` is provided.

    Contract:
      - Each key in `require_extra` must exist in `manifest['extra']`.
      - If the required value is a list/tuple, the manifest value must be one of them.
      - Otherwise, the manifest value must equal the required value.
    """
    if not require_extra:
        return True

    extra = manifest.get("extra")
    if not isinstance(extra, dict):
        msg = "Manifest missing required 'extra' metadata"
        if warn_only:
            print(f"[WARN] {msg}")
            return False
        raise ValueError(msg)

    for k, expected in require_extra.items():
        if k not in extra:
            msg = f"Manifest extra missing required key: {k}"
            if warn_only:
                print(f"[WARN] {msg}")
                return False
            raise ValueError(msg)

        actual = extra.get(k)
        if isinstance(expected, (list, tuple)):
            if actual not in expected:
                msg = f"Manifest extra mismatch for {k}: expected one of {list(expected)}, got {actual!r}"
                if warn_only:
                    print(f"[WARN] {msg}")
                    return False
                raise ValueError(msg)
        else:
            if actual != expected:
                msg = f"Manifest extra mismatch for {k}: expected {expected!r}, got {actual!r}"
                if warn_only:
                    print(f"[WARN] {msg}")
                    return False
                raise ValueError(msg)

    return True


def _validate_required_sources(
    manifest: Dict[str, object],
    *,
    require_sources: Optional[List[str]],
    warn_only: bool,
) -> bool:
    """Validate that a manifest explicitly lists the expected source files.

    This does *not* require the source file hashes to match current files.
    It only ensures the manifest includes the provenance inputs by name, so a
    manifest cannot be trivially "empty" in Clay mode.
    """
    if not require_sources:
        return True

    sf = manifest.get("source_files_sha256")
    if not isinstance(sf, dict):
        msg = "Manifest missing source_files_sha256"
        if warn_only:
            print(f"[WARN] {msg}")
            return False
        raise ValueError(msg)

    missing = [p for p in require_sources if p not in sf]
    if missing:
        msg = f"Manifest missing required source file entries: {missing}"
        if warn_only:
            print(f"[WARN] {msg}")
            return False
        raise ValueError(msg)

    return True


def verify_artifact(
    artifact_path: str,
    warn_only: bool = False,
    *,
    require_extra: Optional[Dict[str, object]] = None,
    require_sources: Optional[List[str]] = None,
) -> bool:
    """
    Verify an artifact against its provenance manifest.

    Checks:
        1. Manifest exists.
        2. Artifact SHA256 matches.
        (Source file hashes are informational; mismatch prints a warning but does not fail.)

    Returns True if valid, False (or raises) if invalid.
    """
    manifest_path = artifact_path + MANIFEST_SUFFIX
    if not os.path.isfile(manifest_path):
        msg = f"Provenance manifest not found for {artifact_path}"
        if warn_only:
            print(f"[WARN] {msg}")
            return False
        raise FileNotFoundError(msg)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Optional semantic / metadata enforcement.
    _validate_required_extra(manifest, require_extra=require_extra, warn_only=warn_only)
    _validate_required_sources(manifest, require_sources=require_sources, warn_only=warn_only)

    expected_hash = manifest.get("artifact_sha256")
    if not expected_hash:
        msg = "Manifest missing artifact_sha256"
        if warn_only:
            print(f"[WARN] {msg}")
            return False
        raise ValueError(msg)

    actual_hash = _sha256_file(artifact_path)
    if actual_hash != expected_hash:
        msg = f"Artifact hash mismatch: expected {expected_hash}, got {actual_hash}"
        if warn_only:
            print(f"[WARN] {msg}")
            return False
        raise ValueError(msg)

    # Informational: check source files (warn only, don't fail)
    for sf, expected_sf_hash in manifest.get("source_files_sha256", {}).items():
        if os.path.isfile(sf):
            actual_sf_hash = _sha256_file(sf)
            if actual_sf_hash != expected_sf_hash:
                print(f"[INFO] Source file changed since derivation: {sf}")
        else:
            print(f"[INFO] Source file not found (may have moved): {sf}")

    return True


def enforce_artifact(
    artifact_path: str,
    *,
    clay_certified: bool,
    label: str = "artifact",
    require_extra: Optional[Dict[str, object]] = None,
    require_sources: Optional[List[str]] = None,
) -> bool:
    """Verify an artifact, failing hard in Clay-certified mode.

    Contract:
      - If clay_certified=True: provenance is mandatory and an invalid/missing
        manifest raises.
      - If clay_certified=False: provenance is best-effort; failures only warn.

    Returns True if valid, False if invalid in non-clay mode.
    """
    if clay_certified:
        # Hard fail: missing or stale manifests are not permitted.
        return verify_artifact(
            artifact_path,
            warn_only=False,
            require_extra=require_extra,
            require_sources=require_sources,
        )

    ok = verify_artifact(
        artifact_path,
        warn_only=True,
        require_extra=require_extra,
        require_sources=require_sources,
    )
    if not ok:
        print(f"  [WARN] {label} provenance check failed or manifest missing.")
        print("         This is permitted in ASSUMPTION-BASED mode.")
    return ok


if __name__ == "__main__":
    # Quick self-test / CLI
    import argparse

    parser = argparse.ArgumentParser(description="Provenance utilities")
    sub = parser.add_subparsers(dest="cmd")

    rec = sub.add_parser("record", help="Record a provenance manifest for an artifact")
    rec.add_argument("artifact", help="Path to artifact file")
    rec.add_argument("--sources", nargs="*", help="Source files to hash")

    ver = sub.add_parser("verify", help="Verify an artifact against its manifest")
    ver.add_argument("artifact", help="Path to artifact file")

    args = parser.parse_args()

    if args.cmd == "record":
        out = record_derivation(args.artifact, source_files=args.sources)
        print(f"Manifest written to {out}")
    elif args.cmd == "verify":
        ok = verify_artifact(args.artifact, warn_only=True)
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()
