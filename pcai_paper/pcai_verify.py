#!/usr/bin/env python3
"""
pcai_verify.py
--------------
Standalone, dependency-free verifier for PCAI JSON certificates.

Features:
  - Recomputes SHA-256 digest over a canonicalized certificate body.
  - Confirms that cert["digest"] (if present) matches the recomputed value.
  - Optionally verifies Ed25519 signatures if PyNaCl is installed.
  - Rejects non-finite floats in JSON (NaN/Infinity).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Tuple

from pcai_canonical import (
    canonical_body,
    canonicalize_json,
    digest_for_certificate,
    verify_signature_ed25519,
)


def _read_json(path: str) -> Dict[str, Any]:
    text = sys.stdin.read() if path == "-" else open(path, "r", encoding="utf-8").read()
    return json.loads(text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pcai_verify",
        description="Verify PCAI certificate digests and (optionally) Ed25519 signatures.",
    )
    p.add_argument(
        "certificate",
        nargs="?",
        default="-",
        help="Path to certificate JSON, or '-' for stdin.",
    )
    p.add_argument(
        "--exclude",
        default="verified_at,digest,signature",
        help="Comma-separated keys to exclude from the canonicalized body.",
    )
    p.add_argument(
        "--show-body",
        action="store_true",
        help="Print the canonical JSON body used for hashing.",
    )
    p.add_argument(
        "--out",
        choices=("text", "json"),
        default="json",
        help="Output format (default: json).",
    )
    return p.parse_args(argv)


def _verify(cert: Dict[str, Any], exclude_keys: Tuple[str, ...]) -> Dict[str, Any]:
    body = canonical_body(cert, exclude=exclude_keys)
    canon = canonicalize_json(body)
    computed = digest_for_certificate(cert, exclude=exclude_keys)
    has_digest_field = "digest" in cert
    digest_ok = bool(has_digest_field and cert.get("digest") == computed)

    sig_result = None
    if "public_key" in cert and "signature" in cert:
        sig_result = verify_signature_ed25519(
            cert["public_key"], cert["signature"], canon
        )

    return {
        "ok": digest_ok and (sig_result is not False),
        "ok_digest": digest_ok,
        "ok_signature": sig_result,  # True/False/None (None = not checked)
        "computed_digest": computed,
        "has_digest_field": has_digest_field,
        "excluded_keys": list(exclude_keys),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exclude_keys = tuple(k for k in args.exclude.split(",") if k)
    try:
        cert = _read_json(args.certificate)
        result = _verify(cert, exclude_keys)
        if args.out == "json":
            print(json.dumps(result, indent=2, sort_keys=True))
            if args.show_body:
                print(canonicalize_json(canonical_body(cert, exclude_keys)))
        else:
            status = "OK" if result["ok"] else "FAIL"
            print(
                f"[{status}] digest={result['computed_digest']} "
                f"sig={result['ok_signature']}"
            )
            if args.show_body:
                print("--- canonical body ---")
                print(canonicalize_json(canonical_body(cert, exclude_keys)))
        return 0 if result["ok"] else 2
    except Exception as e:
        if args.out == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
