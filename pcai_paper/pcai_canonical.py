"""
pcai_canonical.py
-----------------
Deterministic JSON canonicalization and SHA-256 digests for PCAI certificates.

Design goals:
 - Small, dependency-free, easy to audit.
 - Deterministic across runs and platforms for the certificate shapes used here.
 - Fail closed on non-finite floats (NaN/Infinity), which many canonicalization
   schemes disallow and which are ambiguous to hash/sign.

This follows the spirit of RFC 8785 (JSON Canonicalization Scheme) but does not
pull in an external implementation. Keys are sorted lexicographically and
insignificant whitespace is removed before hashing.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Dict, Union, Optional

Json = Any

_DEFAULT_EXCLUDE = ("verified_at", "digest", "signature")


def _check_no_nonfinite_floats(obj: Json) -> None:
    """
    Walk the object and raise ValueError if any float is NaN or Infinity.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError("Non-finite float encountered in certificate JSON")
    elif isinstance(obj, dict):
        for v in obj.values():
            _check_no_nonfinite_floats(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _check_no_nonfinite_floats(v)
    # ints/str/bool/None are fine


def canonicalize_json(obj: Json) -> str:
    """
    Return a deterministic JSON string: keys sorted, no insignificant whitespace,
    UTF-8 clean (no forced ASCII escapes).
    """
    _check_no_nonfinite_floats(obj)
    # `sort_keys=True` + separators removes spacing and fixes key order
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: Union[str, bytes]) -> str:
    """
    Hex-encoded SHA-256 of `data`. Strings are encoded as UTF-8 before hashing.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_body(
    cert: Mapping[str, Json],
    exclude: Iterable[str] = _DEFAULT_EXCLUDE,
) -> Dict[str, Json]:
    """
    Return a shallow copy of `cert` with excluded fields removed. This is the
    structure that is canonicalized and hashed to compute the digest.
    """
    ex = set(exclude)
    return {k: v for k, v in cert.items() if k not in ex}


def digest_for_certificate(
    cert: Mapping[str, Json],
    exclude: Iterable[str] = _DEFAULT_EXCLUDE,
) -> str:
    """
    Compute the stable SHA-256 digest for a certificate by canonicalizing the
    filtered body.
    """
    body = canonical_body(cert, exclude=exclude)
    canon = canonicalize_json(body)
    return sha256_hex(canon)


def verify_signature_ed25519(
    public_key_b64: str,
    signature_b64: str,
    message: Union[str, bytes],
) -> Optional[bool]:
    """
    Verify an Ed25519 signature using PyNaCl if it is available.
    Returns:
      - True  : signature verifies
      - False : signature present but invalid
      - None  : PyNaCl not installed; cannot verify (non-fatal for digest checks)
    """
    try:
        from nacl.signing import VerifyKey  # type: ignore
        import base64
    except Exception:
        return None

    try:
        if isinstance(message, str):
            message = message.encode("utf-8")
        vk = VerifyKey(base64.b64decode(public_key_b64))
        vk.verify(message, base64.b64decode(signature_b64))
        return True
    except Exception:
        return False


__all__ = [
    "canonicalize_json",
    "canonical_body",
    "digest_for_certificate",
    "sha256_hex",
    "verify_signature_ed25519",
]
