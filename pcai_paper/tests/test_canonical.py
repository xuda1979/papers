import json

from pcai_canonical import (
    canonical_body,
    canonicalize_json,
    digest_for_certificate,
)


def test_canonicalize_sorts_keys_and_removes_spaces():
    obj = {"b": 2, "a": 1, "nested": {"y": 2, "x": 1}}
    s = canonicalize_json(obj)
    # Keys appear in sorted order, with no spaces
    assert s == '{"a":1,"b":2,"nested":{"x":1,"y":2}}'


def test_digest_excludes_verified_at_and_digest_fields():
    base = {
        "scheme": "forall_range",
        "claim": {"N": 100},
        "result": {"ok": True},
        "verified_at": "2025-10-12T00:00:00Z",
    }
    d1 = digest_for_certificate(base)

    # Changing verified_at should not change the digest
    mutated = dict(base)
    mutated["verified_at"] = "2025-10-13T12:34:56Z"
    d2 = digest_for_certificate(mutated)
    assert d1 == d2

    # Adding digest/signature must also be excluded
    mutated2 = dict(mutated)
    mutated2["digest"] = "deadbeef"
    mutated2["signature"] = "cafebabe"
    d3 = digest_for_certificate(mutated2)
    assert d1 == d3


def test_canonical_body_round_trip_and_digest_match():
    cert = {
        "scheme": "contract_trace",
        "operation": "divide",
        "args": {"a": 7, "b": 3},
        "tolerance": {"abs": 1e-12, "rel": 1e-9},
        "verified_at": "2025-10-13T00:00:00Z",
    }
    body = canonical_body(cert)
    canon = canonicalize_json(body)
    # Ensure this canonical JSON parses back as the same mapping
    parsed = json.loads(canon)
    assert parsed == body
    # Digest should be stable across canon->parse roundtrip
    d1 = digest_for_certificate(cert)
    d2 = digest_for_certificate({"digest": "ignored", **cert})
    assert d1 == d2
