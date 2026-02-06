import os

import pytest


def _import_audit():
    try:
        from os_reconstruction_evidence import audit_os_reconstruction_evidence
    except Exception:
        from verification.os_reconstruction_evidence import audit_os_reconstruction_evidence
    return audit_os_reconstruction_evidence


def test_default_os_reconstruction_evidence_is_conditional():
    audit_os_reconstruction_evidence = _import_audit()
    res = audit_os_reconstruction_evidence()
    assert res["key"] == "os_reconstruction_evidence_present"
    # Default artifact exists but is explicitly theorem-boundary (sha256 TBD)
    assert res["status"] == "CONDITIONAL"


def test_os_audit_stays_conditional_by_default(monkeypatch):
    monkeypatch.delenv("YM_STRICT", raising=False)

    try:
        from os_audit import audit_os_reconstruction
    except Exception:
        from verification.os_audit import audit_os_reconstruction

    res = audit_os_reconstruction()
    # In non-strict mode, theorem-boundary items are allowed.
    assert res["status"] in {"CONDITIONAL", "PASS"}

    keys = [c["key"] for c in res["checks"]]
    assert "os_reconstruction_evidence_present" in keys


def test_os_audit_fails_in_strict_mode(monkeypatch):
    monkeypatch.setenv("YM_STRICT", "1")

    try:
        from os_audit import audit_os_reconstruction
    except Exception:
        from verification.os_audit import audit_os_reconstruction

    res = audit_os_reconstruction()
    # The default evidence remains theorem-boundary, so strict mode must fail.
    assert res["status"] == "FAIL"
    assert res["reason"] == "strict_mode_disallows_conditional"
