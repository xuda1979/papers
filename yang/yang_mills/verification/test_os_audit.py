import os
import sys

# Ensure `verification/` modules are importable when running pytest from repo root.
import os as _os
sys.path.insert(0, _os.path.dirname(__file__))

from os_audit import audit_os_reconstruction


def test_os_audit_default_is_conditional(monkeypatch):
    monkeypatch.delenv("YM_STRICT", raising=False)
    res = audit_os_reconstruction()
    assert res["status"] in {"CONDITIONAL", "PASS", "FAIL"}
    # Current repo should be theorem-boundary by default.
    assert res["status"] == "CONDITIONAL"
    assert res["ok"] is True


def test_os_audit_strict_fails(monkeypatch):
    monkeypatch.setenv("YM_STRICT", "1")
    res = audit_os_reconstruction()
    assert res["ok"] is False
    assert res["status"] == "FAIL"
    assert res["reason"] == "strict_mode_disallows_conditional"
