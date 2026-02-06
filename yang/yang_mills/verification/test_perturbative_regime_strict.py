import os
import sys

# Ensure `verification/` modules are importable when running pytest from repo root.
import os as _os
sys.path.insert(0, _os.path.dirname(__file__))

from verify_perturbative_regime import verify_asymptotic_freedom_flow_result


def test_strict_mode_disallows_conditional(monkeypatch):
    # In this repository's default proof_status (ASSUMPTION-BASED), strict mode should
    # refuse to treat CONDITIONAL as success.
    monkeypatch.setenv("YM_STRICT", "1")
    res = verify_asymptotic_freedom_flow_result()
    assert res["ok"] is False
    assert res["status"] == "FAIL"
    assert res["reason"] == "strict_mode_disallows_conditional"


def test_default_mode_allows_conditional(monkeypatch):
    monkeypatch.delenv("YM_STRICT", raising=False)
    res = verify_asymptotic_freedom_flow_result()
    assert res["status"] in {"CONDITIONAL", "PASS", "FAIL"}
    if res["status"] == "CONDITIONAL":
        assert res["ok"] is True
