def _import_audit():
    try:
        from schwinger_limit_evidence import audit_schwinger_limit_evidence
    except Exception:
        from verification.schwinger_limit_evidence import audit_schwinger_limit_evidence
    return audit_schwinger_limit_evidence


def test_default_schwinger_limit_evidence_is_conditional():
    audit_schwinger_limit_evidence = _import_audit()
    res = audit_schwinger_limit_evidence()
    assert res["key"] == "schwinger_limit_evidence_present"
    # Default artifact exists but is explicitly theorem-boundary (sha256 TBD)
    assert res["status"] == "CONDITIONAL"


def test_continuum_obligations_include_schwinger_evidence():
    try:
        from continuum_obligations import continuum_obligations
    except Exception:
        from verification.continuum_obligations import continuum_obligations

    obs = continuum_obligations()
    # Ensure evidence is attached to the expected obligations.
    by_key = {o["key"]: o for o in obs}
    assert "cont_uniform_moment_bounds" in by_key
    assert "evidence" in by_key["cont_uniform_moment_bounds"]


def test_continuum_audit_includes_schwinger_evidence_record(monkeypatch):
    monkeypatch.delenv("YM_STRICT", raising=False)

    try:
        from verify_continuum_limit import audit_continuum_limit
    except Exception:
        from verification.verify_continuum_limit import audit_continuum_limit

    res = audit_continuum_limit()
    keys = [c["key"] for c in res["checks"]]
    assert "schwinger_limit_evidence_present" in keys
