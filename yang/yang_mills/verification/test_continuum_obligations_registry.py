import os


def test_continuum_obligations_registry_has_stable_keys():
    try:
        from continuum_obligations import continuum_obligations
    except Exception:
        from verification.continuum_obligations import continuum_obligations

    obs = continuum_obligations()
    assert isinstance(obs, list)
    keys = [o.get("key") for o in obs if isinstance(o, dict)]
    assert "cont_limit_exists_subsequence" in keys
    assert "cont_operator_or_semigroup_convergence" in keys
    assert "cont_gap_transfer_hypotheses_verified" in keys


def test_continuum_audit_includes_granular_obligations(monkeypatch):
    monkeypatch.delenv("YM_STRICT", raising=False)

    try:
        from verify_continuum_limit import audit_continuum_limit
    except Exception:
        from verification.verify_continuum_limit import audit_continuum_limit

    res = audit_continuum_limit()
    keys = [c.get("key") for c in (res.get("checks") or []) if isinstance(c, dict)]
    assert "cont_operator_or_semigroup_convergence" in keys
    # By design: still theorem-boundary until actual proofs exist.
    assert res["status"] == "CONDITIONAL"
