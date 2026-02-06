import os


def test_os_obligations_registry_has_stable_keys():
    try:
        from os_obligations import os_obligations
    except Exception:
        from verification.os_obligations import os_obligations

    obs = os_obligations()
    assert isinstance(obs, list)
    keys = [o.get("key") for o in obs if isinstance(o, dict)]
    # Spot-check a few critical obligations.
    assert "os_action_pinned" in keys
    assert "os_rp_lattice_proved" in keys
    assert "os_reconstruction_invoked_constructively" in keys


def test_os_audit_includes_granular_obligations(monkeypatch):
    monkeypatch.delenv("YM_STRICT", raising=False)

    try:
        from os_audit import audit_os_reconstruction
    except Exception:
        from verification.os_audit import audit_os_reconstruction

    res = audit_os_reconstruction()
    keys = [c.get("key") for c in (res.get("checks") or []) if isinstance(c, dict)]
    # Should include at least one of the granular obligation keys.
    assert "os_rp_lattice_proved" in keys
    # Still theorem-boundary by default.
    assert res["status"] == "CONDITIONAL"
