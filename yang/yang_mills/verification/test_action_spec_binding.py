import json
import os


def test_os_audit_includes_action_spec_metadata():
    from os_audit import audit_os_reconstruction

    res = audit_os_reconstruction()
    action_chk = next((c for c in res.get("checks", []) if c.get("key") == "action_is_wilson_plaquette"), None)
    assert action_chk is not None
    assert action_chk.get("status") in {"PASS", "FAIL"}
    assert "action_spec" in action_chk
    assert action_chk["action_spec"].get("sha256")


def test_mass_gap_certificate_includes_action_spec(tmp_path, monkeypatch):
    from verify_gap_rigorous import verify_spectrum

    monkeypatch.delenv("YM_STRICT", raising=False)
    monkeypatch.chdir(tmp_path)
    ok = verify_spectrum()
    assert ok is True

    cert = json.loads((tmp_path / "mass_gap_certificate.json").read_text(encoding="utf-8"))
    assert "inputs" in cert
    assert "action_spec" in cert["inputs"]
    assert cert["inputs"]["action_spec"].get("sha256")
