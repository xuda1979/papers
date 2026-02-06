import json
import os


def test_continuum_audit_writes_artifact(tmp_path, monkeypatch):
    # Run verify_gap_rigorous in a temp CWD to capture artifact outputs.
    from verify_gap_rigorous import verify_spectrum

    monkeypatch.delenv("YM_STRICT", raising=False)
    monkeypatch.chdir(tmp_path)
    ok = verify_spectrum()
    assert ok is True

    p = tmp_path / "continuum_limit_audit_result.json"
    assert p.exists()

    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] in {"CONDITIONAL", "PASS", "FAIL"}
    # In default (non-strict) mode today, this should remain a theorem-boundary.
    assert data["status"] == "CONDITIONAL"
    assert data["reason"] == "theorem_boundary"


def test_mass_gap_certificate_embeds_continuum_audit(tmp_path, monkeypatch):
    from verify_gap_rigorous import verify_spectrum

    monkeypatch.delenv("YM_STRICT", raising=False)
    monkeypatch.chdir(tmp_path)
    ok = verify_spectrum()
    assert ok is True

    cert_path = tmp_path / "mass_gap_certificate.json"
    assert cert_path.exists()
    cert = json.loads(cert_path.read_text(encoding="utf-8"))

    assert cert.get("schema") == "yangmills.mass_gap_certificate.v1"
    assert "continuum_audit" in cert
    assert cert["continuum_audit"]["status"] == "CONDITIONAL"
