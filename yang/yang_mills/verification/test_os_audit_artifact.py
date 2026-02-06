import json
import os


def test_gap_verifier_writes_os_audit_artifact(tmp_path, monkeypatch):
    # Run in a temp working directory so we don't pollute repo outputs.
    monkeypatch.chdir(tmp_path)

    # Ensure we are not in strict mode; strict mode is expected to fail at the
    # OS theorem-boundary gate.
    monkeypatch.delenv("YM_STRICT", raising=False)

    # Ensure verification modules can still be imported.
    # Put the repo verification directory on sys.path via env.
    # The verifier itself manages sys.path insertion relative to its own file.

    from verify_gap_rigorous import verify_spectrum

    ok = verify_spectrum()
    assert ok is True

    out = tmp_path / "os_audit_result.json"
    assert out.exists(), "verify_gap_rigorous should write os_audit_result.json"

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "status" in data
    assert "checks" in data
    assert isinstance(data["checks"], list)
