import json


def test_gap_verifier_writes_mass_gap_certificate(tmp_path, monkeypatch):
    # Run in a temp working directory so we don't pollute repo outputs.
    monkeypatch.chdir(tmp_path)

    # Ensure we are not in strict mode; strict mode is expected to fail at the
    # OS theorem-boundary gate in this repo.
    monkeypatch.delenv("YM_STRICT", raising=False)

    from verify_gap_rigorous import verify_spectrum

    ok = verify_spectrum()
    assert ok is True

    out = tmp_path / "mass_gap_certificate.json"
    assert out.exists(), "verify_gap_rigorous should write mass_gap_certificate.json to CWD"

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data.get("schema") == "yangmills.mass_gap_certificate.v1"
    assert "mass_gap" in data and "lower_bound" in data["mass_gap"]
    assert data["mass_gap"]["lower_bound"] > 0
    assert "os_audit" in data and "status" in data["os_audit"]
    assert "checkpoints" in data and "rows" in data["checkpoints"]
    assert isinstance(data["checkpoints"]["rows"], list)
