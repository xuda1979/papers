import os


def test_strict_mode_blocks_theorem_boundaries(monkeypatch, tmp_path):
    # In strict mode, theorem-boundary audits should cause the verifier to fail.
    try:
        from verification.verify_gap_rigorous import verify_spectrum
    except Exception:
        from verify_gap_rigorous import verify_spectrum

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YM_STRICT", "1")
    ok = verify_spectrum()
    assert ok is False
