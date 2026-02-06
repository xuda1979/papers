"""Tests for Clay-mode enforcement: provenance preflight and strict gating."""

import json
import os
import subprocess
import sys


def _run(cmd, *, cwd, env):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_certificate_runner_v2_clay_preflight_fails_when_provenance_missing(tmp_path, monkeypatch):
    """
    When proof_status.json says clay_standard=true, the preflight must fail
    if required provenance manifests are missing or invalid.
    """
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    import certificate_runner_v2

    # Simulate clay mode via proof_status
    proof_status = {"clay_standard": True, "claim": "CLAY-CERTIFIED"}

    # Create minimal artifacts without any provenance manifests
    (tmp_path / "rigorous_constants.json").write_text("{}", encoding="utf-8")
    (tmp_path / "uv_hypotheses.json").write_text("{}", encoding="utf-8")
    (tmp_path / "verification_results.json").write_text("{}", encoding="utf-8")

    # Monkeypatch dirname so preflight looks in tmp_path
    monkeypatch.setattr(certificate_runner_v2.os.path, "dirname", lambda _: str(tmp_path))

    # Preflight should FAIL (return nonzero) because manifests are missing
    rc = certificate_runner_v2._clay_provenance_preflight(proof_status)
    assert rc != 0, "Clay-mode preflight must fail when provenance manifests are missing"


def test_verify_full_proof_fails_in_strict_mode():
    """
    verify_full_proof.py must exit nonzero in strict mode when
    mass_gap_certificate is CONDITIONAL (theorem-boundary).
    """
    here = os.path.dirname(__file__)
    py = sys.executable
    env = dict(os.environ)
    env["YM_STRICT"] = "1"

    p = _run([py, "verify_full_proof.py"], cwd=here, env=env)
    assert p.returncode != 0, (
        f"verify_full_proof must fail in strict mode with theorem-boundary certificate; "
        f"got rc={p.returncode}"
    )


def test_generate_final_audit_status_is_conditional_or_fail():
    """
    The repo currently has theorem-boundary items, so generate_final_audit
    must NOT claim status=PASS.
    """
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    from generate_final_audit import generate_final_audit

    cert = generate_final_audit()
    assert cert["status"] in {"CONDITIONAL", "FAIL"}, (
        f"Expected CONDITIONAL or FAIL, got {cert['status']}"
    )
