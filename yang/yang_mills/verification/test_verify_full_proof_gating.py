import os
import sys
import subprocess


def _run(cmd, *, cwd, env):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_verify_full_proof_default_mode_not_clay_message():
    """Default mode must not emit 'RIGOROUS PROOF VERIFIED.' over-claim."""

    here = os.path.dirname(__file__)
    py = sys.executable
    env = dict(os.environ)
    env.pop("YM_STRICT", None)

    p = _run([py, "verify_full_proof.py"], cwd=here, env=env)
    # Script may still exit 0 in non-strict mode.
    assert p.returncode == 0
    assert "CONCLUSION: RIGOROUS PROOF VERIFIED." not in (p.stdout + p.stderr)
    assert "NOT CLAY-CERTIFIED" in (p.stdout + p.stderr)


def test_verify_full_proof_strict_mode_fails_on_theorem_boundary():
    """Strict mode must fail if mass_gap_certificate is CONDITIONAL."""

    here = os.path.dirname(__file__)
    py = sys.executable
    env = dict(os.environ)
    env["YM_STRICT"] = "1"

    p = _run([py, "verify_full_proof.py"], cwd=here, env=env)
    assert p.returncode != 0
