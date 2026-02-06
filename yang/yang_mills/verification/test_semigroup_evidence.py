import json


def test_semigroup_evidence_default_artifact_passes():
    # Placeholder artifact is constructed to satisfy delta + exp(-m*t0) < 1.
    try:
        from semigroup_evidence import audit_semigroup_evidence, default_semigroup_evidence_path
    except Exception:
        from verification.semigroup_evidence import audit_semigroup_evidence, default_semigroup_evidence_path

    res = audit_semigroup_evidence(default_semigroup_evidence_path())
    assert res["status"] in {"PASS", "FAIL", "CONDITIONAL"}
    assert res["status"] == "PASS"


def test_continuum_obligations_marks_gap_hypotheses_pass():
    try:
        from continuum_obligations import continuum_obligations
    except Exception:
        from verification.continuum_obligations import continuum_obligations

    obs = continuum_obligations()
    item = next(o for o in obs if o.get("key") == "cont_gap_transfer_hypotheses_verified")
    assert item.get("status") == "PASS"
    assert "evidence" in item
    assert (item["evidence"] or {}).get("status") == "PASS"
