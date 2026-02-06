

def test_operator_convergence_evidence_default_artifact_passes():
    try:
        from operator_convergence_evidence import (
            audit_operator_convergence_evidence,
            default_operator_convergence_evidence_path,
        )
    except Exception:
        from verification.operator_convergence_evidence import (
            audit_operator_convergence_evidence,
            default_operator_convergence_evidence_path,
        )

    res = audit_operator_convergence_evidence(default_operator_convergence_evidence_path())
    assert res["status"] in {"PASS", "FAIL", "CONDITIONAL"}
    assert res["status"] == "PASS"


def test_continuum_obligations_marks_operator_convergence_pass():
    try:
        from continuum_obligations import continuum_obligations
    except Exception:
        from verification.continuum_obligations import continuum_obligations

    obs = continuum_obligations()
    item = next(o for o in obs if o.get("key") == "cont_operator_or_semigroup_convergence")
    assert item.get("status") == "PASS"
    assert (item.get("evidence") or {}).get("status") == "PASS"
