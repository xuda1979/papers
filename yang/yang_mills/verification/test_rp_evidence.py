import json


def test_rp_evidence_missing_or_mismatch_is_not_pass():
    # Default repo ships a placeholder rp_evidence.json with sha256=MISSING.
    # That should fail verification (not silently PASS).
    try:
        from rp_evidence import audit_rp_evidence, default_rp_evidence_path
    except Exception:
        from verification.rp_evidence import audit_rp_evidence, default_rp_evidence_path

    res = audit_rp_evidence(default_rp_evidence_path())
    assert res["status"] in {"PASS", "FAIL", "CONDITIONAL"}
    assert res["status"] != "PASS"
    # Prefer keeping placeholders as theorem-boundary rather than hard FAIL.
    assert res["status"] == "CONDITIONAL"


def test_os_obligations_rp_item_exists():
    try:
        from os_obligations import os_obligations
    except Exception:
        from verification.os_obligations import os_obligations

    obs = os_obligations()
    rp_item = next(o for o in obs if o.get("key") == "os_rp_lattice_proved")
    assert rp_item.get("status") in {"PASS", "CONDITIONAL", "FAIL"}
    # Default should remain CONDITIONAL unless a valid rp evidence is provided.
    assert rp_item.get("status") != "PASS"
