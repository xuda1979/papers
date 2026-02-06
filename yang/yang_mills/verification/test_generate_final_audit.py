"""Tests for generate_final_audit.py and hardened export_results_to_latex.py behavior."""

import json
import os
import sys


def test_generate_final_audit_is_not_unconditional_pass(monkeypatch):
    """
    The current repo has theorem-boundary continuum items (CONDITIONAL).
    generate_final_audit must NOT produce status=PASS until those are resolved.
    """
    monkeypatch.delenv("YM_STRICT", raising=False)

    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    from generate_final_audit import generate_final_audit

    cert = generate_final_audit()
    assert cert["status"] in {"PASS", "CONDITIONAL", "FAIL"}
    # Because the repo currently has theorem-boundary items, we expect NOT PASS.
    assert cert["status"] != "PASS", (
        "generate_final_audit must not claim PASS while continuum/OS audits are CONDITIONAL"
    )


def test_generate_final_audit_axiomatic_conditions_match_status():
    """
    axiomatic_conditions flags (mass_gap_positivity, continuum_limit_existence, lsi_uniformity)
    must only be True when the corresponding audits are PASS.
    """
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    from generate_final_audit import generate_final_audit

    cert = generate_final_audit()
    cont_status = cert.get("audit_details", {}).get("continuum", {}).get("status")
    os_status = cert.get("audit_details", {}).get("os_reconstruction", {}).get("status")

    # If continuum is not PASS, continuum_limit_existence must be False
    if cont_status != "PASS":
        assert cert["axiomatic_conditions"]["continuum_limit_existence"] is False

    # If OS reconstruction is not PASS, lsi_uniformity must be False
    if os_status != "PASS":
        assert cert["axiomatic_conditions"]["lsi_uniformity"] is False


def test_export_results_to_latex_status_reflects_audit(monkeypatch, tmp_path):
    """
    run_full_verification() should produce metadata.status != PASS
    when the final audit is not PASS.
    """
    monkeypatch.delenv("YM_STRICT", raising=False)

    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    from export_results_to_latex import run_full_verification

    results = run_full_verification()
    status = results["metadata"]["status"]
    # Because theorem-boundary audits are CONDITIONAL, we expect CONDITIONAL or FAIL, NOT PASS.
    assert status in {"CONDITIONAL", "FAIL"}, (
        f"export_results_to_latex must not claim PASS; got {status}"
    )
