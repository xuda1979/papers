"""Tests for the CSEDT verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_carter_carter_scale_derivation as v  # noqa: E402

CERT = ROOT / "certificates" / "csedt" / "spin_half_clean_chart_carter_scale_derivation.json"


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_csedt_accepts():
    result = v.verify_file(CERT)
    assert result["accepted"] is True
    assert result["claim_scope"] == "chart_carter_scale_derivation"
    assert result["derivation_status"] == "exact_carter_scale"
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["carter_scale_packet"]["coefficient_variable"] == "tau"
    assert result["carter_scale_packet"]["phase_variable"] == "q"
    assert result["proves_full_q_grav"] is False
    # Checked identities include the angular-principal-symbol identity.
    ids = result["checked_identities"]
    assert "angular_teukolsky_principal_symbol_is_xi_theta_sq_plus_xi_phi_sq_over_sin_sq_theta" in ids
    assert "carter_scale_coefficient_is_rho_over_sin_sq_theta" in ids
    assert "carter_scale_packet_expression_is_tau_times_q_squared" in ids
    assert "derivation_chain_provenance_is_noncandidate" in ids
    assert "chart_matches_principal_symbol_trace_certificate" in ids


def test_csedt_rejects_wrong_claim_scope():
    data = _load(CERT)
    data["metadata"]["claim_scope"] = "chart_radial_hessian_derivation"
    with pytest.raises(ValueError, match="metadata.claim_scope must be 'chart_carter_scale_derivation'"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_wrong_derivation_status():
    data = _load(CERT)
    data["metadata"]["derivation_status"] = "candidate_carter_scale"
    with pytest.raises(ValueError, match="metadata.derivation_status must be one of"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_candidate_provenance_in_chain():
    data = _load(CERT)
    data["derivation_chain"][0]["provenance"] = "candidate placeholder for the angular Teukolsky operator"
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_placeholder_provenance_in_packet():
    data = _load(CERT)
    data["carter_scale_packet"]["provenance"] = "placeholder Carter-scale packet"
    with pytest.raises(ValueError, match="provenance contains a forbidden"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_wrong_coefficient_variable():
    data = _load(CERT)
    data["carter_scale_packet"]["coefficient_variable"] = "h"
    with pytest.raises(ValueError, match="coefficient_variable must be 'tau'"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_wrong_phase_variable():
    data = _load(CERT)
    data["carter_scale_packet"]["phase_variable"] = "x"
    with pytest.raises(ValueError, match="phase_variable must be 'q'"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_wrong_packet_expression():
    data = _load(CERT)
    # Replace tau*q^2 with h*x^2
    data["carter_scale_packet"]["packet_expression"] = {
        "prod": [{"var": "h"}, {"square": {"var": "x"}}]
    }
    with pytest.raises(ValueError, match="prod\\[0\\] must be \\{'var': 'tau'\\}"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_wrong_coefficient_formula():
    data = _load(CERT)
    data["carter_scale_coefficient_formula"]["formula"] = "rho/cos(theta)**2"
    with pytest.raises(ValueError, match="Carter-scale coefficient formula"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_missing_trace_path():
    data = _load(CERT)
    data["principal_symbol_trace_certificate_path"] = "nonexistent.json"
    with pytest.raises(ValueError, match="principal_symbol_trace_certificate_path does not exist"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_chart_mismatch_with_trace():
    data = _load(CERT)
    data["chart"] = "different_chart"
    with pytest.raises(ValueError, match="chart must match the principal symbol trace"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_duplicate_step_names():
    data = _load(CERT)
    data["derivation_chain"][1]["step_name"] = data["derivation_chain"][0]["step_name"]
    with pytest.raises(ValueError, match="step_name duplicates an earlier step"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_empty_derivation_chain():
    data = _load(CERT)
    data["derivation_chain"] = []
    with pytest.raises(ValueError, match="derivation_chain must be a nonempty list"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_wrong_phase_variables():
    data = _load(CERT)
    data["variables"]["phase_variables"] = ["x", "ell", "z"]
    with pytest.raises(ValueError, match="phase_variables\\[2\\] must be 'q'"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")


def test_csedt_rejects_missing_tau_in_symbolic_coefficients():
    data = _load(CERT)
    data["variables"]["symbolic_coefficient_variables"] = ["h", "s", "c", "diag_slack", "bilinear_slack"]
    with pytest.raises(ValueError, match="symbolic_coefficient_variables must contain 'tau'"):
        v.verify(data, base_dir=ROOT / "certificates" / "csedt")
