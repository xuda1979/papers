"""Tests for the TCDEDT verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_carter_trapped_constraint_correction_derivation as v  # noqa: E402

CERT = (ROOT / "certificates" / "tcdedt"
        / "spin_half_clean_chart_trapped_constraint_correction_derivation.json")


def _load(path: Path = CERT) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_tcdedt_accepts_well_formed_certificate():
    data = _load()
    result = v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")
    assert result["accepted"] is True
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["derivation_chain_length"] == 6
    assert "rho_R" in result["symbolic_coefficient_variables"]
    assert "rho_dR" in result["symbolic_coefficient_variables"]
    assert len(result["trapped_constraint_correction_packets"]) == 2
    expected_identities = {
        "P_R_unique_diagonal_invariant_degree_2_polynomial",
        "P_dR_unique_mixed_degree_2_polynomial_with_horizon_identity",
        "quotient_identity_holds_with_a_R_equals_a_dR_equals_1",
        "P_R_and_P_dR_linearly_independent",
        "correction_decomposes_into_rho_R_and_rho_dR_coefficients",
        "kerr_horizon_identity_r_plus_squared_plus_a_squared_equals_2_M_r_plus",
    }
    assert expected_identities.issubset(set(result["checked_identities"]))
    # The derived packets must match the expected formulas.
    assert result["derived_packets"]["P_R"] == "ell**2 + q**2 + x**2"
    assert result["derived_packets"]["P_dR"] == "ell*x + q**2"
    assert result["multiplier_solution"] == {"a_R": "1", "a_dR": "1"}


def test_tcdedt_rejects_wrong_claim_scope():
    data = _load()
    data["metadata"]["claim_scope"] = "chart_diagonal_slack_derivation"
    with pytest.raises(ValueError, match="claim_scope must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_candidate_derivation_status():
    data = _load()
    data["metadata"]["derivation_status"] = "candidate_trapped_constraint"
    with pytest.raises(ValueError, match="derivation_status must be one of"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_status_without_trapped_keyword():
    data = _load()
    data["metadata"]["status"] = "exact sympy-verified derivation of correction packets"
    with pytest.raises(ValueError, match="status must mention 'trapped'"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_candidate_provenance_in_chain():
    data = _load()
    data["derivation_chain"][0]["provenance"] = (
        "candidate placeholder for the radial trapped constraint")
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_placeholder_provenance_in_packet():
    data = _load()
    data["trapped_constraint_correction_packets"][0]["provenance"] = (
        "TODO radial correction packet")
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_wrong_phase_variables():
    data = _load()
    data["variables"]["phase_variables"] = ["x", "ell", "z"]
    with pytest.raises(ValueError, match="phase_variables must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_missing_rho_R_in_symbolic_coefficients():
    data = _load()
    data["variables"]["symbolic_coefficient_variables"] = [
        "h", "s", "tau", "c", "diag_slack", "bilinear_slack", "rho_dR"]
    with pytest.raises(ValueError, match="must contain 'rho_R'"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_missing_rho_dR_in_symbolic_coefficients():
    data = _load()
    data["variables"]["symbolic_coefficient_variables"] = [
        "h", "s", "tau", "c", "diag_slack", "bilinear_slack", "rho_R"]
    with pytest.raises(ValueError, match="must contain 'rho_dR'"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_wrong_packet_count():
    data = _load()
    data["trapped_constraint_correction_packets"] = (
        data["trapped_constraint_correction_packets"][:1])
    with pytest.raises(ValueError, match="must be a list of 2 packets"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_wrong_packet_name():
    data = _load()
    data["trapped_constraint_correction_packets"][0]["name"] = "wrong_name_packet"
    with pytest.raises(ValueError, match="is not one of"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_wrong_radial_packet_expression():
    data = _load()
    # Replace the radial packet expression with a wrong formula.
    data["trapped_constraint_correction_packets"][0]["packet_expression"] = (
        "x**2 + ell**2")
    with pytest.raises(ValueError, match="must equal the derived formula"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_wrong_derivative_packet_expression():
    data = _load()
    # Replace the derivative packet expression with a wrong formula.
    data["trapped_constraint_correction_packets"][1]["packet_expression"] = (
        "x*ell + q")
    with pytest.raises(ValueError, match="must equal the derived formula"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_missing_trace_certificate():
    data = _load()
    data["principal_symbol_trace_certificate_path"] = "nonexistent.json"
    with pytest.raises(ValueError, match="does not exist"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_chart_mismatch_with_trace():
    data = _load()
    data["chart"] = "different_chart_name"
    with pytest.raises(ValueError, match="chart must match"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_rejects_duplicated_step_name():
    data = _load()
    data["derivation_chain"][1]["step_name"] = data["derivation_chain"][0]["step_name"]
    with pytest.raises(ValueError, match="step_name.*is duplicated"):
        v.verify(data, base_dir=ROOT / "certificates" / "tcdedt")


def test_tcdedt_verify_file_accepts_real_certificate():
    """End-to-end: verify_file on the real certificate must succeed."""
    result = v.verify_file(CERT)
    assert result["accepted"] is True
