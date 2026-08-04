"""Tests for the QIEDT verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_carter_quotient_identity_derivation as v  # noqa: E402

CERT = (ROOT / "certificates" / "qiedt"
        / "spin_half_clean_chart_quotient_identity_derivation.json")


def _load(path: Path = CERT) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_qiedt_accepts_well_formed_certificate():
    data = _load()
    result = v.verify(data, base_dir=ROOT / "certificates" / "qiedt")
    assert result["accepted"] is True
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["derivation_chain_length"] == 7
    assert len(result["constraint_correction_packets"]) == 3
    assert len(result["sos_defects"]) == 1
    # The QIEDT quotient identity must be the first checked identity.
    assert ("qiedt_quotient_identity_q_grav_equals_backend_plus_sos_plus_ideal"
            in result["checked_identities"])
    # The Carter-mixed packet uniqueness must be checked.
    assert ("carter_mixed_packet_P_xq_unique_swap_invariant_decomposition"
            in result["checked_identities"])
    # The bilinear packet decomposition must be checked.
    assert ("bilinear_packet_decomposition_P_bil_equals_P_dR_plus_P_xq"
            in result["checked_identities"])
    # Linear independence must be checked.
    assert ("linear_independence_of_P_R_P_dR_P_xq"
            in result["checked_identities"])
    # The derived packets must include P_xq.
    assert "P_xq" in result["derived_packets"]
    # Sympy may reorder monomials; check by parsing.
    import sympy as sp
    P_xq_parsed = sp.sympify(result["derived_packets"]["P_xq"])
    P_xq_expected = sp.sympify("x*q + ell*q - q**2")
    assert sp.expand(P_xq_parsed - P_xq_expected) == 0


def test_qiedt_rejects_wrong_derivation_status():
    data = _load()
    data["metadata"]["derivation_status"] = "candidate_derivation"
    with pytest.raises(ValueError, match="metadata.derivation_status must be one of"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_candidate_provenance_in_chain():
    data = _load()
    data["derivation_chain"][0]["provenance"] = (
        "candidate placeholder for the GPDDT density assembly")
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_todo_provenance_in_packet():
    data = _load()
    data["constraint_correction_packets"][0]["provenance"] = (
        "TODO radial correction packet")
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_placeholder_provenance_in_sos_defect():
    data = _load()
    data["sos_defects"][0]["provenance"] = (
        "placeholder gauge-tangent SOS defect")
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_phase_variables():
    data = _load()
    data["variables"]["phase_variables"] = ["x", "ell", "z"]
    with pytest.raises(ValueError, match="phase_variables must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_missing_rho_xq_in_auxiliary_variables():
    data = _load()
    data["variables"]["constraint_auxiliary_variables"] = ["rho_R", "rho_dR"]
    with pytest.raises(ValueError, match="constraint_auxiliary_variables must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_packet_count():
    data = _load()
    data["constraint_correction_packets"] = (
        data["constraint_correction_packets"][:2])
    with pytest.raises(ValueError, match="constraint_correction_packets must contain exactly"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_packet_name():
    data = _load()
    data["constraint_correction_packets"][0]["name"] = "wrong_packet_name"
    with pytest.raises(ValueError, match="is not one of"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_radial_packet_expression():
    data = _load()
    data["constraint_correction_packets"][0]["packet_expression"] = (
        "x**2 + ell**2")
    with pytest.raises(ValueError, match="must equal the derived formula"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_carter_mixed_packet_expression():
    data = _load()
    data["constraint_correction_packets"][2]["packet_expression"] = (
        "x*q + ell*q")
    with pytest.raises(ValueError, match="must equal the derived formula"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_constraint_variable():
    data = _load()
    data["constraint_correction_packets"][2]["constraint_variable"] = "rho_R"
    with pytest.raises(ValueError, match="constraint_variable must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_multiplier():
    data = _load()
    data["constraint_correction_packets"][0]["multiplier"] = "bilinear_slack"
    with pytest.raises(ValueError, match="multiplier must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_sos_defect_count():
    data = _load()
    data["sos_defects"] = data["sos_defects"] + [
        {"name": "extra_defect", "square_expression": "x",
         "weight": "s", "provenance": "extra defect provenance"}]
    with pytest.raises(ValueError, match="sos_defects must contain exactly"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_sos_square_expression():
    data = _load()
    data["sos_defects"][0]["square_expression"] = "c*x + ell"
    with pytest.raises(ValueError, match="square_expression must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_sos_weight():
    data = _load()
    data["sos_defects"][0]["weight"] = "tau"
    with pytest.raises(ValueError, match="weight must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_backend_density():
    data = _load()
    data["backend_density_expression"] = "h*x**2 + tau*q**2 + diag_slack*x**2"
    with pytest.raises(ValueError, match="backend_density_expression must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_wrong_q_grav_expression():
    data = _load()
    data["q_grav_expression"] = "h*x**2 + tau*q**2"
    with pytest.raises(ValueError, match="q_grav_expression must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_negative_sos_weight_interval():
    data = _load()
    data["sos_weight_interval"]["interval_lo"] = "-1/8"
    with pytest.raises(ValueError, match="interval_lo.*must be >= 0"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_sos_interval_outside_tube():
    data = _load()
    data["sos_weight_interval"]["interval_hi"] = "1/2"
    with pytest.raises(ValueError, match="must lie inside accepted tube"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_missing_trace_certificate():
    data = _load()
    data["principal_symbol_trace_certificate_path"] = "nonexistent.json"
    with pytest.raises(ValueError, match="does not exist"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_chart_mismatch_with_trace():
    data = _load()
    data["chart"] = "different_chart_name"
    with pytest.raises(ValueError, match="chart must match"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_rejects_duplicated_step_name():
    data = _load()
    data["derivation_chain"][1]["step_name"] = data["derivation_chain"][0]["step_name"]
    with pytest.raises(ValueError, match="step_name.*is duplicated"):
        v.verify(data, base_dir=ROOT / "certificates" / "qiedt")


def test_qiedt_verify_file_accepts_real_certificate():
    """End-to-end: verify_file on the real certificate must succeed."""
    result = v.verify_file(CERT)
    assert result["accepted"] is True
