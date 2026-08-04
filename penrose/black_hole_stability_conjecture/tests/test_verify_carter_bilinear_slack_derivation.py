"""Tests for the BSEDT verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_carter_bilinear_slack_derivation as v  # noqa: E402

CERT = (ROOT / "certificates" / "bsedt"
        / "spin_half_clean_chart_bilinear_slack_derivation.json")


def _load(path: Path = CERT) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_bsedt_accepts_well_formed_certificate():
    data = _load()
    result = v.verify(data, base_dir=ROOT / "certificates" / "bsedt")
    assert result["accepted"] is True
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["derivation_chain_length"] == 5
    assert "bilinear_slack" in result["symbolic_coefficient_variables"]
    assert len(result["bilinear_slack_packets"]) == 3
    # All five core sympy reductions must be checked.
    expected_identities = {
        "sigma_B[0,1]_equals_bilinear_slack_times_x_times_ell",
        "sigma_B[0,2]_equals_bilinear_slack_times_x_times_q",
        "sigma_B[1,2]_equals_bilinear_slack_times_ell_times_q",
        "bianchi_principal_symbol_symmetric",
        "shared_coefficient_reduction_a01_a02_a12_equals_bilinear_slack",
        "offdiag_sum_equals_bilinear_slack_times_cross_term_sum",
        "bilinear_slack_tube_is_half_diagonal_slack_tube",
        "bilinear_slack_interval_inside_accepted_tube",
        "bilinear_slack_interval_symmetric_around_zero",
    }
    assert expected_identities.issubset(set(result["checked_identities"]))


def test_bsedt_rejects_wrong_claim_scope():
    data = _load()
    data["metadata"]["claim_scope"] = "chart_diagonal_slack_derivation"
    with pytest.raises(ValueError, match="claim_scope must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_candidate_derivation_status():
    data = _load()
    data["metadata"]["derivation_status"] = "candidate_bilinear_slack"
    with pytest.raises(ValueError, match="derivation_status must be one of"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_status_without_bilinear_keyword():
    data = _load()
    data["metadata"]["status"] = "exact sympy-verified derivation of the cross-term packets"
    with pytest.raises(ValueError, match="status must mention 'bilinear'"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_candidate_provenance_in_chain():
    data = _load()
    data["derivation_chain"][0]["provenance"] = (
        "candidate placeholder for the Bianchi principal symbol")
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_placeholder_provenance_in_packet():
    data = _load()
    data["bilinear_slack_packets"][0]["provenance"] = (
        "TODO bilinear-slack packet")
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_wrong_phase_variables():
    data = _load()
    data["variables"]["phase_variables"] = ["x", "ell", "z"]
    with pytest.raises(ValueError, match="phase_variables must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_missing_bilinear_slack_in_symbolic_coefficients():
    data = _load()
    data["variables"]["symbolic_coefficient_variables"] = ["h", "s", "tau", "c", "diag_slack"]
    with pytest.raises(ValueError, match="symbolic_coefficient_variables must contain 'bilinear_slack'"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_wrong_packet_count():
    data = _load()
    data["bilinear_slack_packets"] = data["bilinear_slack_packets"][:2]
    with pytest.raises(ValueError, match="bilinear_slack_packets must be a list of 3"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_wrong_packet_name():
    data = _load()
    data["bilinear_slack_packets"][0]["name"] = "wrong_name_bilinear_slack_packet"
    with pytest.raises(ValueError, match="is not one of"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_wrong_phase_pair():
    data = _load()
    # Swap the phase pair of the first packet.
    data["bilinear_slack_packets"][0]["phase_pair"] = ["x", "q"]
    # Also swap the packet_expression factors to match, so the only failure
    # is the phase_pair mismatch.
    data["bilinear_slack_packets"][0]["packet_expression"]["prod"][2] = {"var": "q"}
    with pytest.raises(ValueError, match="phase_pair must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_wrong_coefficient_variable_in_packet():
    data = _load()
    data["bilinear_slack_packets"][0]["coefficient_variable"] = "diag_slack"
    with pytest.raises(ValueError, match="coefficient_variable must be 'bilinear_slack'"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsdet_rejects_interval_outside_tube():
    data = _load()
    data["bilinear_slack_coefficient_formula"]["interval_lo"] = "-1/100"
    data["bilinear_slack_coefficient_formula"]["interval_hi"] = "1/100"
    data["bilinear_slack_coefficient_formula"]["interval"] = "[-1/100, 1/100]"
    with pytest.raises(ValueError, match="must lie inside the accepted tube"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_asymmetric_interval():
    data = _load()
    # Use an interval inside the tube but asymmetric around 0.
    # [-1/8192, 1/16384] is inside [-1/8192, 1/8192] but not symmetric.
    data["bilinear_slack_coefficient_formula"]["interval_lo"] = "-1/8192"
    data["bilinear_slack_coefficient_formula"]["interval_hi"] = "1/16384"
    data["bilinear_slack_coefficient_formula"]["interval"] = "[-1/8192, 1/16384]"
    with pytest.raises(ValueError, match="must be symmetric around 0"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_wrong_accepted_tube():
    data = _load()
    data["bilinear_slack_coefficient_formula"]["accepted_tube_lo"] = "-1/4096"
    data["bilinear_slack_coefficient_formula"]["accepted_tube_hi"] = "1/4096"
    data["bilinear_slack_coefficient_formula"]["accepted_tube"] = "[-1/4096, 1/4096]"
    with pytest.raises(ValueError, match="accepted_tube.*must match the verifier's constant"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_missing_trace_certificate():
    data = _load()
    data["principal_symbol_trace_certificate_path"] = "nonexistent.json"
    with pytest.raises(ValueError, match="does not exist"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_chart_mismatch_with_trace():
    data = _load()
    data["chart"] = "different_chart_name"
    with pytest.raises(ValueError, match="chart must match"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_rejects_duplicated_step_name():
    data = _load()
    data["derivation_chain"][1]["step_name"] = data["derivation_chain"][0]["step_name"]
    with pytest.raises(ValueError, match="step_name.*is duplicated"):
        v.verify(data, base_dir=ROOT / "certificates" / "bsedt")


def test_bsedt_verify_file_accepts_real_certificate():
    """End-to-end: verify_file on the real certificate must succeed."""
    result = v.verify_file(CERT)
    assert result["accepted"] is True
