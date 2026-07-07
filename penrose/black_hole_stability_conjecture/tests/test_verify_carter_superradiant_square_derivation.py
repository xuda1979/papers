"""Tests for the SSEDT verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_carter_superradiant_square_derivation as v  # noqa: E402

CERT = ROOT / "certificates" / "ssedt" / "spin_half_clean_chart_superradiant_square_derivation.json"


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_ssedt_accepts():
    result = v.verify_file(CERT)
    assert result["accepted"] is True
    assert result["claim_scope"] == "chart_superradiant_square_derivation"
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["derivation_status"] == "exact_superradiant_square"
    assert result["derivation_chain_length"] == 5
    assert "superradiant_square_packet_expands_to_s_ell_squared_minus_2_s_c_x_ell_plus_s_c_squared_x_squared" in result["checked_identities"]
    assert "kerr_horizon_identity_r_plus_squared_plus_a_squared_equals_2_M_r_plus" in result["checked_identities"]
    assert result["proves_full_q_grav"] is False

def test_ssedt_rejects_wrong_claim_scope():
    data = _load(CERT)
    data["metadata"]["claim_scope"] = "wrong_scope"
    with pytest.raises(ValueError, match="claim_scope must be"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_wrong_derivation_status():
    data = _load(CERT)
    data["metadata"]["derivation_status"] = "candidate"
    with pytest.raises(ValueError, match="derivation_status must be one of"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_candidate_provenance_in_chain():
    data = _load(CERT)
    data["derivation_chain"][0]["provenance"] = "candidate Teukolsky operator"
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_placeholder_provenance_in_packet():
    data = _load(CERT)
    data["superradiant_square_packet"]["provenance"] = "placeholder packet"
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_wrong_coefficient_variable():
    data = _load(CERT)
    data["superradiant_square_packet"]["coefficient_variable"] = "h"
    with pytest.raises(ValueError, match="coefficient_variable must be 's'"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_wrong_coupling_variable():
    data = _load(CERT)
    data["superradiant_square_packet"]["coupling_coefficient_variable"] = "h"
    with pytest.raises(ValueError, match="coupling_coefficient_variable must be 'c'"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_wrong_packet_expression():
    data = _load(CERT)
    data["superradiant_square_packet"]["packet_expression"] = {"prod": [{"var": "s"}, {"var": "ell"}]}
    with pytest.raises(ValueError, match="must be a square"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_wrong_coefficient_s_formula():
    data = _load(CERT)
    data["superradiant_coefficient_formula"]["coefficient_s_formula"] = "rho * forbidden_var"
    with pytest.raises(ValueError, match="must be a function of"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_missing_trace_path():
    data = _load(CERT)
    data["principal_symbol_trace_certificate_path"] = "nonexistent/path.json"
    with pytest.raises(ValueError, match="principal_symbol_trace_certificate_path does not exist"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_chart_mismatch_with_trace():
    data = _load(CERT)
    data["chart"] = "different_chart"
    with pytest.raises(ValueError, match="chart must match the principal symbol trace"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_duplicate_step_names():
    data = _load(CERT)
    data["derivation_chain"][1]["step_name"] = data["derivation_chain"][0]["step_name"]
    with pytest.raises(ValueError, match="step_name duplicates an earlier step"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")


def test_ssedt_rejects_empty_derivation_chain():
    data = _load(CERT)
    data["derivation_chain"] = []
    with pytest.raises(ValueError, match="derivation_chain must be a nonempty list"):
        v.verify(data, base_dir=ROOT / "certificates" / "ssedt")
