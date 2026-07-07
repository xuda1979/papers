"""Tests for the RHEDT verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_carter_radial_hessian_derivation as v  # noqa: E402

CERT = ROOT / "certificates" / "rhedt" / "spin_half_clean_chart_radial_hessian_derivation.json"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _dump(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    with open(p, "w") as f:
        json.dump(data, f)
    return p


def test_rhedt_accepts():
    result = v.verify_file(CERT)
    assert result["accepted"] is True
    assert result["claim_scope"] == "chart_radial_hessian_derivation"
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["derivation_status"] == "exact_radial_hessian"
    assert result["derivation_chain_length"] == 6
    assert "teukolsky_radial_principal_symbol_is_minus_Delta_xi_r_squared_plus_P_squared" in result["checked_identities"]
    assert "gauge_reduction_xi_r_to_Delta_minus_half_x_sends_radial_hessian_to_minus_x_squared" in result["checked_identities"]
    assert "radial_hessian_coefficient_is_minus_rho_over_Delta" in result["checked_identities"]
    assert "radial_hessian_packet_expression_is_h_times_x_squared" in result["checked_identities"]
    assert result["proves_full_q_grav"] is False


def test_rhedt_rejects_wrong_claim_scope(tmp_path):
    data = _load(CERT)
    data["metadata"]["claim_scope"] = "wrong_scope"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="claim_scope must be"):
        v.verify_file(p)


def test_rhedt_rejects_wrong_derivation_status(tmp_path):
    data = _load(CERT)
    data["metadata"]["derivation_status"] = "candidate"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="derivation_status must be one of"):
        v.verify_file(p)


def test_rhedt_rejects_candidate_provenance_in_chain(tmp_path):
    data = _load(CERT)
    data["derivation_chain"][0]["provenance"] = (
        "candidate Teukolsky radial equation"
    )
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify_file(p)


def test_rhedt_rejects_placeholder_provenance_in_packet(tmp_path):
    data = _load(CERT)
    data["radial_hessian_packet"]["provenance"] = (
        "placeholder radial Hessian packet"
    )
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        v.verify_file(p)


def test_rhedt_rejects_wrong_packet_expression(tmp_path):
    data = _load(CERT)
    # Replace the packet expression with h * x (degree 1 in x, not degree 2).
    data["radial_hessian_packet"]["packet_expression"] = {
        "prod": [
            {"var": "h"},
            {"var": "x"}
        ]
    }
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="must be a square"):
        v.verify_file(p)


def test_rhedt_rejects_wrong_coefficient_variable(tmp_path):
    data = _load(CERT)
    data["radial_hessian_packet"]["coefficient_variable"] = "s"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="coefficient_variable must be 'h'"):
        v.verify_file(p)


def test_rhedt_rejects_wrong_phase_variable(tmp_path):
    data = _load(CERT)
    data["radial_hessian_packet"]["phase_variable"] = "ell"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="phase_variable must be 'x'"):
        v.verify_file(p)


def test_rhedt_rejects_wrong_coefficient_formula(tmp_path):
    data = _load(CERT)
    # Replace the formula with something that doesn't match -rho/Delta.
    data["radial_hessian_coefficient_formula"]["formula"] = "-rho"
    with pytest.raises(ValueError, match="does not match the expected formula"):
        v.verify(data, base_dir=ROOT / "certificates" / "rhedt")


def test_rhedt_rejects_wrong_positivity_weight(tmp_path):
    data = _load(CERT)
    # Use a positivity weight with a forbidden variable.
    data["radial_hessian_coefficient_formula"]["positivity_weight_expression"] = "rho + xi_r"
    with pytest.raises(ValueError, match="must be a function of"):
        v.verify(data, base_dir=ROOT / "certificates" / "rhedt")


def test_rhedt_rejects_missing_trace_path(tmp_path):
    data = _load(CERT)
    data["principal_symbol_trace_certificate_path"] = "nonexistent/path.json"
    with pytest.raises(ValueError, match="principal_symbol_trace_certificate_path does not exist"):
        v.verify(data, base_dir=ROOT / "certificates" / "rhedt")


def test_rhedt_rejects_chart_mismatch_with_trace(tmp_path):
    data = _load(CERT)
    data["chart"] = "different_chart"
    with pytest.raises(ValueError, match="chart must match the principal symbol trace"):
        v.verify(data, base_dir=ROOT / "certificates" / "rhedt")


def test_rhedt_rejects_duplicate_step_names(tmp_path):
    data = _load(CERT)
    data["derivation_chain"][1]["step_name"] = data["derivation_chain"][0]["step_name"]
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="step_name duplicates an earlier step"):
        v.verify_file(p)


def test_rhedt_rejects_empty_derivation_chain(tmp_path):
    data = _load(CERT)
    data["derivation_chain"] = []
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="derivation_chain must be a nonempty list"):
        v.verify_file(p)
