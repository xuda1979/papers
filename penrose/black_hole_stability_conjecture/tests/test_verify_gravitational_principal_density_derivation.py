"""Tests for the GPDDT verifier and the composed symbolic + GPDDT verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_gravitational_principal_density_derivation as vgp  # noqa: E402
import verify_composed_gpddt_with_symbolic_normal_form as vcomposed  # noqa: E402

TOY_GPDDT = ROOT / "certificates" / "gpddt" / "spin_half_clean_chart_gravitational_density_derivation.json"
SYMBOLIC = (ROOT / "certificates" / "carter_chart_certificates"
            / "spin_half_clean_chart_symbolic_kerr_identification_candidate.json")


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _dump(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    with open(p, "w") as f:
        json.dump(data, f)
    return p


def test_toy_gpddt_accepts():
    result = vgp.verify_file(TOY_GPDDT)
    assert result["accepted"] is True
    assert result["theorem"] == "GPDDT"
    assert result["version"] == "1.0"
    assert result["spin"] == "1/2"
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["background"] == "fixed_subextremal_kerr"
    assert result["principal_density_packet_count"] == 11
    assert result["constraint_generator_count"] == 0
    assert result["total_packet_count"] == 11
    assert result["max_packet_phase_degree"] == 2
    assert result["identity_holds"] is True


def test_gpddt_rejects_wrong_version(tmp_path):
    data = _load(TOY_GPDDT)
    data["version"] = "0.9"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="root.version must be '1.0'"):
        vgp.verify_file(p)


def test_gpddt_rejects_wrong_claim_scope(tmp_path):
    data = _load(TOY_GPDDT)
    data["claim_scope"] = "wrong_scope"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="root.claim_scope must be"):
        vgp.verify_file(p)


def test_gpddt_rejects_candidate_provenance(tmp_path):
    data = _load(TOY_GPDDT)
    data["linearized_einstein_packets"][0]["provenance"] = (
        "candidate radial Hessian packet of the linearized Kerr principal symbol"
    )
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        vgp.verify_file(p)


def test_gpddt_rejects_placeholder_provenance(tmp_path):
    data = _load(TOY_GPDDT)
    data["linearized_einstein_packets"][0]["provenance"] = (
        "placeholder radial Hessian packet"
    )
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        vgp.verify_file(p)


def test_gpddt_rejects_toy_provenance(tmp_path):
    data = _load(TOY_GPDDT)
    data["linearized_einstein_packets"][0]["provenance"] = (
        "toy radial Hessian packet of the linearized Kerr principal symbol"
    )
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="provenance contains a forbidden substring"):
        vgp.verify_file(p)


def test_gpddt_rejects_packet_with_wrong_phase_degree(tmp_path):
    data = _load(TOY_GPDDT)
    # Replace a phase-degree-2 packet with a phase-degree-3 expression.
    data["linearized_einstein_packets"][0]["expression"] = {
        "prod": [
            {"var": "h"},
            {"var": "x"},
            {"var": "x"},
            {"var": "x"}
        ]
    }
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="must be homogeneous of phase degree 2"):
        vgp.verify_file(p)


def test_gpddt_rejects_derived_expression_mismatch(tmp_path):
    data = _load(TOY_GPDDT)
    # Drop one summand from the derived density expression so the declared
    # linear combination no longer matches the declared density expression.
    data["gravitational_density_derivation"]["derived_density_expression"] = {
        "sum": data["gravitational_density_derivation"]["derived_density_expression"]["sum"][:-1]
    }
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="does not match the declared linear combination"):
        vgp.verify_file(p)


def test_gpddt_rejects_identity_check_false(tmp_path):
    data = _load(TOY_GPDDT)
    data["gravitational_density_derivation"]["identity_check"][
        "difference_reduces_to_zero_modulo_ideal"] = False
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="difference_reduces_to_zero_modulo_ideal must be true"):
        vgp.verify_file(p)


def test_gpddt_rejects_missing_target_path(tmp_path):
    data = _load(TOY_GPDDT)
    data["gravitational_density_derivation"]["target_symbolic_normal_form_path"] = (
        "certificates/carter_chart_certificates/nonexistent.json"
    )
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="target_symbolic_normal_form_path does not exist"):
        vgp.verify_file(p)


def test_gpddt_rejects_unknown_packet_in_coefficients(tmp_path):
    data = _load(TOY_GPDDT)
    data["gravitational_density_derivation"]["derivation_coefficients"][
        "principal_density"][0]["packet_name"] = "nonexistent_packet"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="references unknown packet"):
        vgp.verify_file(p)


def test_gpddt_rejects_primitive_formula_count_mismatch(tmp_path):
    data = _load(TOY_GPDDT)
    data["provenance_audit"]["primitive_formula_count"] = 999
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="primitive_formula_count .* must equal"):
        vgp.verify_file(p)


def test_gpddt_rejects_noncandidate_count_mismatch(tmp_path):
    data = _load(TOY_GPDDT)
    data["provenance_audit"]["noncandidate_primitive_formula_count"] = 999
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="noncandidate_primitive_formula_count .* must equal"):
        vgp.verify_file(p)


def test_gpddt_rejects_derivation_budget_count_mismatch(tmp_path):
    data = _load(TOY_GPDDT)
    data["derivation_budget"]["principal_density_packet_count"] = 999
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="principal_density_packet_count .* must equal"):
        vgp.verify_file(p)


def test_composed_accepts_symbolic_plus_gpddt():
    result = vcomposed.verify_composed(SYMBOLIC, TOY_GPDDT)
    assert result["accepted"] is True
    assert result["composition"] == "symbolic_normal_form + GPDDT"
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["spin"] == "1/2"
    assert result["identity_holds"] is True
    assert result["symbolic_accepted"] is True
    assert result["gpddt_accepted"] is True


def test_composed_rejects_chart_mismatch(tmp_path):
    # Keep the symbolic cert at its original path so its relative extraction
    # path resolves; only the GPDDT certificate goes to tmp_path.
    gpddt = _load(TOY_GPDDT)
    gpddt["chart"] = "different_chart_name"
    p_gpddt = _dump(tmp_path, "gpddt.json", gpddt)
    result = vcomposed.verify_composed(SYMBOLIC, p_gpddt)
    assert result["accepted"] is False
    assert any("chart mismatch" in e for e in result["errors"])


def test_composed_rejects_phase_variable_mismatch(tmp_path):
    gpddt = _load(TOY_GPDDT)
    # Change the third phase variable name. The GPDDT verifier itself fails
    # first because the packet expressions reference the original 'q'.
    gpddt["variables"]["phase_variables"] = ["x", "ell", "p"]
    p_gpddt = _dump(tmp_path, "gpddt.json", gpddt)
    result = vcomposed.verify_composed(SYMBOLIC, p_gpddt)
    assert result["accepted"] is False
    # The rejection may surface either as a GPDDT-side parse failure or as a
    # cross-certificate phase-variables mismatch.
    assert any("phase variables mismatch" in e or "unknown variable" in e
               for e in result["errors"])


def test_composed_rejects_target_path_mismatch(tmp_path):
    # Make a duplicate of the symbolic cert at a different path so the GPDDT
    # target_symbolic_normal_form_path can point to a *valid* symbolic cert
    # (so the GPDDT verifier itself passes) while the cross-certificate
    # path-agreement check fails.
    sym = _load(SYMBOLIC)
    # Write the duplicate inside the project's certificates directory so the
    # GPDDT verifier's relative path resolution can find it.
    duplicate_path = ROOT / "certificates" / "gpddt" / "_test_duplicate_symbolic.json"
    duplicate_path.parent.mkdir(parents=True, exist_ok=True)
    with open(duplicate_path, "w") as f:
        json.dump(sym, f)
    try:
        gpddt = _load(TOY_GPDDT)
        gpddt["gravitational_density_derivation"]["target_symbolic_normal_form_path"] = (
            "certificates/gpddt/_test_duplicate_symbolic.json"
        )
        p_gpddt = _dump(tmp_path, "gpddt.json", gpddt)
        result = vcomposed.verify_composed(SYMBOLIC, p_gpddt)
        assert result["accepted"] is False
        assert any("target_symbolic_normal_form_path" in e and "must reference" in e
                   for e in result["errors"])
    finally:
        duplicate_path.unlink(missing_ok=True)


def test_composed_rejects_derived_density_not_matching_symbolic(tmp_path):
    gpddt = _load(TOY_GPDDT)
    # Corrupt the derived density expression so the polynomial-level identity
    # check fails. The GPDDT verifier itself will also fail because the
    # declared linear combination no longer matches the declared density.
    gpddt["gravitational_density_derivation"]["derived_density_expression"]["sum"].append(
        {"prod": [{"var": "h"}, {"square": {"var": "q"}}]}
    )
    # Also append a new packet so the linear-combination budget balances.
    new_packet = {
        "name": "extra_packet",
        "role": "principal_density_packet",
        "provenance": (
            "Extra diagonal packet of the linearized spin-half Kerr principal "
            "symbol, derived from an additional Bianchi identity contraction."
        ),
        "expression": {"prod": [{"var": "h"}, {"square": {"var": "q"}}]},
    }
    gpddt["linearized_einstein_packets"].append(new_packet)
    gpddt["gravitational_density_derivation"]["derivation_coefficients"][
        "principal_density"].append(
        {"packet_name": "extra_packet", "coefficient": "1"}
    )
    gpddt["provenance_audit"]["primitive_formula_count"] = 12
    gpddt["provenance_audit"]["noncandidate_primitive_formula_count"] = 12
    gpddt["derivation_budget"]["principal_density_packet_count"] = 12
    gpddt["derivation_budget"]["total_packet_count"] = 12
    p_gpddt = _dump(tmp_path, "gpddt.json", gpddt)
    result = vcomposed.verify_composed(SYMBOLIC, p_gpddt)
    assert result["accepted"] is False
    assert any("reduced modulo the constraint ideal, is nonempty" in e
               or "does not match the declared linear combination" in e
               for e in result["errors"])
