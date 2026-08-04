"""Tests for the GRLEPS verifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the project root importable.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_carter_gauge_reduced_einstein_principal_symbol as grleps

CERT_PATH = (ROOT / "certificates" / "grleps"
             / "spin_half_clean_chart_gauge_reduced_einstein_principal_symbol.json")


def _load_cert() -> dict:
    with CERT_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_cert(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cert.json"
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return p


# ---------------------------------------------------------------- accept path

def test_accept_valid_certificate():
    result = grleps.verify_file(CERT_PATH)
    assert result["accepted"] is True
    assert result["theorem"] == "GRLEPS"
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["derivation_status"] == "exact_derivation"
    assert result["proves_full_conjecture"] is False
    assert result["gauge_rewrite_rule_count"] == 4
    assert result["reduction_chain_length"] == 4
    assert result["critical_pair_count"] == 0
    assert result["normal_form"] == "h*x**2 + q**2*tau"
    assert result["backend_density"] == "h*x**2 + q**2*tau"
    # Both closed obligations must be present.
    closed = set(result["closed_backend_obligations"])
    assert "derive the gauge-reduced linearized vacuum Einstein principal symbol in this chart" in closed
    assert "prove the exported source density equals q_grav modulo those trapped constraints" in closed
    # All six identities checked.
    assert len(result["checked_identities"]) == 6


def test_accept_via_verify_function_directly():
    data = _load_cert()
    result = grleps.verify(data)
    assert result["accepted"] is True


# ---------------------------------------------------------------- reject paths

def test_reject_missing_metadata():
    data = _load_cert()
    del data["metadata"]
    with pytest.raises(ValueError, match="metadata must be an object"):
        grleps.verify(data)


def test_reject_wrong_theorem_name():
    data = _load_cert()
    data["metadata"]["theorem"] = "NOT_GRLEPS"
    with pytest.raises(ValueError, match="metadata.theorem must be 'GRLEPS'"):
        grleps.verify(data)


def test_reject_wrong_claim_scope():
    data = _load_cert()
    data["metadata"]["claim_scope"] = "wrong_scope"
    with pytest.raises(ValueError, match="metadata.claim_scope must be"):
        grleps.verify(data)


def test_reject_candidate_status():
    data = _load_cert()
    data["metadata"]["status"] = "candidate interface contract"
    with pytest.raises(ValueError, match="metadata.status contains a forbidden substring"):
        grleps.verify(data)


def test_reject_wrong_chart():
    data = _load_cert()
    data["chart"] = "some_other_chart"
    with pytest.raises(ValueError, match="chart must be the spin-half clean chart"):
        grleps.verify(data)


def test_reject_wrong_phase_variables():
    data = _load_cert()
    data["variables"]["phase_variables"] = ["a", "b", "c"]
    with pytest.raises(ValueError, match="phase_variables must be"):
        grleps.verify(data)


def test_reject_overlapping_variables():
    data = _load_cert()
    data["variables"]["symbolic_coefficient_variables"] = ["h", "s", "tau", "c", "x", "bilinear_slack"]
    with pytest.raises(ValueError, match="must be"):
        grleps.verify(data)


def test_reject_wrong_derivation_status():
    data = _load_cert()
    data["derivation_status"] = "candidate"
    with pytest.raises(ValueError, match="derivation_status must be one of"):
        grleps.verify(data)


def test_reject_input_symbol_with_undeclared_variable():
    data = _load_cert()
    data["input_principal_symbol"]["expression"] = (
        "h*x**2 - 2*s*c*x*ell + s*c**2*x**2 + s*ell**2 + tau*q**2 "
        "+ diag_slack*(x**2 + ell**2 + q**2) "
        "+ bilinear_slack*(x*ell + x*q + ell*q) + unknown_var*x**2")
    with pytest.raises(ValueError, match="undeclared symbols"):
        grleps.verify(data)


def test_reject_input_symbol_wrong_phase_degree():
    data = _load_cert()
    # Add a degree-3 phase monomial.
    data["input_principal_symbol"]["expression"] = (
        "h*x**2 - 2*s*c*x*ell + s*c**2*x**2 + s*ell**2 + tau*q**2 "
        "+ diag_slack*(x**2 + ell**2 + q**2) "
        "+ bilinear_slack*(x*ell + x*q + ell*q) + h*x**3")
    with pytest.raises(ValueError, match="phase-homogeneous of degree 2"):
        grleps.verify(data)


def test_reject_input_symbol_not_matching_q_grav():
    data = _load_cert()
    # Drop the radial Hessian packet; the result is no longer congruent
    # to q_grav modulo the augmented ideal.
    data["input_principal_symbol"]["expression"] = (
        "- 2*s*c*x*ell + s*c**2*x**2 + s*ell**2 + tau*q**2 "
        "+ diag_slack*(x**2 + ell**2 + q**2) "
        "+ bilinear_slack*(x*ell + x*q + ell*q)")
    with pytest.raises(ValueError, match="must agree with the GPDDT q_grav"):
        grleps.verify(data)


def test_reject_gauge_rule_not_in_ideal():
    data = _load_cert()
    # Replace rule A's RHS with something that doesn't reduce to the
    # augmented ideal.
    data["gauge_rewrite_rules"][0]["rhs"] = "x*ell"
    with pytest.raises(ValueError, match="lhs - rhs must lie in the augmented ideal"):
        grleps.verify(data)


def test_reject_gauge_rule_phase_degree_increasing():
    data = _load_cert()
    # Add a degree-increasing rule.
    data["gauge_rewrite_rules"].append({
        "name": "bad_degree_increasing",
        "lhs": "x",
        "rhs": "x**3",
        "provenance": "A phase-degree-increasing rule that should be rejected"
    })
    with pytest.raises(ValueError, match="phase-degree-non-increasing"):
        grleps.verify(data)


def test_reject_duplicate_rule_names():
    data = _load_cert()
    data["gauge_rewrite_rules"][1]["name"] = data["gauge_rewrite_rules"][0]["name"]
    with pytest.raises(ValueError, match="gauge_rewrite_rules names must be unique"):
        grleps.verify(data)


def test_reject_empty_critical_pairs_with_non_monomial_gcd():
    """If the rules' LHS have a non-monomial GCD, an explicit critical
    pair must be declared."""
    data = _load_cert()
    # Replace rule D with a rule whose LHS shares a non-monomial factor
    # with rule C's LHS.  rule_C LHS = bilinear_slack*(x*ell + q^2).
    # If rule_D LHS = bilinear_slack*(x*ell + q^2) + bilinear_slack*x*q,
    # then gcd(rule_C, rule_D) = bilinear_slack*(x*ell + q^2), which is
    # NOT a monomial.  But this rule would also fail (G1) because its
    # LHS - 0 must lie in the augmented ideal.  We need a more careful
    # construction.
    # Instead: add a new rule whose LHS = bilinear_slack*(x*ell + q^2) * x
    # = bilinear_slack*(x^2*ell + x*q^2), reducing to 0.  This shares
    # the non-monomial factor bilinear_slack*(x*ell + q^2) with rule C,
    # but the GCD as sympy computes it would be... let's see.
    # Actually, gcd(bilinear_slack*(x*ell+q^2), bilinear_slack*(x^2*ell+x*q^2))
    # = bilinear_slack * gcd(x*ell+q^2, x*(x*ell+q^2))
    # = bilinear_slack * (x*ell + q^2), which is non-monomial.
    data["gauge_rewrite_rules"].append({
        "name": "overlapping_rule",
        "lhs": "bilinear_slack*(x**2*ell + x*q**2)",
        "rhs": "0",
        "provenance": "An overlapping rule that shares a non-monomial factor with rule C"
    })
    # This new rule must also pass (G1).  bilinear_slack*(x^2*ell + x*q^2)
    # = bilinear_slack * x * (x*ell + q^2) = bilinear_slack * x * P_dR,
    # which is in I_trap.  So (G1) passes.  But the empty-critical-pairs
    # declaration should now fail (G2).
    with pytest.raises(ValueError, match="non-monomial GCD"):
        grleps.verify(data)


def test_reject_critical_pairs_justification_missing():
    data = _load_cert()
    data["critical_pairs_justification"] = ""
    with pytest.raises(ValueError, match="critical_pairs_justification must be"):
        grleps.verify(data)


def test_reject_critical_pairs_justification_forbidden_word():
    data = _load_cert()
    data["critical_pairs_justification"] = "The leading monomials are coprime (candidate justification)"
    with pytest.raises(ValueError, match="critical_pairs_justification contains a forbidden substring"):
        grleps.verify(data)


def test_reject_critical_pairs_justification_missing_keyword():
    data = _load_cert()
    data["critical_pairs_justification"] = "The leading monomials have monomial GCDs"
    with pytest.raises(ValueError, match="must mention coprime/disjoint/vacuous"):
        grleps.verify(data)


def test_reject_reduction_chain_wrong_arithmetic():
    data = _load_cert()
    # Corrupt the post-step expression of step 0.
    data["reduction_chain"][0]["post_step_expression"] = (
        "h*x**2 + tau*q**2 + diag_slack*(x**2 + ell**2 + q**2) "
        "+ bilinear_slack*(x*ell + x*q + ell*q) + x**2")
    with pytest.raises(ValueError, match="reduction step arithmetic identity failed"):
        grleps.verify(data)


def test_reject_reduction_chain_unknown_rule():
    data = _load_cert()
    data["reduction_chain"][0]["rule_name"] = "nonexistent_rule"
    with pytest.raises(ValueError, match="must name a gauge_rewrite_rule"):
        grleps.verify(data)


def test_reject_reduction_chain_non_monomial_multiplier():
    data = _load_cert()
    data["reduction_chain"][0]["monomial_multiplier"] = "x + ell"
    with pytest.raises(ValueError, match="monomial_multiplier must be a single monomial"):
        grleps.verify(data)


def test_reject_wrong_on_shell_identity():
    """If the final post-step expression is corrupted to not equal
    rho_backend, (G5) fails."""
    data = _load_cert()
    # Add a residual term at the end.
    data["reduction_chain"][-1]["post_step_expression"] = "h*x**2 + tau*q**2 + x*ell"
    # The arithmetic identity of the last step must still hold; we need
    # to adjust the multiplier to make it hold, OR let it fail at (G5).
    # Actually, the arithmetic identity check will fail first because
    # post != current - lhs*mult + rhs*mult.  Let's just check that.
    with pytest.raises(ValueError, match="arithmetic identity failed"):
        grleps.verify(data)


def test_reject_wrong_closed_obligations_count():
    data = _load_cert()
    data["closed_backend_obligations"] = [
        "derive the gauge-reduced linearized vacuum Einstein principal symbol in this chart"
    ]
    with pytest.raises(ValueError, match="closed_backend_obligations must be a list of length 2"):
        grleps.verify(data)


def test_reject_wrong_closed_obligations_content():
    data = _load_cert()
    data["closed_backend_obligations"] = [
        "derive the gauge-reduced linearized vacuum Einstein principal symbol in this chart",
        "a different obligation that is not one of the two closed ones"
    ]
    with pytest.raises(ValueError, match="closed_backend_obligations must be exactly"):
        grleps.verify(data)


def test_reject_trace_path_forbidden_substring():
    data = _load_cert()
    data["principal_symbol_trace_certificate_path"] = (
        "certificates/some_TODO_path.json")
    with pytest.raises(ValueError, match="forbidden substring"):
        grleps.verify(data)


def test_reject_trace_path_not_json():
    data = _load_cert()
    data["principal_symbol_trace_certificate_path"] = (
        "certificates/carter_chart_certificates/spin_half_clean_chart_principal_symbol_trace_candidate.txt")
    with pytest.raises(ValueError, match="must be a .json path"):
        grleps.verify(data)


def test_reject_non_polynomial_rule_lhs():
    data = _load_cert()
    data["gauge_rewrite_rules"][0]["lhs"] = "1/x"
    with pytest.raises(ValueError, match="is not a polynomial"):
        grleps.verify(data)


def test_reject_rule_rhs_with_auxiliary_variable():
    """The rule RHS must not contain constraint auxiliary variables."""
    data = _load_cert()
    data["gauge_rewrite_rules"][0]["rhs"] = "rho_R"
    with pytest.raises(ValueError, match="undeclared symbols"):
        grleps.verify(data)


def test_module_main_accepts(tmp_path, capsys):
    """The CLI main should exit 0 and print the accepted result."""
    p = _write_cert(tmp_path, _load_cert())
    rc = grleps.main(["main", str(p)])
    captured = capsys.readouterr()
    assert rc == 0
    out = json.loads(captured.out)
    assert out["accepted"] is True


def test_module_main_rejects(tmp_path, capsys):
    """The CLI main should exit 1 on a bad certificate."""
    data = _load_cert()
    del data["metadata"]
    p = _write_cert(tmp_path, data)
    rc = grleps.main(["main", str(p)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "GRLEPS rejected" in captured.err


# ---------------------------------------------------------------- internal APIs

def test_is_in_augmented_ideal_for_zero():
    assert grleps.is_in_augmented_ideal(0) is True


def test_is_in_augmented_ideal_for_sos_defect():
    import sympy as sp
    assert grleps.is_in_augmented_ideal(grleps.SOS_DEFECT) is True


def test_is_in_augmented_ideal_for_constraint_packet():
    assert grleps.is_in_augmented_ideal(grleps.diag_slack * grleps.P_R) is True


def test_is_in_augmented_ideal_rejects_unrelated_polynomial():
    import sympy as sp
    # x*ell alone is not in the augmented ideal (it's a phase monomial
    # that is not a multiple of any trapped constraint).
    assert grleps.is_in_augmented_ideal(grleps.x * grleps.ell) is False


def test_monomials_coprime_disjoint():
    # (x^2, ell^3) share no variable: coprime.
    assert grleps.monomials_coprime((2, 0, 0), (0, 3, 0)) is True


def test_monomials_coprime_shared_variable():
    # (x^2, x*ell) share x: not coprime.
    assert grleps.monomials_coprime((2, 0, 0), (1, 1, 0)) is False


def test_monomial_support_matches_q_grav():
    """Sanity check: q_grav has the expected number of distinct monomials."""
    support = grleps.monomial_support(grleps.Q_GRAV)
    # q_grav = h*x^2 - 2*s*c*x*ell + s*c^2*x^2 + s*ell^2 + tau*q^2
    #          + diag_slack*x^2 + diag_slack*ell^2 + diag_slack*q^2
    #          + bilinear_slack*x*ell + bilinear_slack*x*q + bilinear_slack*ell*q
    # = 11 distinct monomials.
    assert len(support) == 11
