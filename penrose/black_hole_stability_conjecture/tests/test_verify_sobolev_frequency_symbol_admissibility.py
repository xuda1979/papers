"""Tests for the SFSAT verifier and the composed completion+SFSAT verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_sobolev_frequency_symbol_admissibility as vsfsat  # noqa: E402
import verify_composed_completion_with_sfsat as vcomposed  # noqa: E402

TOY_SFSAT = ROOT / "certificates" / "sfsat" / "toy_sfsat_certificate.json"
TOY_PKG = ROOT / "certificates" / "completion_packages" / "toy_unconditional_completion_package.json"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _dump(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    with open(p, "w") as f:
        json.dump(data, f)
    return p


def test_toy_sfsat_accepts():
    result = vsfsat.verify_file(TOY_SFSAT)
    assert result["accepted"] is True
    assert result["theorem"] == "SFSAT"
    assert result["version"] == "1.0"
    assert result["N_shared"] == 16
    assert result["N_min"] == 8
    assert result["layer_count"] == 8
    assert result["block_count"] == 7


def test_toy_composed_accepts():
    result = vcomposed.verify_composed(TOY_PKG, TOY_SFSAT)
    assert result["accepted"] is True
    assert result["composition"] == "completion_package + SFSAT"
    assert result["sfsat_N_shared"] == 16


def test_sfsat_rejects_wrong_theorem_name(tmp_path):
    data = _load(TOY_SFSAT)
    data["theorem"] = "NOT_SFSAT"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="theorem must be 'SFSAT'"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_wrong_version(tmp_path):
    data = _load(TOY_SFSAT)
    data["version"] = "0.9"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="version must be '1.0'"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_small_N_shared(tmp_path):
    """N_shared must be >= N_symbol_class + max_per_block_derivative_loss = 4 + 2 = 6."""
    data = _load(TOY_SFSAT)
    data["sobolev_hierarchy"]["N_shared"] = 5  # too small
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="N_shared .* must be >= N_symbol_class"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_small_N_min(tmp_path):
    """N_min must be >= N_symbol_class + max_per_block_derivative_loss - headroom_floor_int."""
    data = _load(TOY_SFSAT)
    data["sobolev_hierarchy"]["N_min"] = 1  # too small (must be >= 4 + 2 - 1 = 5)
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="N_min .* must be >="):
        vsfsat.verify_file(p)


def test_sfsat_rejects_too_few_layers(tmp_path):
    data = _load(TOY_SFSAT)
    data["frequency_decomposition"]["layer_count"] = 3  # less than minimum 4
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="layer_count .* must be >="):
        vsfsat.verify_file(p)


def test_sfsat_rejects_nonpositive_layer_lower(tmp_path):
    data = _load(TOY_SFSAT)
    data["frequency_decomposition"]["layer_bounds"][0]["lower"] = "0/1"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="lower must be positive"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_layer_gap(tmp_path):
    """Adjacent layers must touch or overlap."""
    data = _load(TOY_SFSAT)
    # introduce a gap between layer 0 (upper 2/1) and layer 1 by bumping layer 1
    # to lower=8/1, upper=16/1 (so lower <= upper but a gap exists)
    data["frequency_decomposition"]["layer_bounds"][1]["lower"] = "8/1"
    data["frequency_decomposition"]["layer_bounds"][1]["upper"] = "16/1"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="adjacent layers must touch or overlap"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_layer0_lower_above_one(tmp_path):
    data = _load(TOY_SFSAT)
    data["frequency_decomposition"]["layer_bounds"][0]["lower"] = "2/1"  # > 1
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="layer_bounds\\[0\\].lower .* must be <= 1"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_wrong_symbol_class_order(tmp_path):
    data = _load(TOY_SFSAT)
    data["symbol_class"]["order"] = 1  # must be 0
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="symbol_class.order must be"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_regularity_scale_mismatch(tmp_path):
    data = _load(TOY_SFSAT)
    data["symbol_class"]["regularity_scale"] = 5  # must match N_symbol_class=4
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="regularity_scale .* must match"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_margin_not_dominating_remainder(tmp_path):
    data = _load(TOY_SFSAT)
    # make margin == remainder for one block
    data["block_margins"][0]["positive_margin"] = "[1/16, 1/16]"
    data["block_margins"][0]["remainder_bound"] = "[1/16, 1/16]"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="positive_margin must strictly dominate"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_margin_floor_too_large(tmp_path):
    data = _load(TOY_SFSAT)
    # set margin_floor higher than the actual margin-remainder gap
    data["block_margins"][0]["margin_floor"] = "1/2"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="inf\\(margin - remainder\\)"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_duplicate_composition_position(tmp_path):
    data = _load(TOY_SFSAT)
    data["block_margins"][1]["composition_position"] = data["block_margins"][0]["composition_position"]
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="composition_position duplicates"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_bad_composition_order_length(tmp_path):
    data = _load(TOY_SFSAT)
    data["composition_order"] = data["composition_order"][:-1]  # too short
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="composition_order must be a list of length"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_composition_order_not_permutation(tmp_path):
    data = _load(TOY_SFSAT)
    data["composition_order"][0] = "not_a_real_block"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="must be one of"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_prefix_headroom_violation(tmp_path):
    """If we put the biggest-derivative-loss block first, the prefix headroom
    after position 1 is N_shared - 2 = 14 >= N_min=8, so this passes. To trigger
    the violation, raise max_per_block_derivative_loss by setting a block to dloss=8."""
    data = _load(TOY_SFSAT)
    # bump trapped_superradiant_morawetz to dloss=8 and update max_per_block_derivative_loss
    for entry in data["block_margins"]:
        if entry["name"] == "trapped_superradiant_morawetz":
            entry["derivative_loss"] = 8
    data["sobolev_hierarchy"]["max_per_block_derivative_loss"] = 8
    # Now N_shared (16) >= 4 + 8 = 12, OK. But N_min must be >= 4 + 8 - 1 = 11.
    data["sobolev_hierarchy"]["N_min"] = 11
    # Update total_derivative_loss
    data["admissibility_budget"]["total_derivative_loss"] = 12  # 1+8+1+1+1+0+0
    # Now the prefix headroom after position 1 (with trapped_superradiant first in order
    # having dloss=8) is 16 - 8 = 8 < N_min=11.
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="prefix headroom"):
        vsfsat.verify_file(p)


def test_sfsat_rejects_product_lower_bound_too_large(tmp_path):
    data = _load(TOY_SFSAT)
    # claim a product lower bound larger than the actual product
    data["admissibility_budget"]["product_lower_bound"] = "1/1"
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="product_lower_bound .* must be <="):
        vsfsat.verify_file(p)


def test_sfsat_rejects_total_loss_mismatch(tmp_path):
    data = _load(TOY_SFSAT)
    data["admissibility_budget"]["total_derivative_loss"] = 7  # actual is 6
    # also bump max_total_derivative_loss and buffer so the <= max check passes
    data["admissibility_budget"]["max_total_derivative_loss"] = 10
    data["admissibility_budget"]["available_derivative_buffer"] = 10
    p = _dump(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="total_derivative_loss .* must equal"):
        vsfsat.verify_file(p)


def test_composed_rejects_claim_scope_mismatch(tmp_path):
    pkg = _load(TOY_PKG)
    sfsat = _load(TOY_SFSAT)
    sfsat["claim_scope"] = "different_scope"
    p_pkg = _dump(tmp_path, "pkg.json", pkg)
    p_sfsat = _dump(tmp_path, "sfsat.json", sfsat)
    result = vcomposed.verify_composed(p_pkg, p_sfsat)
    assert result["accepted"] is False
    assert any("claim_scope mismatch" in e for e in result["errors"])


def test_composed_rejects_sobolev_index_mismatch(tmp_path):
    pkg = _load(TOY_PKG)
    sfsat = _load(TOY_SFSAT)
    sfsat["sobolev_hierarchy"]["N_shared"] = 99
    # Re-validate SFSAT in isolation (should still pass since N_shared=99 >= 4+2=6
    # and 99 - 6 = 93 >= N_min=8). Then the cross-check catches the mismatch.
    p_pkg = _dump(tmp_path, "pkg.json", pkg)
    p_sfsat = _dump(tmp_path, "sfsat.json", sfsat)
    result = vcomposed.verify_composed(p_pkg, p_sfsat)
    assert result["accepted"] is False
    assert any("sobolev index mismatch" in e for e in result["errors"])


def test_composed_rejects_block_margin_mismatch(tmp_path):
    pkg = _load(TOY_PKG)
    sfsat = _load(TOY_SFSAT)
    # change positive_margin for one block in sfsat to something inconsistent with pkg
    for entry in sfsat["block_margins"]:
        if entry["name"] == "linear_to_nonlinear_estimates":
            entry["positive_margin"] = "[1/32, 1/32]"  # mismatch with pkg's [1/8, 1/8]
            # recompute margin_floor for this block so the standalone SFSAT verifier
            # still passes; the cross-check then catches the mismatch.
            entry["margin_floor"] = "1/64"  # 1/32 - 1/64 = 1/64
    # update the budget to keep the standalone SFSAT verifier happy
    sfsat["admissibility_budget"]["product_lower_bound"] = "1/1000000000000"
    sfsat["admissibility_budget"]["margin_floor"] = "1/64"
    p_pkg = _dump(tmp_path, "pkg.json", pkg)
    p_sfsat = _dump(tmp_path, "sfsat.json", sfsat)
    result = vcomposed.verify_composed(p_pkg, p_sfsat)
    assert result["accepted"] is False
    assert any("positive_margin mismatch" in e for e in result["errors"])


def test_composed_rejects_composition_order_not_permutation(tmp_path):
    pkg = _load(TOY_PKG)
    sfsat = _load(TOY_SFSAT)
    sfsat["composition_order"][0] = "not_a_block"
    p_pkg = _dump(tmp_path, "pkg.json", pkg)
    p_sfsat = _dump(tmp_path, "sfsat.json", sfsat)
    result = vcomposed.verify_composed(p_pkg, p_sfsat)
    assert result["accepted"] is False
    # the standalone SFSAT verifier rejects first
    assert any("must be one of" in e or "composition_order" in e for e in result["errors"])
