"""Tests for the BCCT composed verifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import verify_composed_backend_contract_closure as bcct

BACKEND = (ROOT / "certificates" / "carter_chart_certificates"
           / "spin_half_clean_chart_kerr_source_backend_contract_candidate.json")
GRLEPS = (ROOT / "certificates" / "grleps"
          / "spin_half_clean_chart_gauge_reduced_einstein_principal_symbol.json")
GPDDT = (ROOT / "certificates" / "gpddt"
         / "spin_half_clean_chart_gravitational_density_derivation.json")
TRAPPED = (ROOT / "certificates" / "carter_chart_certificates"
           / "spin_half_clean_chart_trapped_constraint_derivation_certificate.json")
QIEDT = (ROOT / "certificates" / "qiedt"
         / "spin_half_clean_chart_quotient_identity_derivation.json")


def test_accept_valid_closure():
    result = bcct.verify_closure(BACKEND, GRLEPS, GPDDT, TRAPPED, QIEDT)
    assert result["accepted"] is True
    assert result["theorem"] == "BCCT"
    assert result["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert result["proves_full_conjecture"] is False
    assert result["remaining_exact_derivation_obligations"] == []
    assert len(result["closed_obligations"]) == 4
    # All four obligations are closed.
    expected = {
        "derive the gauge-reduced linearized vacuum Einstein principal symbol in this chart",
        "compute the normalized gravitational principal density q_grav in the listed phase variables",
        "derive rho_R and rho_dR as the trapped characteristic constraints R=0 and dR=0",
        "prove the exported source density equals q_grav modulo those trapped constraints",
    }
    assert set(result["closed_obligations"]) == expected
    # All seven identities checked.
    assert len(result["checked_identities"]) == 7
    # Normal form and backend density agree.
    assert result["normal_form"] == result["backend_density"]


def test_main_accepts(capsys):
    rc = bcct.main([
        "main",
        str(BACKEND), str(GRLEPS), str(GPDDT), str(TRAPPED), str(QIEDT),
    ])
    captured = capsys.readouterr()
    assert rc == 0
    out = json.loads(captured.out)
    assert out["accepted"] is True
    assert out["remaining_exact_derivation_obligations"] == []


def test_reject_missing_backend_certificate(tmp_path):
    bad = tmp_path / "missing.json"
    with pytest.raises((OSError, ValueError)):
        bcct.verify_closure(bad, GRLEPS, GPDDT, TRAPPED, QIEDT)


def test_reject_missing_grleps_certificate(tmp_path):
    bad = tmp_path / "missing.json"
    with pytest.raises((OSError, ValueError)):
        bcct.verify_closure(BACKEND, bad, GPDDT, TRAPPED, QIEDT)


def test_reject_missing_gpddt_certificate(tmp_path):
    bad = tmp_path / "missing.json"
    with pytest.raises((OSError, ValueError)):
        bcct.verify_closure(BACKEND, GRLEPS, bad, TRAPPED, QIEDT)


def test_reject_missing_trapped_certificate(tmp_path):
    bad = tmp_path / "missing.json"
    with pytest.raises((OSError, ValueError)):
        bcct.verify_closure(BACKEND, GRLEPS, GPDDT, bad, QIEDT)


def test_reject_missing_qiedt_certificate(tmp_path):
    bad = tmp_path / "missing.json"
    with pytest.raises((OSError, ValueError)):
        bcct.verify_closure(BACKEND, GRLEPS, GPDDT, TRAPPED, bad)


def test_reject_corrupted_grleps(tmp_path):
    """If the GRLEPS cert is corrupted, BCCT must reject."""
    data = json.loads(GRLEPS.read_text())
    del data["metadata"]
    bad = tmp_path / "bad_grleps.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="GRLEPS certificate rejected"):
        bcct.verify_closure(BACKEND, bad, GPDDT, TRAPPED, QIEDT)


def test_reject_corrupted_backend(tmp_path):
    """If the backend cert is corrupted, BCCT must reject."""
    data = json.loads(BACKEND.read_text())
    del data["metadata"]
    bad = tmp_path / "bad_backend.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="backend contract certificate rejected"):
        bcct.verify_closure(bad, GRLEPS, GPDDT, TRAPPED, QIEDT)


def test_reject_chart_mismatch(tmp_path):
    """If the GRLEPS cert has a different chart, BCCT must reject."""
    data = json.loads(GRLEPS.read_text())
    data["chart"] = "some_other_chart"
    bad = tmp_path / "bad_chart.json"
    bad.write_text(json.dumps(data))
    # The GRLEPS verifier itself will reject the bad chart first.
    with pytest.raises(ValueError):
        bcct.verify_closure(BACKEND, bad, GPDDT, TRAPPED, QIEDT)


def test_all_obligations_constant():
    """Sanity check: the ALL_OBLIGATIONS constant has 4 entries."""
    assert len(bcct.ALL_OBLIGATIONS) == 4
    assert len(set(bcct.ALL_OBLIGATIONS)) == 4


def test_closed_sets_partition_obligations():
    """The three closed sets (GRLEPS, GPDDT, TRAPPED) partition the four
    obligations, with GRLEPS covering two."""
    union = bcct.GRLEPS_CLOSED | bcct.GPDDT_CLOSED | bcct.TRAPPED_CLOSED
    assert union == set(bcct.ALL_OBLIGATIONS)
    # GRLEPS closes 2, GPDDT closes 1, TRAPPED closes 1.
    assert len(bcct.GRLEPS_CLOSED) == 2
    assert len(bcct.GPDDT_CLOSED) == 1
    assert len(bcct.TRAPPED_CLOSED) == 1
    # The three sets are pairwise disjoint.
    assert bcct.GRLEPS_CLOSED & bcct.GPDDT_CLOSED == set()
    assert bcct.GRLEPS_CLOSED & bcct.TRAPPED_CLOSED == set()
    assert bcct.GPDDT_CLOSED & bcct.TRAPPED_CLOSED == set()


def test_closure_certificate_valid_json():
    """The BCCT closure certificate is valid JSON and well-formed."""
    p = (ROOT / "certificates" / "bcct"
         / "spin_half_clean_chart_backend_contract_closure.json")
    data = json.loads(p.read_text())
    assert data["metadata"]["theorem"] == "BCCT"
    assert data["chart"] == "spin_half_r_7_2_to_18_5_sub04"
    assert data["remaining_exact_derivation_obligations"] == []
    assert len(data["closed_obligations"]) == 4
    # The obligation closure map covers all 4 obligations.
    assert len(data["obligation_closure_map"]) == 4
    # All input certificate paths are present.
    assert "backend_contract" in data["input_certificates"]
    assert "grleps" in data["input_certificates"]
    assert "gpddt" in data["input_certificates"]
    assert "trapped_constraint_derivation" in data["input_certificates"]
    assert "qiedt" in data["input_certificates"]
