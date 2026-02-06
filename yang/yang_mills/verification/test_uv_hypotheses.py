import json
import os
import sys

# Ensure `verification/` modules are importable when running pytest from repo root.
sys.path.insert(0, os.path.dirname(__file__))

from uv_hypotheses import build_uv_hypotheses, get_uv_parameters, write_uv_hypotheses_json


def test_build_uv_hypotheses_is_deterministic():
    a = build_uv_hypotheses({"clay_standard": False, "claim": "ASSUMPTION-BASED"})
    b = build_uv_hypotheses({"clay_standard": False, "claim": "ASSUMPTION-BASED"})
    assert a["items_sha256"] == b["items_sha256"]
    assert a["counts"]["total"] >= 1


def test_get_uv_parameters_contract():
    params = get_uv_parameters({"claim": "ASSUMPTION-BASED", "clay_standard": False})
    assert "balaban_epsilon" in params
    assert "c1_interval" in params and len(params["c1_interval"]) == 2
    assert "c2_interval" in params and len(params["c2_interval"]) == 2
    assert "strong_u2_prefactor_interval" in params
    assert "crossover_beta" in params
    assert "weak_C_remainder" in params
    assert "flow_C_remainder" in params


def test_write_uv_hypotheses_json_roundtrip(tmp_path):
    path = tmp_path / "uv_hypotheses.json"
    bundle = write_uv_hypotheses_json(str(path), proof_status={"claim": "X", "clay_standard": False})

    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema"] == bundle["schema"]
    assert loaded["items_sha256"] == bundle["items_sha256"]
    assert loaded["counts"] == bundle["counts"]
