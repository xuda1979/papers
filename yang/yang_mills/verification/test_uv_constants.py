import json
import os
import sys

# Ensure `verification/` modules are importable when running pytest from repo root.
sys.path.insert(0, os.path.dirname(__file__))

from uv_constants_derivation import derive_uv_constants, write_uv_constants_json
from uv_constants import get_uv_constants


def test_derive_uv_constants_schema_and_fields():
    bundle = derive_uv_constants(proof_status={"clay_standard": False, "claim": "X"})
    assert bundle["schema"] == "yangmills.uv_constants.v1"
    assert "derived" in bundle
    assert "epsilon_Balaban" in bundle["derived"]
    eps = float(bundle["derived"]["epsilon_Balaban"])
    assert 0.01 <= eps <= 0.25


def test_write_and_load_uv_constants_json_roundtrip(tmp_path, monkeypatch):
    # Write artifact into a temp dir, then monkeypatch module dir resolution by
    # creating a sibling `uv_constants.json` next to uv_constants.py isn't easy.
    # Instead, validate JSON structure and that loader falls back cleanly.
    out = tmp_path / "uv_constants.json"
    bundle = write_uv_constants_json(str(out), proof_status={"clay_standard": False, "claim": "X"})

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema"] == bundle["schema"]
    assert loaded["derived"]["beta_handoff"] == 6.0

    # Loader should always return the expected keys (from artifact if present,
    # otherwise from fallback); here we just sanity-check the fallback contract.
    c = get_uv_constants(proof_status={"clay_standard": False, "claim": "X"})
    assert c["schema"] == "yangmills.uv_constants.v1"
    assert "epsilon_Balaban" in c["derived"]
