import os
import sys


def test_verify_axioms_requires_provenance_in_clay_mode(monkeypatch, tmp_path):
    """In Clay-certified mode, verify_axioms should hard-fail if provenance is missing.

    We simulate Clay mode by monkeypatching verify_axioms._load_proof_status.
    Then we point its constants file to a temp artifact without a manifest.
    """

    # Ensure local imports resolve correctly
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    import verify_axioms

    # Create a fake constants file without a provenance manifest.
    constants_path = tmp_path / "rigorous_constants.json"
    constants_path.write_text('{"0.25": {"lsi_constant": {"lower": 1e-3}}}', encoding="utf-8")

    # Force Clay-certified mode
    monkeypatch.setattr(
        verify_axioms,
        "_load_proof_status",
        lambda: {"clay_standard": True, "claim": "CLAY-CERTIFIED"},
    )

    # Force verify_axioms to look for constants in tmp_path.
    monkeypatch.setattr(verify_axioms.os.path, "dirname", lambda _: str(tmp_path))

    # Missing manifest should trigger a failure status in Clay mode.
    rc = verify_axioms.verify_axiom_compliance()
    assert rc != 0, "Expected Clay-mode provenance enforcement to fail on missing manifest"
