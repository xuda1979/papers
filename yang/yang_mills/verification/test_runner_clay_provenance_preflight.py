import os
import sys
import json


def _write_manifest(
    path,
    *,
    kind: str,
    phase: str | None = None,
    source_files: list[str] | None = None,
):
    """Write a minimal manifest with correct sha and requested extra metadata."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    manifest = {
        "artifact": os.path.basename(path),
        "artifact_sha256": h.hexdigest(),
        "source_files_sha256": {p: "PLACEHOLDER" for p in (source_files or [])},
        "generated_utc": "1970-01-01T00:00:00Z",
        "python_version": "0.0.0",
        "extra": {"kind": kind},
    }
    if phase is not None:
        manifest["extra"]["phase"] = phase
    with open(path + ".provenance.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def test_runner_preflight_fails_without_manifests(monkeypatch, tmp_path):
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    import certificate_runner_v2

    # Create fake required artifacts without provenance manifests.
    (tmp_path / "rigorous_constants.json").write_text("{}", encoding="utf-8")
    (tmp_path / "uv_hypotheses.json").write_text("{}", encoding="utf-8")
    (tmp_path / "verification_results.json").write_text("{}", encoding="utf-8")

    # Create some certificate artifacts too; preflight will require them if present.
    (tmp_path / "certificate_phase1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "certificate_phase2_hardened.json").write_text("{}", encoding="utf-8")

    # Force the runner to treat tmp_path as its directory.
    monkeypatch.setattr(certificate_runner_v2.os.path, "dirname", lambda _: str(tmp_path))

    proof_status = {"clay_standard": True, "claim": "CLAY-CERTIFIED"}
    rc = certificate_runner_v2._clay_provenance_preflight(proof_status)
    assert rc != 0


def test_runner_preflight_fails_on_metadata_mismatch(monkeypatch, tmp_path):
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    import certificate_runner_v2

    # Required artifacts exist
    rc_path = tmp_path / "rigorous_constants.json"
    uv_path = tmp_path / "uv_hypotheses.json"
    vr_path = tmp_path / "verification_results.json"
    rc_path.write_text("{}", encoding="utf-8")
    uv_path.write_text("{}", encoding="utf-8")
    vr_path.write_text("{}", encoding="utf-8")

    # Manifests exist but one has wrong metadata.
    # (Include required source file entries so we isolate the metadata failure.)
    src_common = [
        str(tmp_path / "interval_arithmetic.py"),
        str(tmp_path / "uv_hypotheses.py"),
        str(tmp_path / "export_results_to_latex.py"),
    ]
    (tmp_path / "interval_arithmetic.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "uv_hypotheses.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "export_results_to_latex.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "rigorous_constants_derivation.py").write_text("# stub", encoding="utf-8")

    _write_manifest(
        str(rc_path),
        kind="not_constants",
        source_files=src_common + [str(tmp_path / "rigorous_constants_derivation.py")],
    )
    _write_manifest(str(uv_path), kind="uv_obligations", source_files=src_common)
    _write_manifest(str(vr_path), kind="results", source_files=src_common)

    monkeypatch.setattr(certificate_runner_v2.os.path, "dirname", lambda _: str(tmp_path))

    proof_status = {"clay_standard": True, "claim": "CLAY-CERTIFIED"}
    rc = certificate_runner_v2._clay_provenance_preflight(proof_status)
    assert rc != 0


def test_runner_preflight_passes_with_correct_metadata(monkeypatch, tmp_path):
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    import certificate_runner_v2

    rc_path = tmp_path / "rigorous_constants.json"
    uv_path = tmp_path / "uv_hypotheses.json"
    vr_path = tmp_path / "verification_results.json"
    rc_path.write_text("{}", encoding="utf-8")
    uv_path.write_text("{}", encoding="utf-8")
    vr_path.write_text("{}", encoding="utf-8")

    src_common = [
        str(tmp_path / "interval_arithmetic.py"),
        str(tmp_path / "uv_hypotheses.py"),
        str(tmp_path / "export_results_to_latex.py"),
    ]
    (tmp_path / "interval_arithmetic.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "uv_hypotheses.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "export_results_to_latex.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "rigorous_constants_derivation.py").write_text("# stub", encoding="utf-8")

    _write_manifest(
        str(rc_path),
        kind="constants",
        source_files=src_common + [str(tmp_path / "rigorous_constants_derivation.py")],
    )
    _write_manifest(str(uv_path), kind="uv_obligations", source_files=src_common)
    _write_manifest(str(vr_path), kind="results", source_files=src_common)

    monkeypatch.setattr(certificate_runner_v2.os.path, "dirname", lambda _: str(tmp_path))

    proof_status = {"clay_standard": True, "claim": "CLAY-CERTIFIED"}
    rc = certificate_runner_v2._clay_provenance_preflight(proof_status)
    assert rc == 0


def test_runner_preflight_fails_when_required_sources_missing(monkeypatch, tmp_path):
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    import certificate_runner_v2

    rc_path = tmp_path / "rigorous_constants.json"
    uv_path = tmp_path / "uv_hypotheses.json"
    vr_path = tmp_path / "verification_results.json"
    rc_path.write_text("{}", encoding="utf-8")
    uv_path.write_text("{}", encoding="utf-8")
    vr_path.write_text("{}", encoding="utf-8")

    # Provide correct kinds, but omit required source entries.
    _write_manifest(str(rc_path), kind="constants", source_files=[])
    _write_manifest(str(uv_path), kind="uv_obligations", source_files=[])
    _write_manifest(str(vr_path), kind="results", source_files=[])

    monkeypatch.setattr(certificate_runner_v2.os.path, "dirname", lambda _: str(tmp_path))

    proof_status = {"clay_standard": True, "claim": "CLAY-CERTIFIED"}
    rc = certificate_runner_v2._clay_provenance_preflight(proof_status)
    assert rc != 0
