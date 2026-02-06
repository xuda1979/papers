import os
import sys


def test_record_provenance_manifests_creates_manifests(monkeypatch, tmp_path):
    """Generator should create <artifact>.provenance.json for present artifacts."""

    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    import record_provenance_manifests

    # Create a minimal set of artifacts so the generator has something to do.
    (tmp_path / "rigorous_constants.json").write_text("{}", encoding="utf-8")
    (tmp_path / "uv_hypotheses.json").write_text("{}", encoding="utf-8")
    (tmp_path / "verification_results.json").write_text("{}", encoding="utf-8")

    # Point generator to tmp_path by monkeypatching its directory resolution.
    monkeypatch.setattr(record_provenance_manifests.os.path, "dirname", lambda _: str(tmp_path))

    rc = record_provenance_manifests.main()
    assert rc == 0

    assert (tmp_path / "rigorous_constants.json.provenance.json").is_file()
    assert (tmp_path / "uv_hypotheses.json.provenance.json").is_file()
    assert (tmp_path / "verification_results.json.provenance.json").is_file()


def test_record_provenance_manifests_ok_when_missing(monkeypatch, tmp_path):
    """Generator should be robust when most artifacts are missing."""

    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    import record_provenance_manifests

    # Only one file exists.
    (tmp_path / "uv_hypotheses.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(record_provenance_manifests.os.path, "dirname", lambda _: str(tmp_path))

    rc = record_provenance_manifests.main()
    assert rc == 0
    assert (tmp_path / "uv_hypotheses.json.provenance.json").is_file()
