import os
import sys
import unittest
import tempfile
import json

sys.path.insert(0, os.path.dirname(__file__))

from provenance import record_derivation, verify_artifact, MANIFEST_SUFFIX


class TestProvenance(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_and_verify_artifact(self):
        # Create a fake artifact
        artifact_path = os.path.join(self.tmpdir, "test_artifact.json")
        with open(artifact_path, "w") as f:
            json.dump({"foo": 1}, f)

        # Record provenance
        manifest_path = record_derivation(artifact_path)
        self.assertTrue(os.path.isfile(manifest_path))
        self.assertEqual(manifest_path, artifact_path + MANIFEST_SUFFIX)

        # Verify should pass
        self.assertTrue(verify_artifact(artifact_path))

    def test_verify_fails_on_modified_artifact(self):
        artifact_path = os.path.join(self.tmpdir, "modified.json")
        with open(artifact_path, "w") as f:
            json.dump({"bar": 2}, f)

        record_derivation(artifact_path)

        # Now modify the artifact
        with open(artifact_path, "w") as f:
            json.dump({"bar": 999}, f)

        # Verify should fail (raise ValueError)
        with self.assertRaises(ValueError):
            verify_artifact(artifact_path)

    def test_verify_warns_if_manifest_missing(self):
        artifact_path = os.path.join(self.tmpdir, "no_manifest.json")
        with open(artifact_path, "w") as f:
            json.dump({}, f)

        # No manifest recorded -> warn_only mode returns False
        self.assertFalse(verify_artifact(artifact_path, warn_only=True))

    def test_enforce_artifact_warn_only_non_clay(self):
        from provenance import enforce_artifact

        artifact_path = os.path.join(self.tmpdir, "no_manifest2.json")
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write("{}")

        # Non-clay mode: should not raise, returns False.
        ok = enforce_artifact(artifact_path, clay_certified=False, label="no_manifest2")
        self.assertFalse(ok)

    def test_enforce_artifact_raises_in_clay_mode(self):
        from provenance import enforce_artifact

        artifact_path = os.path.join(self.tmpdir, "no_manifest3.json")
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write("{}")

        # Clay mode: missing manifest is a hard failure.
        with self.assertRaises(FileNotFoundError):
            enforce_artifact(artifact_path, clay_certified=True, label="no_manifest3")


if __name__ == "__main__":
    unittest.main()
