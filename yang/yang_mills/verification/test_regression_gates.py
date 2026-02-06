"""Regression gates to prevent math/paper drift.

These tests are intentionally lightweight and run fast:
- Regenerate the LaTeX macro export (`split/verification_results.tex`).
- Re-run the drift detector to ensure no hard-coded verification constants
  leak back into the compiled manuscript.

They are designed to catch the most damaging class of "mathematical correctness"
errors in this repository: text/code divergence in certified regime boundaries.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SINGLE = (REPO / "single").resolve()


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    """Run a command and return (exit_code, combined_output)."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={
            **os.environ,
            # Avoid user TeX env vars etc. affecting deterministic runs.
            "PYTHONUTF8": "1",
        },
    )
    return proc.returncode, proc.stdout


class RegressionGates(unittest.TestCase):
    def test_export_and_drift_gate(self) -> None:
        # 1) Regenerate exported LaTeX constants.
        export_script = ROOT / "export_results_to_latex.py"
        self.assertTrue(export_script.exists(), "export_results_to_latex.py not found")

        code, out = _run([sys.executable, str(export_script)], cwd=ROOT)
        # When the repo is in theorem-boundary mode, the export script is
        # expected to exit non-zero (to prevent authors from silently updating
        # paper claims). It should still be *reproducible* and produce outputs.
        self.assertIn(
            code,
            {0, 1},
            msg=(
                "export_results_to_latex.py returned an unexpected exit code\n"
                f"--- output ---\n{out}"
            ),
        )

        # Sanity: exported file should exist.
        exported = SINGLE / "verification_results.tex"
        self.assertTrue(exported.exists(), "single/verification_results.tex was not generated")

        # 2) Drift detector must pass.
        drift_script = ROOT / "drift_check_latex_constants.py"
        self.assertTrue(drift_script.exists(), "drift_check_latex_constants.py not found")

        code, out = _run([sys.executable, str(drift_script)], cwd=ROOT)
        self.assertEqual(
            code,
            0,
            msg=(
                "drift_check_latex_constants.py detected drift; paper and certificates diverged\n"
                f"--- output ---\n{out}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
