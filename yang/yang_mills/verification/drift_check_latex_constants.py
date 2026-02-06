r"""LaTeX drift detector for verification constants.

Goal
----
Prevent "paper drift": numbers derived by the verification suite should not be
hand-copied or retyped across the LaTeX sources.

This script scans `../split/**/*.tex` and flags suspicious hard-coded numeric
literals that match (or are extremely close to) values that *should* come from
`../split/verification_results.tex` or from JSON certificates.

What it checks (conservative)
-----------------------------
1) Whether `split/main.tex` includes `\input{verification_results}`.
2) Whether any `.tex` file (excluding `verification_results.tex` and archives)
   contains hard-coded boundary constants like 0.25/0.40/6.0 in contexts related
   to beta ranges or contraction results.

This is intentionally heuristic so it can run without a full TeX parser.

Usage
-----
    python drift_check_latex_constants.py

Exit codes
----------
0: no drift found
2: drift found
3: unexpected error
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SINGLE = (ROOT / ".." / "single").resolve()

# Files to ignore: generated or archived content.
IGNORE_PARTS = {
    str((SINGLE / "verification_results.tex").resolve()).lower(),
}
IGNORE_DIR_NAMES = {
    "_archive",
    "__pycache__",
    "back_matter",
    # Explicit non-authoritative sources (kept for historical record).
    "_scripts",
    "drafts",
}


SUSPICIOUS_NUMBERS = {
    # Verification / handshake cutoffs that are easy to accidentally hardcode.
    # Prefer macros: \VerBetaStrongMax, \VerBetaIntermediateMax, \VerBetaWeakMin.
    "0.40",
    "6.0",
    "6.00",
    "0.50",
    # "0.015",  # Common SU(2) bound approx, often hardcoded in discussions.
    # Contraction thresholds that should almost always be described symbolically.
    "0.99",
}

# Regex: match numbers as standalone tokens (avoid matching parts of longer numbers).
NUM_TOKEN_RE = re.compile(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)(?![0-9.])")


@dataclass(frozen=True)
class Finding:
    path: Path
    line_no: int
    line: str
    number: str


def should_ignore_path(path: Path) -> bool:
    p = str(path.resolve()).lower()
    if p in IGNORE_PARTS:
        return True
    # Ignore explicit backup/non-authoritative TeX sources.
    if path.name.lower().endswith("_backup.tex"):
        return True
    # Ignore archive trees
    for part in path.parts:
        if part.lower() in IGNORE_DIR_NAMES:
            return True
    return False


def check_main_includes_verification_results() -> list[str]:
    main_tex = SINGLE / "yang_mills_mass_gap.tex"
    if not main_tex.exists():
        return ["single/yang_mills_mass_gap.tex not found"]

    txt = main_tex.read_text(encoding="utf-8", errors="replace")
    if "\\input{verification_results}" not in txt and "\\input{verification_results.tex}" not in txt:
        return ["single/yang_mills_mass_gap.tex does not include \\input{verification_results} (paper may drift from code)"]
    return []


def scan_tex_files() -> list[Finding]:
    findings: list[Finding] = []

    for path in SINGLE.rglob("*.tex"):
        if should_ignore_path(path):
            continue

        try:
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                for m in NUM_TOKEN_RE.finditer(line):
                    num = m.group(1)
                    if num in SUSPICIOUS_NUMBERS:
                        # Allow the dedicated appendices that *derive* certain
                        # small-radius constants to state them explicitly.
                        # We only want to prevent duplicating these numbers
                        # across the manuscript.
                        if path.name.lower() in {"app_mayer_montroll.tex"} and num == "0.015":
                            continue
                        # Allow in obvious macro definitions or already macro-based lines.
                        if "\newcommand" in line or "\Ver" in line:
                            continue
                        # Ignore comments
                        if line.strip().startswith("%"):
                            continue
                        # Ignore Python code snippets (heuristics)
                        if "Interval(" in line or "div_interval(" in line or '"range":' in line or "# Start" in line:
                            continue
                        # Allow common, non-verification constants used as model parameters.
                        # Example: Iwasaki c1=-0.25 appears frequently and is not a CAP output.
                        if "c_1" in line or "Iwasaki" in line:
                            continue
                        findings.append(Finding(path=path, line_no=i, line=line.rstrip(), number=num))
        except OSError:
            continue

    return findings


def main() -> int:
    try:
        problems = check_main_includes_verification_results()
        findings = scan_tex_files()

        if problems:
            print("[FAIL] Configuration issues:")
            for p in problems:
                print(f"  - {p}")

        if findings:
            print("[FAIL] Potential LaTeX drift detected (suspicious hard-coded constants):")
            for f in findings[:200]:
                rel = f.path.relative_to(SINGLE)
                print(f"  - {rel}:{f.line_no}: {f.number} :: {f.line.strip()}")
            if len(findings) > 200:
                print(f"  ... and {len(findings) - 200} more")

        if problems or findings:
            return 2

        print("[OK] No suspicious hard-coded verification constants found in single/*.tex")
        return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
