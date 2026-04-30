"""Legacy cleanup helper for app174 TeX escape repair.

This file is intentionally import-safe so pytest collection at the repository
root does not execute workspace-mutating cleanup code.
"""

from __future__ import annotations

from pathlib import Path


TARGET = Path("split/app174_holographic_stochastic_transport.tex")


def repair_app174_escapes(path: Path = TARGET) -> bool:
    """Repair historical literal escape artifacts if the legacy file exists."""

    if not path.exists():
        return False


    text = path.read_text(encoding="utf-8")
    text = text.replace(r"\n", "\n")
    text = text.replace(r"\x08egin", r"\begin")
    text = text.replace(r"\x0crac", r"\frac")
    text = text.replace(r"\rangle", r"\rangle")
    text = text.replace(r"\to", r"\to")
    text = text.replace(r"\tau", r"\tau")
    path.write_text(text, encoding="utf-8")
    return True


if __name__ == "__main__":
    repair_app174_escapes()
