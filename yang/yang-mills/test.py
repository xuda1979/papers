"""Legacy cleanup helper for app174 TeX control characters.

Kept import-safe so pytest collection does not mutate files or fail when the
historical app174 source is absent.
"""

from __future__ import annotations

from pathlib import Path


TARGET = Path("split/app174_holographic_stochastic_transport.tex")


def repair_app174_control_chars(path: Path = TARGET) -> bool:
    if not path.exists():
        return False

    text = path.read_text(encoding="utf-8")
    text = text.replace("	", "\t")
    text = text.replace("\x08", "\b")
    text = text.replace("\x0c", "\f")
    path.write_text(text, encoding="utf-8")
    return True


if __name__ == "__main__":
    repair_app174_control_chars()
