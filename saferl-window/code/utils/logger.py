"""
Minimal CSV logger helper (placeholder).
"""
from __future__ import annotations
import csv
from pathlib import Path

class CSVLogger:
    def __init__(self, path, header):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(header)

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()
