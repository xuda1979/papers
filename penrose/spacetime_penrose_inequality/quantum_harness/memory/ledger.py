"""
ExperimentLedger: durable, JSON-backed record of every claim, verification, and
experiment the agent performs.

SOTA agent principle: an agent without memory repeats its mistakes Mk. II.
This ledger gives the harness:
  * append-only provenance of every executed experiment,
  * JSON on disk (git-friendly, mergeable, diffable),
  * query/recall so later sessions reuse validated results instead of
    recomputing or contradicting earlier ones.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4


@dataclass
class ExperimentRecord:
    kind: str                          # 'simulation' | 'verification' | 'qec' | 'benchmark' | 'transpile' | ...
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    verified: bool | None = None
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    src: str = ""                       # optional provenance / session marker

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentLedger:
    """Append-only, JSON-backed ledger with in-memory index for fast lookup."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(
            os.environ.get("QUANTUM_LEDGER_DIR", os.getcwd()),
            "quantum_ledger.json",
        )
        self._records: list[ExperimentRecord] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self._records = [ExperimentRecord(**r) for r in data]
            except (json.JSONDecodeError, TypeError, KeyError):
                self._records = []

    def record(self, rec: ExperimentRecord) -> str:
        self._records.append(rec)
        self._persist()
        return rec.id

    def add(
        self,
        kind: str,
        name: str,
        results: dict | None = None,
        params: dict | None = None,
        verified: bool | None = None,
        src: str = "",
    ) -> str:
        return self.record(
            ExperimentRecord(
                kind=kind, name=name,
                params=params or {}, results=results or {},
                verified=verified, src=src,
            )
        )

    def _persist(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([r.to_dict() for r in self._records], f, indent=2, default=str)

    # ---- queries ---------------------------------------------------------
    def all(self) -> list[ExperimentRecord]:
        return list(self._records)

    def count(self) -> int:
        return len(self._records)

    def by_kind(self, kind: str) -> list[ExperimentRecord]:
        return [r for r in self._records if r.kind == kind]

    def by_name(self, name: str) -> list[ExperimentRecord]:
        return [r for r in self._records if r.name == name]

    def recall(self, name: str, kind: Optional[str] = None) -> Optional[ExperimentRecord]:
        for r in reversed(self._records):
            if r.name == name and (kind is None or r.kind == kind):
                return r
        return None

    def last_verified(self, kind: str = None) -> Optional[ExperimentRecord]:
        for r in reversed(self._records):
            if r.verified and (kind is None or r.kind == kind):
                return r
        return None

    def to_markdown(self) -> str:
        if not self._records:
            return "# Quantum Experiment Ledger\n\n*(empty)*\n"
        lines = ["# Quantum Experiment Ledger", ""]
        for r in reversed(self._records):
            status = "PASS" if r.verified else ("FAIL" if r.verified is False else "n/a")
            lines.append(f"- `[{r.kind}]` **{r.name}** ({r.id}) — {status}")
            if r.results:
                lines.append(f"  results: `{json.dumps(r.results, default=str)[:220]}`")
        return "\n".join(lines) + "\n"
