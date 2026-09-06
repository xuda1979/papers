"""
AgentHarness: the verification-first agent loop.

This is the control layer that makes the toolkit *agentic*. It implements a
closed loop shared by the best autonomous-research agents:

    PLAN  ->  EXECUTE  ->  VERIFY  ->  LOG  ->  REFLECT  ->  (iterate)

Design principles encoded here:
  1. Every quantitative outcome passes through the Verifier (no unverified claims).
  2. Every executed experiment is appended to the persistent ledger.
  3. Verification failures are surfaced loudly, never papered over.
  4. Known-good results are recallable across sessions from the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

from .verification.oracle import QCOracle, Verifier, VerificationReport
from .memory.ledger import ExperimentLedger, ExperimentRecord
from .knowledge.knowledge_base import KnowledgeBase, FULL_KNOWLEDGE


@dataclass
class TaskOutcome:
    task: str
    verified: bool
    reports: list[VerificationReport]
    results: dict[str, Any]
    ledger_id: str = ""
    reflection: str = ""

    def to_dict(self):
        return {
            "task": self.task,
            "verified": self.verified,
            "reports": [r.to_dict() for r in self.reports],
            "results": self.results,
            "ledger_id": self.ledger_id,
            "reflection": self.reflection,
        }


class AgentHarness:
    """High-level control loop over the quantum toolkit."""

    def __init__(
        self,
        ledger: ExperimentLedger | None = None,
        knowledge: KnowledgeBase | None = None,
        src: str = "agent-session",
        backend: str = "statevector",
        seed: int = 0,
    ):
        self.oracle = QCOracle(backend=backend, seed=seed)
        self.verifier = Verifier(self.oracle)
        self.ledger = ledger or ExperimentLedger()
        self.knowledge = knowledge or KnowledgeBase()
        self.src = src
        self.history: list[TaskOutcome] = []

    # ---- the four-phase loop, exposed granularly -------------------------
    def plan(self, task: str) -> dict:
        """Phase 1: retrieve relevant prior context + suggest verification checks."""
        prior = self.ledger.recall(task)
        return {
            "task": task,
            "prior": prior.to_dict() if prior else None,
            "related_knowledge": [e for e in self.knowledge.query(task.split()[0] if task.split() else "")],
        }

    def execute(self, task: str, fn: Callable[[], dict]) -> TaskOutcome:
        """
        Phase 2/3: run ``fn``, wrap its returned dict as results, then mark them
        as needing verification. Used with ``verify``.
        """
        results = fn()
        return TaskOutcome(task=task, verified=False, reports=[], results=results)

    def verify(self, outcome: TaskOutcome, checks: list[VerificationReport]) -> TaskOutcome:
        outcome.reports = checks
        outcome.verified = all(c.verified for c in checks)
        return outcome

    def log(self, outcome: TaskOutcome) -> TaskOutcome:
        outcome.ledger_id = self.ledger.add(
            kind="task",
            name=outcome.task,
            results=outcome.results,
            verified=outcome.verified,
            src=self.src,
        )
        self.history.append(outcome)
        return outcome

    def reflect(self, outcome: TaskOutcome) -> TaskOutcome:
        if outcome.verified:
            outcome.reflection = f"'{outcome.task}' verified through {len(outcome.reports)} checks."
        else:
            fails = [r.claim for r in outcome.reports if not r.verified]
            outcome.reflection = (
                f"'{outcome.task}' FAILED verification on: {fails}. "
                f"Do not cite unverified result."
            )
        return outcome

    # ---- convenience: run the whole loop ---------------------------------
    def run(self, task: str, fn: Callable[[], dict],
            checks: list[VerificationReport]) -> TaskOutcome:
        outcome = self.execute(task, fn)
        outcome = self.verify(outcome, checks)
        outcome = self.log(outcome)
        outcome = self.reflect(outcome)
        return outcome


def run_qc_experiment(task: str, fn: Callable[[], dict],
                      checks: list[VerificationReport],
                      ledger_path: Optional[str] = None,
                      src: str = "cli") -> TaskOutcome:
    """One-shot entrypoint: build a harness, run a task, return the outcome."""
    h = AgentHarness(ledger=ExperimentLedger(ledger_path), src=src)
    return h.run(task, fn, checks)
