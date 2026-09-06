"""
QCOracle / Verifier: the anti-hallucination layer.

An agent that fabricates quantum claims is worthless; this harness instead
routes every quantitative claim through an *exact numerical oracle* and records
a typed report. The design:

  * ``QCOracle`` bundles the simulator + a registry of known-good factual
    constants (e.g., known algorithm complexities) for cross-checking.
  * ``Verifier`` executes a *claim*: given a circuit and an expected behaviour,
    it returns a structured VerificationReport {verified, margin, evidence}.
  * ``Claim`` objects let the planning layer express intended properties before
    execution --- the harness then *disproves or confirms* them numerically.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import numpy as np

from ..core.simulator import QuantumSimulator
from ..core.circuits import CircuitBuilder
from qiskit import QuantumCircuit


@dataclass
class VerificationReport:
    """Structured output of one verification action."""
    claim: str
    verified: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self):
        return asdict(self)


class QCOracle:
    """Central well-known-facts registry + simulator façade."""

    # Curated, citable complexity facts (useful for cross-checking agent claims).
    KNOWN_COMPLEXITIES = {
        "shor": "O((log N)^3) gates on O(log N) qubits (polynomial)",
        "grover": "O(sqrt(M)) oracle queries on O(log N) qubits",
        "vqe": "variational; no worst-case guarantee",
        "simon": "O(n) oracle queries, O(n) qubits",
        "bernstein_vazirani": "1 quantum query, n qubits",
        "deutsch_jozsa": "1 query, n qubits",
        "qft": "O(n^2) two-qubit gates",
        "teleportation": "3 qubits, 2 classical bits, 1 ebit",
    }

    def __init__(self, backend="statevector", shots=8192, seed=None):
        self.sim = QuantumSimulator(backend=backend, shots=shots, seed=seed)

    def complexity(self, algo: str) -> Optional[str]:
        return self.KNOWN_COMPLEXITIES.get(algo.lower())


class Verifier:
    """Execute claims against the oracle and produce typed reports."""

    def __init__(self, oracle: QCOracle | None = None):
        self.oracle = oracle or QCOracle()
        self.sim = self.oracle.sim

    # ---- concrete checks --------------------------------------------------
    def state_is(self, circuit, target: np.ndarray, tol: float = 1e-9) -> VerificationReport:
        sv = self.sim.statevector(circuit)
        f = float(np.abs(np.vdot(np.asarray(target), sv)) ** 2)
        ok = f >= 1.0 - tol
        return VerificationReport(
            claim="statevector equals target (up to global phase-aware fidelity)",
            verified=bool(ok),
            evidence={"fidelity": f, "tol": tol},
        )

    def outcome_prob_at_least(self, circuit, bitstring: str, p_min: float) -> VerificationReport:
        """Confirm P(measure = bitstring) is consistent with a lower bound."""
        probs = self.sim.probability_distribution(circuit)
        p = probs.get(bitstring, 0.0)
        ok = p + 1e-12 >= p_min
        return VerificationReport(
            claim=f"P({bitstring}) >= {p_min}",
            verified=bool(ok),
            evidence={"p_actual": p, "p_bound": p_min},
        )

    def norm_is(self, vector, expected_norm: float, tol: float = 1e-8) -> VerificationReport:
        v = np.asarray(vector)
        nrm = float(np.linalg.norm(v))
        ok = abs(nrm - expected_norm) < tol
        return VerificationReport(
            claim=f"||v|| == {expected_norm}", verified=bool(ok),
            evidence={"norm": nrm, "tol": tol},
        )

    def unitary_check(self, matrix: np.ndarray, tol: float = 1e-8) -> VerificationReport:
        """Verify a matrix is unitary: U^dag U = I."""
        U = np.asarray(matrix)
        err = float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
        ok = err < tol
        return VerificationReport(
            claim="matrix is unitary",
            verified=bool(ok),
            evidence={"deviation": err, "tol": tol},
        )

    def shots_match_statevector(self, circuit, tol: float = 5e-2) -> VerificationReport:
        """Statistical cross-check: shot histogram vs exact statevector probs."""
        exact = self.sim.probability_distribution(circuit)
        empirical = self.sim.empirical_probs(circuit)
        keys = set(exact) | set(empirical)
        max_dev = max(abs(exact.get(k, 0.0) - empirical.get(k, 0.0)) for k in keys)
        ok = max_dev < tol
        return VerificationReport(
            claim="shot distribution consistent with exact statevector",
            verified=bool(ok),
            evidence={"max_abs_dev": max_dev, "tol": tol},
        )

    def run_all(self, checks: list[VerificationReport]) -> VerificationReport:
        """Aggregate a battery of checks into a single verdict."""
        ok = all(c.verified for c in checks)
        return VerificationReport(
            claim="all checks in battery",
            verified=bool(ok),
            evidence={"n_checks": len(checks), "n_pass": sum(c.verified for c in checks)},
            detail="; ".join(c.claim for c in checks),
        )
