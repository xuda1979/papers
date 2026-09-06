"""
Variational quantum algorithms: QAOA for MaxCut / MaxEigenCut and a small VQE.

QAOA is the reference NISQ algorithm and the primary benchmark target for the
harness's near-term tooling. This module builds the circuit, computes the exact
cost function via statevector evolution, and provides a shot-based estimate so
the agent can compare exact vs empirical landscape behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.circuits import CircuitBuilder
from ..core.simulator import QuantumSimulator


@dataclass
class QAOAProblem:
    """An undirected weighted graph MaxCut / Ising instance."""
    n: int
    edges: list[tuple[int, int, float]]      # (u, v, weight)
    name: str = "graph"

    def num_edges(self) -> int:
        return len(self.edges)

    def maxcut_value(self, assignment: list[int]) -> float:
        """Cut value for a bitstring (assignment[i] in {0,1})."""
        val = 0.0
        for u, v, w in self.edges:
            val += w * (assignment[u] != assignment[v])
        return val

    def brute_force_best(self) -> tuple[float, list[int]]:
        """Exact optimum over all 2^n assignments (n small)."""
        import itertools
        best_val, best_assign = -1, None
        for bits in itertools.product([0, 1], repeat=self.n):
            v = self.maxcut_value(list(bits))
            if v > best_val:
                best_val, best_assign = v, list(bits)
        return best_val, best_assign


class QAOABuilder:
    """Construct the QAOA circuit for a given problem and parameters (beta, gamma)."""

    def __init__(self, problem: QAOAProblem):
        self.problem = problem

    def circuit(self, beta: list[float], gamma: list[float]) -> CircuitBuilder:
        p = len(beta)
        assert len(gamma) == p
        n = self.problem.n
        b = CircuitBuilder(n)
        b.uniform_superposition(range(n))
        for l in range(p):
            # Problem (cost) unitary: exp(-i gamma_l * sum_edges w Z_u Z_v)
            for (u, v, w) in self.problem.edges:
                b.arbitrary_pauli_rotation("ZZ", 2 * gamma[l] * w, [u, v])
            # Mixer unitary: exp(-i beta_l * sum_i X_i)
            for i in range(n):
                b.rx(2 * beta[l], i)
        return b

    def cost_function(self, beta, gamma, backend="statevector") -> callable:
        b = self.circuit(beta, gamma)
        sim = QuantumSimulator(backend=backend)
        probs = sim.probability_distribution(b)
        n = self.problem.n
        total = 0.0
        for bs, pr in probs.items():
            assign = [int(c) for c in bs]
            total += self.problem.maxcut_value(assign) * pr
        return -total      # negate: max cut = minimize -cut

    def expected_cut(self, beta, gamma) -> float:
        return -self.cost_function(beta, gamma)


def random_qaoa_results(
    n: int = 4,
    p: int = 1,
    seed: int = 0,
    trials: int = 50,
) -> dict:
    """Sample random (beta,gamma) and report best expected cut vs brute-force optimum."""
    from ..verification.oracle import QCOracle
    rng = np.random.default_rng(seed)
    edges = [(i, j, float(rng.integers(1, 5))) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
    problem = QAOAProblem(n=n, edges=edges, name=f"random_n{n}")
    opt, assign = problem.brute_force_best()
    qaoa = QAOABuilder(problem)
    best_expected = -1e9
    best_params = None
    for _ in range(trials):
        beta = rng.uniform(0, np.pi, p).tolist()
        gamma = rng.uniform(0, np.pi, p).tolist()
        ex = qaoa.expected_cut(beta, gamma)
        if ex > best_expected:
            best_expected, best_params = ex, (beta, gamma)
    return {
        "n": n, "p": p, "edges": problem.edges,
        "optimum": opt, "opt_assignment": assign,
        "best_expected_cut": float(best_expected),
        "best_params": best_params,
        "approx_ratio": float(best_expected / opt) if opt > 0 else 1.0,
        "seed": seed, "trials": trials,
    }
