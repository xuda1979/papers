"""
KnowledgeBase: curated, citable facts for the quantum agent.

Because the harness can *numerically verify* many claims, the knowledge base is
used to (a) seed correct expectations, (b) cross-check agent statements against
the QCOracle, and (c) attach citations/units so reported numbers are auditable.
"""
from __future__ import annotations

from typing import Optional

# ---- curated facts -------------------------------------------------------

ALGORITHM_KNOWLEDGE = {
    "shor": {
        "complexity": "O((log N)^2 (log log N) (log log log N)) gates on O(log N) qubits",
        "cite": "Shor 1994; polynomial-time integer factoring",
        "main_obstacle": "large error-corrected qubit overhead, ~millions of logical qubits",
    },
    "grover": {
        "complexity": "O(sqrt(N)) oracle queries, O(N log N) gates",
        "cite": "Grover 1996; quadratic speedup, optimal by BBBV",
        "main_obstacle": "quadratic (not exponential) speedup; still useful in search",
    },
    "qft": {
        "complexity": "O(n^2) gates, depth O(n)",
        "cite": "Coppersmith 1994; Shor 1994",
        "note": "n = number of qubits",
    },
    "qaoa": {
        "complexity": "p-layer circuit; approximation ratio for MaxCut >= 0.6924 at p=1",
        "cite": "Farhi, Goldstone, Gutmann 2014",
        "note": "noisy intermediate-scale (NISQ) flagship",
    },
    "vqe": {
        "complexity": "variational; no worst-case guarantee",
        "cite": "Peruzzo et al 2014",
        "note": "barren plateau issue for random depth-2 ansatze",
    },
    "simon": {
        "complexity": "O(n) queries, exponential separation against classical",
        "cite": "Simon 1994; oracle separation P^Simon vs BQP",
    },
    "bernstein_vazirani": {
        "complexity": "1 quantum query vs n classical queries",
        "cite": "Bernstein & Vazirani 1993",
    },
    "deutsch_jozsa": {
        "complexity": "1 query vs 2^(n-1)+1 deterministic classical",
        "cite": "Deutsch & Jozsa 1992",
    },
}

QEC_KNOWLEDGE = {
    "steane": {"params": "[[7,1,3]]", "cite": "Steane 1996; CSS from Hamming [7,4,3]"},
    "shor":   {"params": "[[9,1,3]]", "cite": "Shor 1995; first QEC code"},
    "five_qubit": {"params": "[[5,1,3]]", "cite": "Laflamme et al 1996; perfect code"},
    "surface": {
        "params": "[[d^2,1,d]] lattice with d>=3",
        "cite": "Kitaev 1997, Fowler et al 2012",
        "threshold": "~1% fault-tolerant threshold under circuit-level depolarizing noise",
    },
    "threshold_theorem": {
        "claim": "concatenated error correction suppresses logical error below any target if physical error below threshold",
        "cite": "Aharonov & Ben-Or 1997; Knill, Laflamme, Zurek 1998",
    },
}

SIMULATION_KNOWLEDGE = {
    "statevector": {
        "note": "exact; memory O(2^n) complex amplitudes; n<=30 feasible on ~16GB",
    },
    "qiskit_aer": {"note": "local simulator with noise models; supports density-matrix method"},
    "pennylane": {"note": "autodiff + differentiable circuits; good for VQE/QAOA gradients"},
    "qutip": {"note": "open quantum systems (Lindblad), master equation, mesolve"},
}

HARDWARE_KNOWLEDGE = {
    "transmon": {"note": "superconducting; dominant platform (IBM, Google). CX error ~1e-2 to 1e-3"},
    "trapped_ion": {"note": "high fidelity two-qubit gates ~1e-3; slower; Quantinuum, IonQ"},
    "photonic": {"note": "Xanadu; gate-based + CV; fault-tolerance via fusion"},
    "neutral_atom": {"note": "QuEra/Pasqal; native long-range interactions; good for analog"},
    "topological": {"note": "Microsoft; Majorana-based; still early"},
}

FULL_KNOWLEDGE = {
    "algorithms": ALGORITHM_KNOWLEDGE,
    "qec": QEC_KNOWLEDGE,
    "simulation": SIMULATION_KNOWLEDGE,
    "hardware": HARDWARE_KNOWLEDGE,
}


class KnowledgeBase:
    """Queryable, sectioned knowledge store with citation metadata."""

    def __init__(self, data: dict | None = None):
        self.data = data or FULL_KNOWLEDGE

    def sections(self) -> list[str]:
        return list(self.data.keys())

    def lookup(self, section: str, key: str) -> Optional[dict]:
        return self.data.get(section, {}).get(key)

    def query(self, term: str) -> list[tuple[str, str, dict]]:
        """Fuzzy substring search over all knowledge entries."""
        t = term.lower()
        hits = []
        for sec, entries in self.data.items():
            for k, v in entries.items():
                blob = f"{sec} {k} {v}".lower()
                if t in blob:
                    hits.append((sec, k, v))
        return hits

    def all_entries(self) -> list[tuple[str, str, dict]]:
        return [(s, k, v) for s, entries in self.data.items() for k, v in entries.items()]
