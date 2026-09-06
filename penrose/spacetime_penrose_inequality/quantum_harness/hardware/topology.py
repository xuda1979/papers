"""Device topology helpers: coupling maps, connectivity metrics."""
from __future__ import annotations

import numpy as np
from qiskit.transpiler import CouplingMap


class CouplingMapInfo:
    def __init__(self, edges: list[tuple[int, int]]):
        self.edges = edges
        self.nodes = sorted({u for u, v in edges} | {v for u, v in edges})
        self.map = CouplingMap(couplinglist=edges)

    def n_qubits(self) -> int:
        return len(self.nodes)

    def connectivity(self) -> float:
        max_possible = self.n_qubits() * (self.n_qubits() - 1) / 2
        return len(self.edges) / max_possible if max_possible else 1.0

    def degree(self, q: int) -> int:
        return sum(1 for u, v in self.edges if u == q or v == q)

    def summary(self) -> str:
        return (
            f"CouplingMap: {self.n_qubits()} qubits, {len(self.edges)} edges, "
            f"connectivity={self.connectivity():.3f}"
        )


def random_coupling_map(
    n: int,
    p: float = 0.6,
    seed: int = 0,
    bidirectional: bool = True,
) -> CouplingMapInfo:
    rng = np.random.default_rng(seed)
    edges = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
                if bidirectional:
                    edges.append((j, i))
    return CouplingMapInfo(edges)
