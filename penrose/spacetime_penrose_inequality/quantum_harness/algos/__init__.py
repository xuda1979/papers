"""Curated, verifiable quantum algorithm implementations (benchmark suite)."""
from .core_algorithms import (
    deutsch_jozsa, bernstein_vazirani, simon, grover_search, teleport,
    qft_circuit, shor_period_lookup,
)
from .variational import QAOABuilder, random_qaoa_results

__all__ = [
    "deutsch_jozsa", "bernstein_vazirani", "simon", "grover_search",
    "teleport", "qft_circuit", "shor_period_lookup",
    "QAOABuilder", "random_qaoa_results",
]
