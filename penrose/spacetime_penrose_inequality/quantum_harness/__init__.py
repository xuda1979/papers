"""
QuantumComputing Harness
=======================
A state-of-the-art agentic harness for quantum computing science, technology,
and development. It combines:

  * an exact *numerical truth oracle* (QCOracle/Verifier) so the agent's claims
    are always checked, never guessed;
  * a curated *verified algorithm + QEC + noise + hardware* toolkit;
  * *persistent experiment memory* (ExperimentLedger) for cross-session
    continuity;
  * a *verification-first agent loop* (AgentHarness) that plans -> executes ->
    verifies -> logs -> reflects.

Submodules:
  core          : circuits, exact/statistical simulation, noise models
  qec           : stabilizer formalism, [[n,k,d]] derivation, syndrome decode
  algos         : verified reference algorithms + QAOA/VQE
  verification  : QCOracle + Verifier (the truth oracle layer)
  hardware      : topology + transpilation (with unitary-equivalence checks)
  knowledge     : citable domain knowledge store
  memory        : persistent experiment ledger
  orchestrator  : the agent loop and CLI
"""
from .orchestrator import AgentHarness, run_qc_experiment

__version__ = "0.1.0"

__all__ = ["AgentHarness", "run_qc_experiment", "__version__"]
