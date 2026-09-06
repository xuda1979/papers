"""
QuantumSimulator: deterministic, numerically-grounded execution of circuits.

This is the agent's "truth oracle" --- every claimed quantum outcome in the
harness should be checked here by exact or shot-based simulation before being
reported. Two backends:
  * ``statevector``: exact amplitudes -> entanglement entropy, probabilities,
    expectation values, fidelities.
  * ``aer``: shot-based measurement statistics with an optional noise model.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy
from qiskit_aer import AerSimulator

from .circuits import CircuitBuilder


class SimBackend:
    STATEVECTOR = "statevector"
    AER = "aer"


class QuantumSimulator:
    def __init__(
        self,
        backend: str = SimBackend.STATEVECTOR,
        shots: int = 8192,
        noise_model=None,
        seed: int | None = None,
    ):
        self.backend = backend
        self.shots = shots
        self.noise_model = noise_model
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ---- exact statevector machinery -----------------------------------
    def statevector(self, circuit: QuantumCircuit | CircuitBuilder) -> np.ndarray:
        qc = circuit.circuit() if isinstance(circuit, CircuitBuilder) else circuit
        if qc.num_clbits and not any(isinstance(i, (int, np.integer)) for i in qc.cregs[0]):
            pass
        sv = Statevector(qc)
        return np.asarray(sv.data)

    def probability_distribution(self, circuit) -> dict[str, float]:
        amp = np.abs(self.statevector(circuit)) ** 2
        n = int(np.log2(len(amp)))
        return {f"{i:0{n}b}": float(a) for i, a in enumerate(amp)}

    def amplitudes(self, circuit) -> dict[str, complex]:
        sv = self.statevector(circuit)
        n = int(np.log2(len(sv)))
        return {f"{i:0{n}b}": sv[i] for i in range(len(sv))}

    # ---- entropies / entanglement --------------------------------------
    def reduced_density_matrix(self, circuit, qubits: list[int]) -> np.ndarray:
        """Partial trace onto the subsystems in ``qubits`` (global ordering)."""
        qc = circuit.circuit() if isinstance(circuit, CircuitBuilder) else circuit
        rho = DensityMatrix(qc)
        # qiskit partial_trace takes trace_out indices
        trace_out = [i for i in range(qc.num_qubits) if i not in qubits]
        rhoA = partial_trace(rho, trace_out)
        return np.asarray(rhoA.data)

    def von_neumann_entropy(self, circuit, qubits: list[int]) -> float:
        rhoA = self.reduced_density_matrix(circuit, qubits)
        return float(entropy(rhoA, base=2))

    def purity(self, circuit) -> float:
        qc = circuit.circuit() if isinstance(circuit, CircuitBuilder) else circuit
        rho = DensityMatrix(qc)
        return float(np.real(np.trace(rho @ rho)))

    # ---- expectation values ---------------------------------------------
    def expectation(self, circuit, pauli: str, qubits: list[int]) -> complex:
        """Expectation of a Pauli string (e.g. 'ZZ', 'XI') over ``qubits``."""
        from qiskit.quantum_info import SparsePauliOp
        qc = circuit.circuit() if isinstance(circuit, CircuitBuilder) else circuit
        op = SparsePauliOp([pauli], coeffs=[1.0])
        sv = Statevector(qc)
        return sv.expectation_value(op)

    # ---- shot simulation ------------------------------------------------
    def run_shots(self, circuit, shots: int | None = None) -> dict[str, int]:
        qc = circuit.circuit() if isinstance(circuit, CircuitBuilder) else circuit
        shots = shots or self.shots
        # Ensure measurement: if no classical bits, append measure_all on a copy.
        if qc.num_clbits == 0:
            qc = qc.copy()
            qc.measure_all()
        sim = AerSimulator()
        if self.noise_model is not None:
            result = sim.run(qc, shots=shots, noise_model=self.noise_model, seed_simulator=self.seed).result()
        else:
            result = sim.run(qc, shots=shots, seed_simulator=self.seed).result()
        counts = result.get_counts()
        # normalize key width to n qubits (keys are bitstrings, possibly already padded)
        n = qc.num_qubits
        norm = {}
        for k, v in counts.items():
            k = str(k)
            if len(k) < n:
                k = k.zfill(n)
            norm[k] = v
        return norm

    def empirical_probs(self, circuit, shots: int | None = None) -> dict[str, float]:
        counts = self.run_shots(circuit, shots)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def conditional_probs(self, circuit, condition_qubits: list[int],
                          target_qubit: int, shots: int | None = None) -> dict[tuple, float]:
        """Empirical P(target=0 | fixed values on condition_qubits), per branch.

        Used e.g. to verify teleportation: for every Bell-measurement branch on
        (q0,q1), the conditional distribution of Bob's q2 must match the input.
        Returns {(vals_on_condition_qubits): P(target_qubit=0 | that branch)}.
        """
        counts = self.run_shots(circuit, shots)
        num = len(next(iter(counts)))          # total qubits
        branches: dict[tuple, list[int]] = {}
        for k, cnt in counts.items():
            # key LSB-first: key[-1]=qubit0
            branch = tuple(int(k[num - 1 - q]) for q in condition_qubits)
            tbit = int(k[num - 1 - target_qubit])
            branches.setdefault(branch, [0, 0])
            branches[branch][1] += cnt
            if tbit == 0:
                branches[branch][0] += cnt
        return {b: (n0 / tot if tot else float("nan")) for b, (n0, tot) in branches.items()}

    # ---- fidelity --------------------------------------------------------
    def fidelity_to_statevector(self, circuit, target: np.ndarray | list) -> float:
        sv = self.statevector(circuit)
        t = np.asarray(target)
        return float(np.abs(np.vdot(t, sv)) ** 2)

    def success_probability(self, circuit, target_bitstring: str) -> float:
        """Probability mass on the computational basis state ``target_bitstring``."""
        probs = self.probability_distribution(circuit)
        return probs.get(target_bitstring, 0.0)
