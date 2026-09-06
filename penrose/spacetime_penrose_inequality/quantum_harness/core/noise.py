"""
NoiseModelBuilder: construct qiskit noise models for realistic device studies.

Enables the harness to quantify how algorithms degrade under *realistic* gate
and readout errors --- essential for honest NISQ claims.
"""
from __future__ import annotations

import numpy as np
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, pauli_error, amplitude_damping_error,
    phase_damping_error, ReadoutError,
)


class NoiseModelBuilder:
    def __init__(self):
        self.model = NoiseModel()

    def with_depolarizing(self, p: float, gates: list[str] | None = None):
        gates = gates or ["u1", "u2", "u3", "x", "y", "z", "h", "s", "t", "rx", "ry", "rz"]
        for g in gates:
            self.model.add_all_qubit_quantum_error(depolarizing_error(p, 1), g)
        return self

    def with_two_qubit_depolarizing(self, p: float, gates: list[str] | None = None):
        gates = gates or ["cx", "cz", "swap"]
        for g in gates:
            self.model.add_all_qubit_quantum_error(depolarizing_error(p, 2), g)
        return self

    def with_amplitude_damping(self, gamma: float, gates: list[str] | None = None):
        gates = gates or ["x", "y", "z", "h", "rx", "ry", "rz"]
        err = amplitude_damping_error(gamma)
        for g in gates:
            self.model.add_all_qubit_quantum_error(err, g)
        return self

    def with_readout_error(self, p0_given_1: float, p1_given_0: float):
        # readout matrix rows: P(measured | prepared), indexed [prepared][measured]
        self.model.add_all_qubit_readout_error(
            ReadoutError([[1 - p1_given_0, p1_given_0], [p0_given_1, 1 - p0_given_1]])
        )
        return self

    def build(self) -> NoiseModel:
        return self.model
