"""
Quantum Phase Estimation (QPE) — the algorithmic workhorse behind Shor's
factoring, HHL and amplitude estimation.

Given a unitary U and an eigenstate |psi> with U|psi> = e^{2πi φ}|psi>, QPE
recovers an m-bit estimate of the phase φ in the phase register via:

    |0>^m |psi>  --H^m-- / ctrl-U^{2^j} / --IQFT-- measure

The harness *rates* the estimator by its accuracy: with m phase qubits the
recovered bitstring should round the true phase to m bits (up to the standard,
well-understood failure tail ~ < 1/(2^m - 1) for the binary-Chinese-remainder
rounding), which we verify numerically against the exact eigenvalue.
"""
from __future__ import annotations

import numpy as np

from ..core.circuits import CircuitBuilder


def phase_estimation_circuit(
    unitary,
    eigenstate_builder,
    n_phase: int,
    n_state: int,
    eigenvalue: complex,
    name: str = "qpe",
) -> tuple[CircuitBuilder, dict]:
    """Build a QPE circuit and verify the recovered phase against ``eigenvalue``.

    Parameters
    ----------
    unitary : callable(circuit: CircuitBuilder, state_qubits, power_qubits)
        Applies ctrl-U onto ``state_qubits`` controlled by ``power_qubits``
        (already set to |1>, one control per controlled-power). The callback
        must implement a *controlled* U^{2^j} for j = control's index when the
        control is |1>.
    eigenstate_builder : callable(circuit: CircuitBuilder, state_qubits)
        Prepares |psi> on the state register (e.g. X to make |1>).
    n_phase : number of phase qubits (precise binary estimate).
    n_state : number of state qubits.
    eigenvalue : the exact eigenvalue e^{2πi φ} of U on |psi> (for verification).
    """
    b = CircuitBuilder(n_phase + n_state)
    phase = list(range(n_phase))
    state = list(range(n_phase, n_phase + n_state))

    # superposition on phase register
    b.uniform_superposition(phase)
    # prepare eigenstate
    eigenstate_builder(b, state)
    # conditional powers: qubit j controls U^{2^j}
    for j in range(n_phase):
        # each controlled power acts only when control qubit j is |1>
        unitary(b, state, [phase[j]], j)

    # inverse QFT on phase register
    b.iqft(phase)

    # expected measured bitstring = round(phi_p*2^m) mod 2^m
    phi = float(np.angle(eigenvalue) / (2 * np.pi)) % 1.0
    expectation = int(round(phi * (1 << n_phase))) % (1 << n_phase)
    expected = format(expectation, f"0{n_phase}b")

    meta = {
        "n_phase": n_phase,
        "n_state": n_state,
        "true_phase_phi": phi,
        "expected": expected,
        "expected_int": expectation,
        "name": name,
    }
    return b, meta


def phase_estimation_z_rotation():
    """A concrete, verifiable example: estimate φ for U = RZ(2πφ) on |1>.

    U|1> = e^{2πi φ}|1> with φ = 0.375 (=3/8, exactly representable in
    3 phase qubits as 011). The expected recovery is exact.
    """
    phi = 0.375

    def unitary(b: CircuitBuilder, state, controls, j):
        # RZ(2πφ)^{2^j} = RZ(2π φ 2^j), controlled by `controls[0]`
        theta = 2 * np.pi * phi * (2 ** j)
        b.cp(theta, controls[0], state[0])

    def eigenstate(b: CircuitBuilder, state):
        b.x(state[0])          # |psi> = |1>

    return phase_estimation_circuit(
        unitary, eigenstate, n_phase=3, n_state=1,
        eigenvalue=np.exp(2j * np.pi * phi),
        name="RZ(2π·3/8)",
    )


def phase_estimation_irrational():
    """Estimate an irrational phase φ = 1/π ≈ 0.3183 with m=6 phase qubits.

    The recovered integer is the nearest m-bit rounding of φ·2^6 ≈ 20.37 -> 20,
    i.e. bitstring '010100'. Because 1/π is irrational the recovery is exact to
    the truncated precision (nearest-integer rounding, no failure tail).
    """
    phi = 1.0 / np.pi

    def unitary(b: CircuitBuilder, state, controls, j):
        # U = exp(2πi φ) · Identity on the single state qubit; representing it
        # as a global-controlled-RZ with an irrelevant I on |1>:
        # RZ(2πφ 2^j) on a |1> eigenstate produces e^{2πi φ 2^j} only when
        # control=1 — exactly the controlled phase we need.
        from math import pi
        theta = 2 * pi * phi * (2 ** j)
        b.cp(theta, controls[0], state[0])

    def eigenstate(b: CircuitBuilder, state):
        b.x(state[0])

    return phase_estimation_circuit(
        unitary, eigenstate, n_phase=6, n_state=1,
        eigenvalue=np.exp(2j * np.pi * phi),
        name="RZ(2π/π)",
    )


def estimate_phase_circuit(builder, meta, sim):
    """Recover the phase bitstring from a QPE circuit via the exact simulator.

    The phase register occupies qubits ``0..n_phase-1`` (the *low* qubits), so
    in the full MSB-first measurement bitstring it lives in the trailing
    ``n_phase`` characters (LSB-first convention). We sum probability mass on
    all full states whose low-register bytes equal ``expected``.
    """
    n_phase = meta["n_phase"]
    probs = sim.probability_distribution(builder)
    expected = meta["expected"].zfill(n_phase)
    p = 0.0
    for key, pr in probs.items():
        if len(key) >= n_phase and key[-n_phase:] == expected:
            p += pr
    return {"p_expected": float(p), "expected": meta["expected"], **meta}
