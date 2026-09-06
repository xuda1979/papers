"""
CircuitBuilder: ergonomic, validated quantum-circuit construction on top of qiskit.

Design goals for a *reliable* agent harness:
  * Fail loudly on invalid operations (unknown gate, wrong qubit index).
  * Offer high-level composite templates (arbitrary rotation, uniform superposition,
    phase oracle, reflection / diffusion, Bell, GHZ, teleportation) so the agent
    never hand-assembles error-prone subcircuits.
  * Keep a pure-Qiskit ``QuantumCircuit`` at the end for transpilation/simulation.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT


class CircuitBuilder:
    """A validated wrapper that accumulates gates onto a qiskit circuit."""

    def __init__(self, n_qubits: int, n_clbits: int = 0, name: str = "qc"):
        if n_qubits < 0:
            raise ValueError("n_qubits must be non-negative")
        self.n = n_qubits
        self.qc = QuantumCircuit(n_qubits, n_clbits, name=name)

    # --- primitive single/two-qubit gates -----------------------------------
    def _chk(self, q: Iterable[int]):
        for qi in q:
            if not (0 <= qi < self.n):
                raise IndexError(f"qubit {qi} out of range [0,{self.n})")

    def h(self, q: int):
        self._chk([q]); self.qc.h(q); return self

    def x(self, q: int):
        self._chk([q]); self.qc.x(q); return self

    def y(self, q: int):
        self._chk([q]); self.qc.y(q); return self

    def z(self, q: int):
        self._chk([q]); self.qc.z(q); return self

    def s(self, q: int):
        self._chk([q]); self.qc.s(q); return self

    def t(self, q: int):
        self._chk([q]); self.qc.t(q); return self

    def rx(self, theta: float, q: int):
        self._chk([q]); self.qc.rx(theta, q); return self

    def ry(self, theta: float, q: int):
        self._chk([q]); self.qc.ry(theta, q); return self

    def rz(self, theta: float, q: int):
        self._chk([q]); self.qc.rz(theta, q); return self

    def cnot(self, c: int, t: int):
        self._chk([c, t]); self.qc.cx(c, t); return self

    def cz(self, c: int, t: int):
        self._chk([c, t]); self.qc.cz(c, t); return self

    def swap(self, a: int, b: int):
        self._chk([a, b]); self.qc.swap(a, b); return self

    def toffoli(self, a: int, b: int, c: int):
        self._chk([a, b, c]); self.qc.ccx(a, b, c); return self

    def p(self, lam: float, q: int):
        """Phase gate P(λ) = diag(1, e^{iλ})."""
        self._chk([q]); self.qc.p(lam, q); return self

    def cp(self, lam: float, c: int, t: int):
        """Controlled phase gate (control ``c``, target ``t``)."""
        self._chk([c, t]); self.qc.cp(lam, c, t); return self

    def mcp(self, lam: float, controls: Sequence[int], target: int):
        """Multi-controlled phase gate; -1 (λ=π) flips phase only on all-ones."""
        controls = list(controls)
        self._chk([*controls, target])
        self.qc.mcp(lam, controls, target); return self

    def sx(self, q: int):
        self._chk([q]); self.qc.sx(q); return self

    def sdg(self, q: int):
        self._chk([q]); self.qc.sdg(q); return self

    def tdg(self, q: int):
        self._chk([q]); self.qc.tdg(q); return self

    def ccz(self, a: int, b: int, c: int):
        """Doubly-controlled Z (CCZ) gate."""
        self._chk([a, b, c]); self.qc.ccz(a, b, c); return self

    def ch(self, c: int, t: int):
        """Controlled-Hadamard."""
        self._chk([c, t]); self.qc.ch(c, t); return self

    def cs(self, c: int, t: int):
        """Controlled-S (controlled-phase π/2)."""
        self._chk([c, t]); self.qc.cs(c, t); return self

    # --- composite templates -------------------------------------------------
    def uniform_superposition(self, qubits: Sequence[int]):
        """Apply H to each of ``qubits`` (typically all)."""
        self._chk(qubits)
        for q in qubits:
            self.qc.h(q)
        return self

    def bell(self, a: int = 0, b: int = 1):
        """|00> -> |00>+|11> Bell pair."""
        self._chk([a, b])
        self.qc.h(a); self.qc.cx(a, b)
        return self

    def ghz(self, qubits: Sequence[int]):
        """GHZ state on the given qubits (first is the root)."""
        qubits = list(qubits)
        self._chk(qubits)
        self.qc.h(qubits[0])
        for r in qubits[1:]:
            self.qc.cx(qubits[0], r)
        return self

    def qft(self, qubits: Optional[Sequence[int]] = None):
        """Quantum Fourier transform on ``qubits`` (or all qubits)."""
        if qubits is None:
            qubits = list(range(self.n))
        qubits = list(qubits)
        self._chk(qubits)
        comp = QFT(num_qubits=len(qubits), do_swaps=True)
        # need to map composite onto arbitrary wires
        self.qc.append(comp.to_gate(), qubits)
        return self

    def iqft(self, qubits: Optional[Sequence[int]] = None):
        if qubits is None:
            qubits = list(range(self.n))
        qubits = list(qubits)
        self._chk(qubits)
        comp = QFT(num_qubits=len(qubits), do_swaps=True, inverse=True)
        self.qc.append(comp.to_gate(), qubits)
        return self

    def phase_oracle(
        self,
        marked_states: Sequence[int],
        n_work: int = 1,
        work: Optional[int] = None,
    ):
        """Oracle that flips the phase of each marked computational state.

        Uses a phase oracle via X/controlled-Z/X trick (no work qubit needed for
        the phase flip itself on the target register). ``n_work``/``work`` are
        accepted for signature compatibility and reserved for future multi-register
        use.
        """
        marked = [m % (1 << self.n) for m in marked_states]
        self._chk([])
        for m in marked:
            # format(m, "0nb") is MSB-first: char k is the bit of weight 2^(n-1-k),
            # which lives on qiskit qubit (n-1-k) because qubit 0 is the LSB.
            bstr = format(m, f"0{self.n}b")
            # X on qubits that are 0, controlled-Z on all, then undo X.
            # A phase oracle |x> -> -|x> iff x == marked: flip |0>s to |1>,
            # apply multi-controlled-Z on the all-ones state, then unflip.
            flipped = [self.n - 1 - i for i, ch in enumerate(bstr) if ch == "0"]
            for q in flipped:
                self.qc.x(q)
            # mcp(pi, ctrl, target) applies a phase of -1 exactly when ALL n qubits
            # (the n-1 controls AND the target) are |1>. After flipping the 0-bits of
            # |marked> to 1 this means it flips the phase of exactly |marked>.
            # NOTE: we must NOT wrap this in Hadamards here --- H·P(pi)·H = X would
            # turn the controlled phase INTO a controlled-NOT (a bit flip), which is
            # the wrong operation for a *phase* oracle.
            ctrl = list(range(self.n - 1))
            self.qc.mcp(np_equiv_pi(), ctrl, self.n - 1)  # multi-controlled Z via pi phase
            for q in flipped:
                self.qc.x(q)
        return self

    def diffusion(self, qubits: Optional[Sequence[int]] = None):
        """Grover diffusion operator on ``qubits`` (or all).

        Implements the reflection-about-the-mean
            D = H^{⊗n} X^{⊗n} (I - 2|1...1><1...1|) X^{⊗n} H^{⊗n}.
        The middle factor is a -1 phase flip of the all-ones state, which is
        applied directly with mcp(pi, ctrl, target) --- *not* wrapped in
        Hadamards (H·P(pi)·H = X would produce a controlled-NOT bit-flip and
        break the reflection).
        """
        if qubits is None:
            qubits = list(range(self.n))
        qubits = list(qubits)
        self._chk(qubits)
        self.qc.h(qubits)
        self.qc.x(qubits)
        # -1 phase on the all-ones state (controls == target carried directly)
        self.qc.mcp(np_equiv_pi(), qubits[:-1], qubits[-1])
        self.qc.x(qubits)
        self.qc.h(qubits)
        return self

    def arbitrary_pauli_rotation(self, pauli: str, theta: float, qubits: Sequence[int]):
        """exp(-i theta/2 P) for a Pauli string P over ``qubits`` (len must match)."""
        pauli = pauli.upper()
        qubits = list(qubits)
        self._chk(qubits)
        if len(pauli) != len(qubits):
            raise ValueError(f"pauli length {len(pauli)} != qubits {len(qubits)}")
        # order generously from qiskit's PauliEvolutionGate
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        op = SparsePauliOp([pauli], coeffs=[1.0])
        gate = PauliEvolutionGate(op, time=theta / 2.0)
        self.qc.append(gate, qubits)
        return self

    def measure_all(self):
        self.qc.measure_all()
        return self

    def measure(self, qubits: Sequence[int], creg_start: int = 0):
        qubits = list(qubits)
        self._chk(qubits)
        for i, q in enumerate(qubits):
            self.qc.measure(q, creg_start + i)
        return self

    def circuit(self) -> QuantumCircuit:
        return self.qc

    def __str__(self):
        return self.qc.draw(output="text")

    def depth(self) -> int:
        return self.qc.depth()


import math as _m

def np_equiv_pi():
    """Return pi for use in phase angles (avoids importing numpy in module top)."""
    return _m.pi
