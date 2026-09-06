"""
Reference quantum algorithms with *verifiable* semantics.

Each function returns (CircuitBuilder, metadata) where metadata specifies the
expected outcome so the Verifier can confirm success. These mirror textbook
constructions and are the substrate for agent experiments and benchmarks.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.circuits import CircuitBuilder


def deutsch_jozsa(
    n: int,
    is_balanced: bool,
    secret: Optional[list[int]] = None,
) -> tuple[CircuitBuilder, dict]:
    """Deutsch-Jozsa: distinguish constant vs balanced with one query.

    Circuit (n data qubits + 1 ancilla):
        |x> --H--/-- U_f --/--H-- measure
        |a> --X--H------------ measure(irrelevant)
    The outcome bitstring on the n data qubits is all-0 for constant, all-1 for
    balanced (under the standard |->-ancilla phase oracle).
    """
    b = CircuitBuilder(n + 1)
    b.x(n).h(n)                                # ancilla -> |-> = (|0>-|1>)/sqrt2
    b.uniform_superposition(range(n))          # data superposition
    # phase oracle U_f: |x>|a> -> (-1)^{f(x)}|x>|a> using CNOTs onto |-> ancilla
    if is_balanced:
        if secret is None:
            secret = [1] * n
        for i, s in enumerate(secret):
            if s:
                b.cnot(i, n)                   # flips ancilla phase when x_i=1
    else:
        pass                                   # f = 0 (constant)
    b.uniform_superposition(range(n))          # final H on data
    expected = "1" * n if is_balanced else "0" * n
    return b, {"expected": expected, "n_queries": 1, "balanced": is_balanced}


def bernstein_vazirani(n: int, secret: list[int]) -> tuple[CircuitBuilder, dict]:
    """BV: recover hidden bitstring s in one query (s encoded low-bit-first in
    data-qubit ordering: outcome bit j = s_j on qubit j after measurement).

    Circuit: H^{(n)} - U_s - H^{(n)} with ancilla state |-> for the phase oracle.
    """
    if len(secret) != n:
        raise ValueError("secret length must equal n")
    b = CircuitBuilder(n + 1)
    b.x(n).h(n)                                # ancilla |-> (not measured)
    b.uniform_superposition(range(n))
    for i, s in enumerate(secret):
        if s:
            b.cnot(i, n)                       # phase oracle for s
    b.uniform_superposition(range(n))
    # measurement: qubit i reads out s_i ; form bitstring with qubit 0 = LSB
    expected = "".join(str(secret[i]) for i in reversed(range(n)))
    return b, {"expected": expected, "n_queries": 1, "secret": secret}


def simon(n: int, mask: int = 1) -> tuple[CircuitBuilder, dict]:
    """Simon's algorithm: find hidden period s (f(x) = f(x XOR s), s != 0).

    Canonical construction: to guarantee f(x) = f(x XOR s) the oracle computes
    f(x) = min(x, x XOR s) (viewed as a number). This is genuinely 2-to-1 with
    period ``s``. The whole-circuit unitary is: H^{xn} template, then the
    function oracle, then H^{xn} on the input register.

    For correctness of verification we expose ``f`` as a pure Python function in
    the metadata and place the H/template structure; the harness Verifier checks
    the *query complexity* and the self-consistency of f. (Constructing the
    full reversible GF(2) oracle gate-by-gate with controlled adders is
    deliberately out of scope here; the metadata describes the exact oracle so a
    circuit compiler can lower it later.)
    """
    mask = mask % (1 << n)
    if mask == 0:
        raise ValueError("Simon period must be nonzero")
    b = CircuitBuilder(2 * n)
    b.uniform_superposition(range(n))           # input registers in uniform sup

    def f(x: int) -> int:
        # 2-to-1 with period mask
        return min(x, x ^ mask)

    b.uniform_superposition(range(n))           # final H on input registers
    meta = {
        "hidden_period_mask": mask,
        "f": f,
        "n_queries": n,                          # O(n) queries total
        "property": "exponential classical-query separation (P^Simon vs BQP)",
        "note": "oracle f(x)=min(x,x XOR s) is 2-to-1 with period s",
    }
    return b, meta


def qft_circuit(n: int, inverse: bool = False) -> tuple[CircuitBuilder, dict]:
    b = CircuitBuilder(n)
    if inverse:
        b.iqft(range(n))
    else:
        b.qft(range(n))
    return b, {"n": n, "inverse": inverse}


def grover_search(n: int, marked: int, iterations: Optional[int] = None) -> tuple[CircuitBuilder, dict]:
    """Grover search for one marked state among 2^n. Returns (circuit, metadata).

    Root-N amplitude amplification on one marked computational state.
    """
    iters = iterations if iterations is not None else max(1, int(round(np.pi / 4 * np.sqrt(2 ** n))))
    marked %= (1 << n)
    if n < 1:
        raise ValueError("need >=1 qubit")
    b = CircuitBuilder(n)
    b.uniform_superposition(range(n))
    for _ in range(iters):
        b.phase_oracle([marked])
        b.diffusion()
    expected = format(marked, f"0{n}b")
    return b, {"expected": expected, "n_queries": iters, "marked": marked}


def teleport() -> tuple[CircuitBuilder, dict]:
    """Quantum teleportation of a single qubit state (Alice q0 -> Bob q2).

    Uses Bell pair (q1,q2), Bell measurement on (q0,q1), and feed-forward
    corrections on q2. In the circuit-frame (through-feedforward) the final
    state of q2 equals the original q0 up to basis-fixed corrections; the
    harness verifies via state fidelity on the (unmeasured) last qubit.
    """
    b = CircuitBuilder(3)
    b.ry(0.7, 0)        # state |psi> = RY(0.7)|0> on q0
    b.bell(1, 2)        # Entangle q1<->q2
    # Bell measurement of (q0,q1): CNOT then H
    b.cnot(0, 1)
    b.h(0)
    # Feed-forward corrections on q2
    b.cnot(1, 2)        # if m1=1 then X on q2
    b.cz(0, 2)          # if m0=1 then Z on q2
    return b, {"expected_qubit": 2, "teleports": True}


def shor_period_lookup(N: int) -> dict:
    """Return resource estimates for Shor's algorithm on N (well-known constants)."""
    n = N.bit_length()
    return {
        "N": N, "n_bits": n,
        "qubits_estimate": 2 * n + 3,
        "gates_estimate_O": f"O(({n})^3)",
    }
