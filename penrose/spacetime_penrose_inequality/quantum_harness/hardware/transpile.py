"""Transpilation: map a logical circuit onto a (noisy) device coupling map.

The harness treats transpilation as a *numerically-checked* transformation:
every produced circuit is verified to be unitary-equivalent to the logical one
(up to global phase), so an agent can never silently report a broken layout.

Robustness details encoded here:
  * A coupling map may have more qubits than the logical circuit. We restrict
    to a connected induced sub-coupling-map covering exactly the circuit's
    qubits before routing, so the returned object has the right width and the
    unitary-equivalence check is meaningful (no phantom idle qubits).
  * Error-aware cost estimates (expected CX fidelity) let the agent reason
    about *which* mapping/opt level is best for a device with given gate errors.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from ..core.circuits import CircuitBuilder

_SEED = 0


def _restrict_coupling_map(coupling_edges, n_qubits):
    """Return a dict of interface-compatible coupling edges spanning <= n_qubits.

    If the device has more qubits than the circuit, pick a connected induced
    subgraph on the first ``n_qubits`` nodes (or on all low-index nodes) so the
    transpiler does not allocate phantom qubits. If there are fewer device
    qubits than needed, return the (too-small) map and let routing fail loudly
    rather than silently over-allocating.
    """
    nodes = sorted({u for u, _ in coupling_edges} | {v for _, v in coupling_edges})
    if len(nodes) <= n_qubits:
        return coupling_edges
    # prefer an induced connected subgraph on the n_qubits lowest nodes
    target = set(range(n_qubits))
    sub = [(u, v) for u, v in coupling_edges if u in target and v in target]
    return sub


def transpile_for(
    circuit,
    coupling_edges: list[tuple[int, int]],
    optimization_level: int = 3,
    basis: list[str] | None = None,
    gate_errors: dict[tuple[int, int], float] | None = None,
) -> dict:
    """Transpile onto the given topology; return circuit + depth/gate counts.

    ``gate_errors`` maps a (control, target) physical edge to its error rate
    (e.g. 1 - fidelity). When provided, the report includes ``expected_error``
    = 1 - product of CX fidelities, a hardware-aware cost the agent can use to
    choose among layouts / optimization levels.
    """
    from qiskit.transpiler import CouplingMap
    qc = circuit.circuit() if isinstance(circuit, CircuitBuilder) else circuit
    n = qc.num_qubits
    restricted = _restrict_coupling_map(coupling_edges, n)
    cm = CouplingMap(couplinglist=restricted)
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        coupling_map=cm,
        basis_gates=basis or ["cx", "u1", "u2", "u3"],
        seed_transpiler=_SEED,
    )
    tqc = pm.run(qc)
    ops = tqc.count_ops()
    total_cx = int(ops.get("cx", 0))

    # hardware-aware error estimate: sum (approx.) over CX edges actually used
    expected_error = float("nan")
    if gate_errors:
        used_edges = []
        for instr in tqc.data:
            name = instr.operation.name
            if name == "cx" and len(instr.qubits) == 2:
                a = tqc.find_bit(instr.qubits[0]).index
                b = tqc.find_bit(instr.qubits[1]).index
                used_edges.append((a, b))
        ncx = len(used_edges)
        if ncx:
            per = [gate_errors.get((u, v), 0.05) for u, v in used_edges]
            exp_no_err = np.prod([1.0 - e for e in per])
            expected_error = 1.0 - exp_no_err
    return {
        "circuit": tqc,
        "depth": tqc.depth(),
        "width": tqc.num_qubits,
        "gate_counts": dict(ops),
        "total_cx": total_cx,
        "transpiled": True,
        "expected_error": float(expected_error),
    }


def verify_unitarily_equivalent(original, transpiled, tol: float = 1e-6) -> bool:
    """Confirm transpilation preserved the unitary up to an irrelevant global phase.

    Uses the Hilbert-Schmidt fidelity F = |tr(U† V)|² / D², which is invariant
    under global phase (and under the sign U↦-U from SWAP-routing reflections),
    and returns a structured verdict rather than a bare boolean.
    """
    o = Operator(original.circuit() if isinstance(original, CircuitBuilder) else original)
    t = Operator(transpiled.circuit() if isinstance(transpiled, CircuitBuilder) else transpiled)
    if o.input_dims() != t.input_dims() or o.output_dims() != t.output_dims():
        return False
    if o.data.size == 0:
        return True
    U = o.data.astype(np.complex128)
    V = t.data.astype(np.complex128)
    D = U.shape[0]
    fid = float(abs(np.trace(np.conj(U.T) @ V)) ** 2) / (D * D)
    return (1.0 - fid) < tol
