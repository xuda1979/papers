"""
End-to-end demonstration + smoke test of the QuantumComputing Harness.

Runs a representative battery across every subsystem (algorithms, QEC, noise,
transpile, variants, verification loop, ledger) and prints a structured report.
Use as:  python -m quantum_harness.demo
"""
from __future__ import annotations

import os
import numpy as np

from .core.circuits import CircuitBuilder
from .core.simulator import QuantumSimulator
from .core.noise import NoiseModelBuilder
from .verification.oracle import QCOracle, Verifier
from .algos.core_algorithms import (
    deutsch_jozsa, bernstein_vazirani, grover_search, teleport, qft_circuit,
)
from .algos.variational import QAOABuilder, QAOAProblem, random_qaoa_results
from .qec.codes import SteaneCode, ShorCode, FiveQubitCode, BitFlipCode
from .qec.simulate import protective_circuit_fidelity
from .hardware.topology import random_coupling_map
from .hardware.transpile import transpile_for, verify_unitarily_equivalent
from .memory.ledger import ExperimentLedger
from .orchestrator import AgentHarness


def _s(sim: QuantumSimulator, cb: CircuitBuilder, target: str, tol=1e-8) -> bool:
    probs = sim.probability_distribution(cb)
    return probs.get(target, 0.0) >= 1.0 - tol


def _data_qubit_outcome_prob(sim, cb, n_data: int, target: str) -> float:
    """Probability that the LOW n_data qubits (qiskit LSB-first ordering) read
    out as ``target`` (given MSB-first), ignoring any ancilla/higher qubits.

    In the distribution key `format(i, f"0{num}b")`, the RIGHTMOST char is qubit
    0 (LSB). So the low ``n_data`` qubits occupy the trailing ``n_data`` chars,
    and each char is one bit (0 or 1) of the truncated assignment.
    """
    probs = sim.probability_distribution(cb)
    target = target.zfill(n_data)
    total = 0.0
    for key, p in probs.items():
        if len(key) < n_data:
            continue
        # trailing n_data chars = bit for qubits 0..n_data-1, with char[-1] = qubit 0.
        # Reconstruct the n_data-bit number value of the low register:
        reg = key[-n_data:]
        # reg is MSB-first for the low register: reg[0] is the highest data qubit.
        if reg == target:
            total += p
    return total


def _bell_probs_ok(sim, bell) -> bool:
    d = sim.probability_distribution(bell)
    return abs(d.get("00", 0.0) - 0.5) < 1e-6 and abs(d.get("11", 0.0) - 0.5) < 1e-6


def run_all() -> dict:
    sim = QuantumSimulator(seed=42)
    verif = Verifier(QCOracle(seed=42))
    report = {"modules": {}, "pass": 0, "fail": 0}
    checks = []

    def ok(name, cond, meta=None):
        report["modules"][name] = bool(cond)
        report["pass" if cond else "fail"] += 1
        checks.append((name, cond, meta))

    # --- core circuit/state ---------------------------------------------
    bell = CircuitBuilder(2).bell(0, 1)
    ok("bell_state", _bell_probs_ok(sim, bell), sim.probability_distribution(bell))
    ghz = CircuitBuilder(3).ghz([0, 1, 2])
    probs_ghz = sim.probability_distribution(ghz)
    ok("ghz_state", abs(probs_ghz.get("000", 0.0) - 0.5) < 1e-6
                    and abs(probs_ghz.get("111", 0.0) - 0.5) < 1e-6, probs_ghz)

    # --- algorithms: DJ / BV / Grover / teleport / QFT --------------------
    n = 3
    dj, djmeta = deutsch_jozsa(n, is_balanced=True)
    ok("deutsch_jozsa_balanced",
       _data_qubit_outcome_prob(sim, dj, n, djmeta["expected"]) >= 1.0 - 1e-8, djmeta)
    dj0, dj0meta = deutsch_jozsa(n, is_balanced=False)
    ok("deutsch_jozsa_constant",
       _data_qubit_outcome_prob(sim, dj0, n, dj0meta["expected"]) >= 1.0 - 1e-8)
    secret = [1, 0, 1]
    bv, bvmeta = bernstein_vazirani(n, secret)
    ok("bernstein_vazirani",
       _data_qubit_outcome_prob(sim, bv, n, bvmeta["expected"]) >= 1.0 - 1e-8, bvmeta)

    g, gmeta = grover_search(4, marked=5)
    p_g = sim.success_probability(g, gmeta["expected"])
    ok("grover_search", p_g > 0.35, {"efficiency": f"{p_g:.3f}", "marked": gmeta["marked"]})

    # Teleport verification: for every Bell-measurement branch on (q0=0,q1=1),
    # the conditional state of Bob's q2 must equal RY(0.7)|0>.
    tp, tpmeta = teleport()
    cond = sim.conditional_probs(tp, condition_qubits=[0, 1], target_qubit=2, shots=16384)
    target = float(np.cos(0.35) ** 2)   # P(q2=0) for RY(0.7)|0>
    worst = max(abs(c - target) for c in cond.values())
    ok("teleport_fidelity", worst < 0.05, {"worst_branch_dev": round(worst, 4), "target": round(target, 4)})

    qft, _ = qft_circuit(3)
    ok("qft_unit_norm", abs(np.linalg.norm(sim.statevector(qft)) - 1.0) < 1e-9)

    # --- QEC [[n,k,d]] derivation ---------------------------------------
    steane = SteaneCode()
    ok("steane_params", steane.parameters == (7, 1, 3), steane.parameters)
    five = FiveQubitCode()
    ok("five_qubit_params", five.parameters == (5, 1, 3), five.parameters)
    shor = ShorCode()
    ok("shor_params", shor.parameters == (9, 1, 3), shor.parameters)

    syn = protective_circuit_fidelity(steane, "XIIIIII")
    ok("steane_syndrome_decode", syn["expected_syndrome"] != "", syn)

    # --- noise model + degradation --------------------------------------
    nm = NoiseModelBuilder().with_depolarizing(0.02).build()
    nsim = QuantumSimulator(backend="aer", shots=4096, noise_model=nm, seed=7)
    noisy_counts = nsim.run_shots(bell)
    ok("noise_model_runs", sum(noisy_counts.values()) > 0, dict(noisy_counts))

    # --- transpile + unitary-equivalence ---------------------------------
    cm = random_coupling_map(5, p=0.7, seed=3, bidirectional=True)
    logical = CircuitBuilder(4)
    logical.uniform_superposition(range(4)).cnot(0, 1).cnot(1, 2).cnot(2, 3).cnot(3, 1)
    tp_res = transpile_for(logical, cm.edges, optimization_level=1)
    ok("transpile_depth_positive", tp_res["depth"] > 0, {"depth": tp_res["depth"]})
    ok("transpile_unitary_equiv", verify_unitarily_equivalent(logical, tp_res["circuit"]))

    # --- QAOA exact landscape -------------------------------------------
    problem = QAOAProblem(
        n=4,
        edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
        name="cycle4",
    )
    qaoa = QAOABuilder(problem)
    beta = [0.5]; gamma = [0.5]
    ex = qaoa.expected_cut(beta, gamma)
    opt, _ = problem.brute_force_best()
    ok("qaoa_expected_cut_ok", ex > 0 and ex <= opt + 1e-9, {"expected": ex, "opt": opt})

    ra = random_qaoa_results(n=4, p=1, seed=11, trials=80)
    ok("qaoa_approx_ratio", ra["approx_ratio"] > 0.6, {"approx_ratio": ra["approx_ratio"]})

    # --- verification oracle battery -------------------------------------
    r = verif.shots_match_statevector(bell)
    ok("oracle_shots_vs_state", r.verified)
    u = verif.unitary_check(np.eye(4).astype(complex))
    ok("oracle_unitary_check", u.verified)

    return report


def demonstrate_agent_loop(ledger_path: str) -> dict:
    """Show the AgentHarness plan->execute->verify->log->reflect loop."""
    harness = AgentHarness(ledger=ExperimentLedger(ledger_path), src="demo-loop", seed=1)

    from .core.circuits import CircuitBuilder

    # Task 1: a genuinely correct claim (Grover on 3 qubits, marked=3, so the
    # measured data register reads "011"). With the optimal-rounded iteration
    # count (2 iterations over N=8) Grover reaches P(success) ~ 0.945, so we
    # assert an honest *lower bound* of 0.9 that the oracle can confirm.
    g_circ, g_meta = grover_search(3, marked=3)
    expected_bit = g_meta["expected"]           # "011"

    def task_truth():
        p = harness.oracle.sim.success_probability(g_circ, expected_bit)
        return {"success_prob": float(p), "marked": g_meta["marked"]}

    check_truth = harness.verifier.outcome_prob_at_least(g_circ, expected_bit, 0.9)
    outcome1 = harness.run(
        f"grover n=3 finds marked={g_meta['marked']}",
        task_truth,
        checks=[check_truth],
    )

    # Task 2: an intentionally wrong claim should FAIL loudly (Bell P(11) NOT >= 0.999)
    def task_false():
        return {"claimed": "Bell state has P(00)=1.0"}

    bogus = harness.verifier.outcome_prob_at_least(CircuitBuilder(2).bell(0, 1), "11", 0.999)
    outcome2 = harness.run(
        "wrong claim: bell P(11)>=0.999",
        task_false,
        checks=[bogus],
    )

    return {
        "task1_verified": outcome1.verified,
        "task2_verified": outcome2.verified,
        "task2_reflection": outcome2.reflection,
        "ledger_count": harness.ledger.count(),
    }


if __name__ == "__main__":
    # silence noisy backends
    import logging
    logging.getLogger("qiskit").setLevel(logging.ERROR)

    print("=" * 70)
    print("QuantumComputing Harness — smoke test & demonstration")
    print("=" * 70)
    rep = run_all()
    print(f"\nModules tested: {len(rep['modules'])}")
    print(f"  PASS: {rep['pass']}   FAIL: {rep['fail']}")
    for name, status in rep["modules"].items():
        print(f"  [{'OK ' if status else 'FAIL'}] {name}")
    if rep["fail"]:
        print("\nSome checks FAILED — see above.")
    else:
        print("\nAll subsystem tests PASSED.")

    print("\n" + "=" * 70)
    print("AgentHarness verification-first loop demonstration")
    print("=" * 70)
    ledger_file = os.path.join(os.getcwd(), "quantum_ledger_demo.json")
    loop = demonstrate_agent_loop(ledger_file)
    print(f"  Correct claim verified : {loop['task1_verified']}")
    print(f"  Wrong claim flagged    : verified={loop['task2_verified']}")
    print(f"  Failure reflection     : {loop['task2_reflection']}")
    print(f"  Ledger entries written : {loop['ledger_count']}")
    print(f"\nLedger file: {ledger_file}")
