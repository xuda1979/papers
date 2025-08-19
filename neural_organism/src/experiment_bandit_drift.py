import os
import numpy as np
import pandas as pd
from .data_generators import ContextualBanditDrift
from .baselines import LinUCB
from .growing_rbf_net_plastic import GrowingRBFNetPlastic, RuntimeGate
from .experiment_utils import plot_bandit_results, save_bandit_summary


def run_experiment(
    out_dir: str,
    seed: int = 33,
    steps_per_phase: int = 2000,
    num_phases: int = 6,
    growth_gate: RuntimeGate | None = None,
) -> dict:
    """Run the contextual bandit experiment with non-stationary drift.

    Parameters
    ----------
    out_dir : str
        Directory where figures and the summary CSV will be written.
    seed : int, optional
        Random seed for reproducibility.
    steps_per_phase : int, optional
        Number of interaction steps before a drift occurs.
    num_phases : int, optional
        Number of drift phases in the experiment.
    growth_gate : RuntimeGate, optional
        A runtime-enforced gate to control structural growth. If None,
        growth is unlimited.

    Returns
    -------
    dict
        Dictionary containing the summary dataframe and paths to generated plots.
    """
    os.makedirs(out_dir, exist_ok=True)
    d, A = 2, 3
    env = ContextualBanditDrift(
        d=d,
        A=A,
        n_components=2,
        sigma=0.5,
        steps_per_phase=steps_per_phase,
        num_phases=num_phases,
        seed=seed,
    )

    linucb = LinUCB(d=d, A=A, alpha=0.8, lam=1.0)
    grbf = GrowingRBFNetPlastic(
        d=d,
        k=A,
        sigma=0.7,
        lr_w=0.25,
        lr_c=0.06,
        min_phi_to_cover=0.25,
        max_prototypes=90,
        growth_gate=growth_gate,
        name="GRBFN+Plastic",
    )

    T = steps_per_phase * num_phases
    results = {
        linucb.name: {"cum_rewards": np.zeros(T), "cum_regrets": np.zeros(T)},
        grbf.name: {"cum_rewards": np.zeros(T), "cum_regrets": np.zeros(T)},
    }
    protos = np.zeros(T, dtype=int)

    for t in range(T):
        x = env.sample(t)

        # --- LinUCB ---
        a_lin = linucb.select(x)
        r_lin, _ = env.pull(x, a_lin)
        linucb.update(x, a_lin, r_lin)

        # --- GRBFN+Plastic ---
        # The policy is to select the arm with the highest predicted reward.
        a_grb = grbf.predict(x)
        r_grb, _ = env.pull(x, a_grb)

        # Convert bandit feedback to a supervised-like signal for update.
        # This is a heuristic: if reward is positive, treat the chosen action
        # as the correct label. If reward is negative, we need to infer a
        # "better" action. Here, we use the second-best action as the target.
        # This encourages exploration away from bad choices.
        if r_grb > 0:
            y = a_grb
        else:
            # A simple heuristic to find a better action to move towards.
            phi, _ = grbf._phi(x)
            scores = (phi * grbf._gate(x, phi)[1]) @ (grbf.W + grbf.H)
            p = np.exp(scores - np.max(scores))
            p /= np.sum(p)
            y = int(np.argsort(p)[-2])
        grbf.update(x, y)

        # --- Logging ---
        ps = [env._reward_prob(x, a) for a in range(A)]
        best_r = np.max(ps)

        results[linucb.name]["cum_rewards"][t] = r_lin if t == 0 else results[linucb.name]["cum_rewards"][t-1] + r_lin
        results[linucb.name]["cum_regrets"][t] = (best_r - ps[a_lin]) if t == 0 else results[linucb.name]["cum_regrets"][t-1] + (best_r - ps[a_lin])

        results[grbf.name]["cum_rewards"][t] = r_grb if t == 0 else results[grbf.name]["cum_rewards"][t-1] + r_grb
        results[grbf.name]["cum_regrets"][t] = (best_r - ps[a_grb]) if t == 0 else results[grbf.name]["cum_regrets"][t-1] + (best_r - ps[a_grb])
        protos[t] = grbf.M

    # Plotting
    cumr_path, regret_path = plot_bandit_results(
        results, out_dir, "bandit"
    )

    # Summary
    summary_data = [
        {
            "Policy": linucb.name,
            "Final Cumulative Reward": results[linucb.name]["cum_rewards"][-1],
            "Final Cumulative Regret": results[linucb.name]["cum_regrets"][-1],
            "Final #Prototypes": np.nan,
            "Growth Budget Remaining": np.nan,
        },
        {
            "Policy": grbf.name,
            "Final Cumulative Reward": results[grbf.name]["cum_rewards"][-1],
            "Final Cumulative Regret": results[grbf.name]["cum_regrets"][-1],
            "Final #Prototypes": protos[-1],
            "Growth Budget Remaining": grbf.growth_gate._budget,
        },
    ]
    summary_path = save_bandit_summary(summary_data, out_dir, "bandit_summary.csv")

    return {"summary_path": summary_path, "cumr_path": cumr_path, "regret_path": regret_path}


if __name__ == "__main__":
    # This run is cited in the paper's case study.
    # We instantiate a gate with a fixed budget of 22 new units,
    # for a total of k=3 initial + 22 new = 25 max prototypes.
    gate = RuntimeGate(22)
    out = run_experiment(out_dir="neural_organism/results", seed=33, growth_gate=gate)
    print(f"Results saved to {out['summary_path']}")
