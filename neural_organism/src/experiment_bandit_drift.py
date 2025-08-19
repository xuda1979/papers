
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_generators import ContextualBanditDrift
from baselines import LinUCB
from growing_rbf_net import GrowingRBFNet


def run_experiment(
    out_dir: str,
    seed: int = 33,
    steps_per_phase: int = 2000,
    num_phases: int = 6,
    growth_budget: int | None = None,
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
    growth_budget : int, optional
        Maximum number of prototypes the GRBF network may spawn after
        initialization. ``None`` means no growth limit.

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
    grbf = GrowingRBFNet(
        d=d,
        k=A,
        sigma=0.7,
        lr_w=0.25,
        lr_c=0.06,
        min_phi_to_cover=0.25,
        max_prototypes=90,
        growth_budget=growth_budget,
        name="GRBFN+Budget",
    )

    T = steps_per_phase * num_phases
    cumr_lin = np.zeros(T)
    cumr_grb = np.zeros(T)
    regret_lin = np.zeros(T)
    regret_grb = np.zeros(T)
    protos = np.zeros(T, dtype=int)

    for t in range(T):
        x = env.sample(t)
        # LinUCB
        a_lin = linucb.select(x)
        r_lin, _ = env.pull(x, a_lin)
        linucb.update(x, a_lin, r_lin)

        # GRBFN+Plastic policy: argmax p(a|x)
        a_grb = grbf.predict(x)
        r_grb, _ = env.pull(x, a_grb)

        # convert bandit feedback to a supervised-like signal
        phi, _ = grbf._phi(x)
        scores = phi @ grbf.W
        zmax = np.max(scores)
        p = np.exp(scores - zmax)
        p /= np.sum(p)
        if r_grb == 1:
            y = a_grb
        else:
            y = int(np.argsort(p)[-2])
        grbf.update(x, y)

        ps = [env._reward_prob(x, a) for a in range(A)]
        best = np.max(ps)
        regret_lin[t] = (best - ps[a_lin])
        regret_grb[t] = (best - ps[a_grb])

        cumr_lin[t] = r_lin if t == 0 else cumr_lin[t - 1] + r_lin
        cumr_grb[t] = r_grb if t == 0 else cumr_grb[t - 1] + r_grb
        protos[t] = grbf.M

    # Plots
    x = np.arange(T)
    plt.figure(figsize=(8,5))
    plt.plot(x, cumr_lin, label=linucb.name)
    plt.plot(x, cumr_grb, label=grbf.name)
    for dp in [i*steps_per_phase for i in range(1, num_phases)]:
        plt.axvline(dp, linestyle='--', linewidth=1)
    plt.title("Contextual bandit: cumulative reward under drift")
    plt.xlabel("Step"); plt.ylabel("Cumulative reward"); plt.legend()
    cumr_path = os.path.join(out_dir, "bandit_cumreward.png")
    plt.tight_layout(); plt.savefig(cumr_path); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(x, np.cumsum(regret_lin), label=linucb.name)
    plt.plot(x, np.cumsum(regret_grb), label=grbf.name)
    for dp in [i*steps_per_phase for i in range(1, num_phases)]:
        plt.axvline(dp, linestyle='--', linewidth=1)
    plt.title("Contextual bandit: cumulative regret under drift")
    plt.xlabel("Step"); plt.ylabel("Cumulative regret"); plt.legend()
    regret_path = os.path.join(out_dir, "bandit_cumregret.png")
    plt.tight_layout(); plt.savefig(regret_path); plt.close()

    summary = pd.DataFrame({
        "Policy": [linucb.name, grbf.name],
        "Final Cumulative Reward": [float(cumr_lin[-1]), float(cumr_grb[-1])],
        "Final Cumulative Regret": [float(np.cumsum(regret_lin)[-1]), float(np.cumsum(regret_grb)[-1])],
        "Final #Prototypes (GRBFN)": [np.nan, int(protos[-1])],
        "Growth Budget Remaining": [np.nan, grbf.growth_budget],
    })
    summary_path = os.path.join(out_dir, "bandit_summary.csv")
    summary.to_csv(summary_path, index=False)

    return {"summary": summary, "cumr_path": cumr_path, "regret_path": regret_path}

if __name__ == "__main__":
    out = run_experiment(out_dir='../results', seed=33)
    print(out["summary"])
