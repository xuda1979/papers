
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from .data_generators import ContextualBanditDrift
from .baselines import LinUCB, MLPOnline
from .growing_rbf_net_plastic import GrowingRBFNetPlastic, RuntimeGate

def run_experiment(out_dir, seed=37, growth_gate: RuntimeGate | None = None):
    os.makedirs(out_dir, exist_ok=True)
    d, A = 2, 3
    steps_per_phase = 2000; num_phases = 6
    env = ContextualBanditDrift(d=d, A=A, n_components=2, sigma=0.5,
                                steps_per_phase=steps_per_phase, num_phases=num_phases, seed=seed)

    linucb = LinUCB(d=d, A=A, alpha=0.8, lam=1.0)
    grbfnp = GrowingRBFNetPlastic(d=d, k=A, sigma=0.7, lr_w=0.25, lr_c=0.06, min_phi_to_cover=0.25,
                                  growth_gate=growth_gate, hebb_eta=0.6, hebb_decay=0.94, gate_beta=1.2, gate_top=16, key_lr=0.03,
                                  r_merge=0.4, merge_every=600, max_prototypes=90, name="GRBFN+Plastic")
    mlp = MLPOnline(d=d, k=A, h=32, lr=0.01, wd=1e-5, name="MLP(32)")

    T = steps_per_phase * num_phases
    cumr_lin = np.zeros(T); cumr_grb = np.zeros(T); cumr_mlp = np.zeros(T)
    regret_lin = np.zeros(T); regret_grb = np.zeros(T); regret_mlp = np.zeros(T)
    protos = np.zeros(T, dtype=int)

    for t in range(T):
        x = env.sample(t)
        # LinUCB
        a_lin = linucb.select(x)
        r_lin, _ = env.pull(x, a_lin)
        linucb.update(x, a_lin, r_lin)

        # GRBFN+Plastic
        a_grb, p_grb = grbfnp.predict(x, return_proba=True)
        r_grb, _ = env.pull(x, a_grb)
        y_grb = a_grb if r_grb == 1 else int(np.argsort(p_grb)[-2])
        grbfnp.update(x, y_grb)

        # MLP policy
        # predict
        _, p_mlp = mlp._forward(x)
        a_mlp = int(np.argmax(p_mlp))
        r_mlp, _ = env.pull(x, a_mlp)
        y_mlp = a_mlp if r_mlp == 1 else int(np.argsort(p_mlp)[-2])
        mlp.update(x, y_mlp)

        # oracle and logging
        ps = [env._reward_prob(x, a) for a in range(A)]
        best = np.max(ps)

        regret_lin[t] = (best - ps[a_lin])
        regret_grb[t] = (best - ps[a_grb])
        regret_mlp[t] = (best - ps[a_mlp])

        cumr_lin[t] = r_lin if t==0 else cumr_lin[t-1] + r_lin
        cumr_grb[t] = r_grb if t==0 else cumr_grb[t-1] + r_grb
        cumr_mlp[t] = r_mlp if t==0 else cumr_mlp[t-1] + r_mlp
        protos[t] = grbfnp.M

    x = np.arange(T)
    plt.figure(figsize=(8,5))
    plt.plot(x, cumr_lin, label=linucb.name)
    plt.plot(x, cumr_grb, label="GRBFN+Plastic")
    plt.plot(x, cumr_mlp, label="MLP(32)")
    for dp in [i*steps_per_phase for i in range(1, num_phases)]:
        plt.axvline(dp, linestyle='--', linewidth=1)
    plt.title("Contextual bandit: cumulative reward under drift (incl. MLP)")
    plt.xlabel("Step"); plt.ylabel("Cumulative reward"); plt.legend()
    cumr_path = os.path.join(out_dir, "bandit_cumreward_plusmlp.png")
    plt.tight_layout(); plt.savefig(cumr_path); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(x, np.cumsum(regret_lin), label=linucb.name)
    plt.plot(x, np.cumsum(regret_grb), label="GRBFN+Plastic")
    plt.plot(x, np.cumsum(regret_mlp), label="MLP(32)")
    for dp in [i*steps_per_phase for i in range(1, num_phases)]:
        plt.axvline(dp, linestyle='--', linewidth=1)
    plt.title("Contextual bandit: cumulative regret under drift (incl. MLP)")
    plt.xlabel("Step"); plt.ylabel("Cumulative regret"); plt.legend()
    regret_path = os.path.join(out_dir, "bandit_cumregret_plusmlp.png")
    plt.tight_layout(); plt.savefig(regret_path); plt.close()

    summary = pd.DataFrame({
        "Policy": [linucb.name, "GRBFN+Plastic", "MLP(32)"],
        "Final Cumulative Reward": [float(cumr_lin[-1]), float(cumr_grb[-1]), float(cumr_mlp[-1])],
        "Final Cumulative Regret": [float(np.cumsum(regret_lin)[-1]), float(np.cumsum(regret_grb)[-1]), float(np.cumsum(regret_mlp)[-1])],
        "Final #Prototypes (GRBFN+)": [np.nan, int(protos[-1]), np.nan]
    })
    summary_path = os.path.join(out_dir, "bandit_summary_plusmlp.csv")
    summary.to_csv(summary_path, index=False)

    return {"summary": summary, "cumr_path": cumr_path, "regret_path": regret_path}

if __name__ == "__main__":
    # This run is also cited in the paper's case study and README.
    gate = RuntimeGate(22) # 3 initial + 22 new = 25 total
    out = run_experiment(out_dir="neural_organism/results", seed=37, growth_gate=gate)
    print(out["summary"])
