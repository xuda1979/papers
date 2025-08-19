
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, time
from data_generators import MixtureDriftGaussians
from baselines import LogisticSGD, MLPOnline
from growing_rbf_net import GrowingRBFNet
from growing_rbf_net_plastic import GrowingRBFNetPlastic

def moving_average(arr, w=200):
    if w <= 1: return arr.copy()
    return np.convolve(arr, np.ones(w)/w, mode='same')

def run_experiment(out_dir, seed=21):
    os.makedirs(out_dir, exist_ok=True)
    d, k = 2, 3
    steps_per_phase = 2000
    num_phases = 6
    data = MixtureDriftGaussians(d=d, k=k, n_components=2, sigma=0.4,
                                 steps_per_phase=steps_per_phase, num_phases=num_phases, seed=seed)

    logreg = LogisticSGD(d=d, k=k, lr=0.05, wd=1e-4, name="LogReg")
    mlp    = MLPOnline(d=d, k=k, h=32, lr=0.01, wd=1e-5, name="MLP(32)")
    grbfn  = GrowingRBFNet(d=d, k=k, sigma=0.65, lr_w=0.25, lr_c=0.06, min_phi_to_cover=0.25, max_prototypes=90, name="GRBFN")
    grbfnp = GrowingRBFNetPlastic(d=d, k=k, sigma=0.65, lr_w=0.25, lr_c=0.06, min_phi_to_cover=0.25,
                                  hebb_eta=0.6, hebb_decay=0.94, gate_beta=1.2, gate_top=16, key_lr=0.03,
                                  r_merge=0.35, merge_every=600, max_prototypes=90, name="GRBFN+Plastic")

    models = [logreg, mlp, grbfn, grbfnp]

    total_steps = steps_per_phase * num_phases
    acc = {m.name: np.zeros(total_steps, dtype=float) for m in models}
    proto_counts = np.zeros((total_steps, 2), dtype=int)  # for GRBFN, GRBFN+Plastic

    drift_points = [i * steps_per_phase for i in range(1, num_phases)]

    for t in range(total_steps):
        x, y = data.sample(t)
        for m in models:
            yhat = m.predict(x)
            acc[m.name][t] = 1.0 if yhat == y else 0.0
        logreg.update(x, y); mlp.update(x, y); grbfn.update(x, y); grbfnp.update(x, y)
        proto_counts[t, 0] = grbfn.M
        proto_counts[t, 1] = grbfnp.M

    ma = {name: moving_average(acc[name], w=200) for name in acc}
    # Post-drift metrics
    post200 = {m.name: [] for m in models}
    post500 = {m.name: [] for m in models}
    for dp in drift_points:
        a, b = dp, min(dp + 200, total_steps)
        c, d2 = dp, min(dp + 500, total_steps)
        for m in models:
            post200[m.name].append(float(np.mean(acc[m.name][a:b])))
            post500[m.name].append(float(np.mean(acc[m.name][c:d2])))
    overall = {m.name: float(np.mean(acc[m.name])) for m in models}

    # Save figures
    x = np.arange(total_steps)
    plt.figure(figsize=(8,5))
    for name, series in ma.items():
        plt.plot(x, series, label=name)
    for dp in drift_points:
        plt.axvline(dp, linestyle='--', linewidth=1)
    plt.title("Classification under drift: moving accuracy (w=200)")
    plt.xlabel("Step"); plt.ylabel("Accuracy (moving avg)")
    plt.legend()
    acc_path = os.path.join(out_dir, "classification_accuracy.png")
    plt.tight_layout(); plt.savefig(acc_path); plt.close()

    plt.figure(figsize=(8,3.8))
    plt.plot(x, proto_counts[:,0], label="GRBFN")
    plt.plot(x, proto_counts[:,1], label="GRBFN+Plastic")
    for dp in drift_points: plt.axvline(dp, linestyle='--', linewidth=1)
    plt.title("Prototype growth over time")
    plt.xlabel("Step"); plt.ylabel("# Prototypes"); plt.legend()
    proto_path = os.path.join(out_dir, "classification_prototypes.png")
    plt.tight_layout(); plt.savefig(proto_path); plt.close()

    # Tables
    overall_df = pd.DataFrame({"Model": list(overall.keys()),
                               "Overall Accuracy": [round(v, 4) for v in overall.values()]}).sort_values("Model")
    post_df = pd.DataFrame({
        "Model": list(post200.keys()),
        "Post@200(avg)": [round(np.mean(v), 4) for v in post200.values()],
        "Post@500(avg)": [round(np.mean(v), 4) for v in post500.values()],
    }).sort_values("Model")
    overall_df.to_csv(os.path.join(out_dir, "classification_overall.csv"), index=False)
    post_df.to_csv(os.path.join(out_dir, "classification_postdrift.csv"), index=False)

    return {"overall": overall_df, "post": post_df, "acc_path": acc_path, "proto_path": proto_path}

if __name__ == '__main__':
    out = run_experiment(out_dir='../results', seed=21)
    print(out["overall"])
    print(out["post"])
