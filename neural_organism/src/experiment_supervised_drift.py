import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data_generators import MixtureDriftGaussians
from .baselines import LogisticSGD, MLPOnline
from .growing_rbf_net_plastic import GrowingRBFNetPlastic, RuntimeGate
from .experiment_utils import plot_classification_results


def run_experiment(out_dir, seed=21):
    """Run the supervised classification experiment with non-stationary drift."""
    os.makedirs(out_dir, exist_ok=True)
    d, k = 2, 3
    steps_per_phase = 2000
    num_phases = 6
    data = MixtureDriftGaussians(
        d=d,
        k=k,
        n_components=2,
        sigma=0.4,
        steps_per_phase=steps_per_phase,
        num_phases=num_phases,
        seed=seed,
    )

    # In this experiment, we allow unlimited growth to test adaptation speed.
    models = [
        LogisticSGD(d=d, k=k, lr=0.05, wd=1e-4, name="LogReg"),
        MLPOnline(d=d, k=k, h=32, lr=0.01, wd=1e-5, name="MLP(32)"),
        GrowingRBFNetPlastic(d=d, k=k, sigma=0.65, lr_w=0.25, lr_c=0.06, min_phi_to_cover=0.25, growth_gate=RuntimeGate(None), hebb_eta=0.6, hebb_decay=0.94, gate_beta=1.2, gate_top=16, key_lr=0.03, r_merge=0.35, merge_every=600, max_prototypes=90, name="GRBFN+Plastic"),
    ]

    total_steps = steps_per_phase * num_phases
    corrects = {m.name: np.zeros(total_steps, dtype=float) for m in models}
    proto_counts = {
        "GRBFN+Plastic": np.zeros(total_steps, dtype=int),
    }

    # Main loop
    for t in range(total_steps):
        x, y = data.sample(t)
        for m in models:
            yhat = m.predict(x)
            corrects[m.name][t] = 1.0 if yhat == y else 0.0
            m.update(x, y)
        proto_counts["GRBFN+Plastic"][t] = models[2].M

    # Plotting
    acc_path = plot_classification_results(
        corrects, out_dir, "classification_accuracy.png", window_size=200
    )

    plt.figure(figsize=(8, 3.8))
    plt.plot(proto_counts["GRBFN+Plastic"], label="GRBFN+Plastic")
    for dp in [i * steps_per_phase for i in range(1, num_phases)]:
        plt.axvline(dp, linestyle='--', linewidth=1)
    plt.title("Prototype growth over time")
    plt.xlabel("Step")
    plt.ylabel("# Prototypes")
    plt.legend()
    proto_path = os.path.join(out_dir, "classification_prototypes.png")
    plt.tight_layout()
    plt.savefig(proto_path)
    plt.close()

    # Performance tables
    drift_points = [i * steps_per_phase for i in range(1, num_phases)]
    post200 = {m.name: [] for m in models}
    post500 = {m.name: [] for m in models}
    for dp in drift_points:
        for m in models:
            post200[m.name].append(np.mean(corrects[m.name][dp : dp + 200]))
            post500[m.name].append(np.mean(corrects[m.name][dp : dp + 500]))

    overall_df = pd.DataFrame({
        "Model": [m.name for m in models],
        "Overall Accuracy": [round(np.mean(corrects[m.name]), 4) for m in models],
    }).sort_values("Model")

    post_df = pd.DataFrame({
        "Model": [m.name for m in models],
        "Post@200(avg)": [round(np.mean(v), 4) for v in post200.values()],
        "Post@500(avg)": [round(np.mean(v), 4) for v in post500.values()],
    }).sort_values("Model")

    overall_df.to_csv(os.path.join(out_dir, "classification_overall.csv"), index=False)
    post_df.to_csv(os.path.join(out_dir, "classification_postdrift.csv"), index=False)

    return {"overall": overall_df, "post": post_df, "acc_path": acc_path, "proto_path": proto_path}


if __name__ == '__main__':
    out = run_experiment(out_dir='neural_organism/results', seed=21)
    print("\nOverall Performance:")
    print(out["overall"])
    print("\nPost-Drift Adaptation:")
    print(out["post"])
