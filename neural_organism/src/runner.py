import argparse
import os
import numpy as np
import pandas as pd

# --- Model and Environment Imports ---
from .data_generators import MixtureDriftGaussians, ContextualBanditDrift
from .baselines import LogisticSGD, MLPOnline, LinUCB
from .growing_rbf_net_plastic import GrowingRBFNetPlastic, RuntimeGate
from .config import CONFIGS, ExperimentConfig
from .experiment_utils import (
    plot_bandit_results,
    save_bandit_summary,
    plot_classification_results,
    plot_prototypes,
)

# --- Model and Environment Factory ---

def get_env(config: ExperimentConfig):
    """Instantiates an environment from a configuration."""
    env_map = {
        "MixtureDriftGaussians": MixtureDriftGaussians,
        "ContextualBanditDrift": ContextualBanditDrift,
    }
    env_class = env_map[config.env.name]
    # Add the seed to the env params
    env_params = {**config.env.params, "seed": config.seed}
    return env_class(**env_params)


def get_models(config: ExperimentConfig):
    """Instantiates a list of models from a configuration."""
    model_map = {
        "LogisticSGD": LogisticSGD,
        "MLPOnline": MLPOnline,
        "LinUCB": LinUCB,
        "GRBFN+Plastic": GrowingRBFNetPlastic,
    }
    models = []
    for model_conf in config.models:
        model_class = model_map[model_conf.name]
        params = model_conf.params.copy()
        # Handle the special case of the RuntimeGate for GRBFN+Plastic
        if model_conf.name == "GRBFN+Plastic":
            budget = params.pop("growth_gate_budget", None)
            # Rename k to A for bandit envs
            if config.env.name == "ContextualBanditDrift":
                params["k"] = params.pop("A", params.get("k"))
            params["growth_gate"] = RuntimeGate(budget=budget)

        # Rename k to A for bandit envs for other models
        if config.env.name == "ContextualBanditDrift" and "k" in params:
             params["k"] = params.pop("A", params.get("k"))

        models.append(model_class(name=model_conf.name, **params))
    return models


# --- Experiment Logic ---

def run_supervised(config: ExperimentConfig, out_dir: str):
    """Runs a supervised learning experiment."""
    env = get_env(config)
    models = get_models(config)

    total_steps = config.steps_per_phase * config.num_phases
    corrects = {m.name: np.zeros(total_steps, dtype=float) for m in models}
    proto_counts = {
        m.name: np.zeros(total_steps, dtype=int)
        for m in models if isinstance(m, GrowingRBFNetPlastic)
    }

    for t in range(total_steps):
        x, y = env.sample(t)
        for m in models:
            yhat = m.predict(x)
            corrects[m.name][t] = 1.0 if yhat == y else 0.0
            m.update(x, y)
            if isinstance(m, GrowingRBFNetPlastic):
                proto_counts[m.name][t] = m.M

    # --- Results and Plotting ---
    print("--- Supervised Experiment Results ---")

    # Overall Accuracy Table
    overall_df = pd.DataFrame({
        "Model": [m.name for m in models],
        "Overall Accuracy": [round(np.mean(corrects[m.name]), 4) for m in models],
    }).sort_values("Model")
    print("\nOverall Performance:")
    print(overall_df)
    overall_df.to_csv(os.path.join(out_dir, "classification_overall.csv"), index=False)

    # Post-Drift Adaptation Table
    drift_points = [i * config.steps_per_phase for i in range(1, config.num_phases)]
    post200 = {m.name: [] for m in models}
    post500 = {m.name: [] for m in models}
    for dp in drift_points:
        for m in models:
            post200[m.name].append(np.mean(corrects[m.name][dp : dp + 200]))
            post500[m.name].append(np.mean(corrects[m.name][dp : dp + 500]))

    post_df = pd.DataFrame({
        "Model": [m.name for m in models],
        "Post@200(avg)": [round(np.mean(v), 4) for v in post200.values()],
        "Post@500(avg)": [round(np.mean(v), 4) for v in post500.values()],
    }).sort_values("Model")
    print("\nPost-Drift Adaptation:")
    print(post_df)
    post_df.to_csv(os.path.join(out_dir, "classification_postdrift.csv"), index=False)

    # Plots
    plot_settings_for_func = config.plot_settings.copy()
    plot_settings_for_func.pop("type", None)
    plot_classification_results(corrects, out_dir, plot_settings=plot_settings_for_func)
    for model_name, counts in proto_counts.items():
        plot_prototypes(counts, model_name, out_dir, config.plot_settings["proto_out_file"])
    print(f"\nPlots saved to {out_dir}")


def run_bandit(config: ExperimentConfig, out_dir: str):
    """Runs a contextual bandit experiment."""
    env = get_env(config)
    models = get_models(config)

    A = config.env.params["A"]
    total_steps = config.steps_per_phase * config.num_phases

    results = {m.name: {"cum_rewards": np.zeros(total_steps), "cum_regrets": np.zeros(total_steps)} for m in models}
    proto_counts = {
        m.name: np.zeros(total_steps, dtype=int)
        for m in models if isinstance(m, GrowingRBFNetPlastic)
    }

    for t in range(total_steps):
        x = env.sample(t)

        # Oracle for regret calculation
        ps = [env._reward_prob(x, a) for a in range(A)]
        best_r_prob = np.max(ps)

        for m in models:
            # --- Action Selection and Update ---
            if isinstance(m, LinUCB):
                a = m.select(x)
                r, _ = env.pull(x, a)
                m.update(x, a, r)
            elif isinstance(m, (GrowingRBFNetPlastic, MLPOnline)):
                # Heuristic policy: predict best action, get reward, update
                if isinstance(m, GrowingRBFNetPlastic):
                    a, p = m.predict(x, return_proba=True)
                else: # MLPOnline
                    _, p = m._forward(x)
                    a = int(np.argmax(p))

                r, _ = env.pull(x, a)

                # Convert bandit feedback to a supervised-like signal for update.
                # This is a heuristic: if reward is positive, treat the chosen action
                # as the correct label. If not, encourage exploration by moving
                # towards the second-best action.
                if r > 0:
                    y = a
                else:
                    y = int(np.argsort(p)[-2])
                m.update(x, y)
            else:
                raise TypeError(f"Model {m.name} not supported in bandit experiment.")

            # --- Logging ---
            regret = best_r_prob - ps[a]
            results[m.name]["cum_rewards"][t] = r if t == 0 else results[m.name]["cum_rewards"][t-1] + r
            results[m.name]["cum_regrets"][t] = regret if t == 0 else results[m.name]["cum_regrets"][t-1] + regret

            if isinstance(m, GrowingRBFNetPlastic):
                proto_counts[m.name][t] = m.M

    # --- Results and Plotting ---
    print("--- Bandit Experiment Results ---")

    # Summary Table
    summary_data = []
    for m in models:
        final_protos = np.nan
        budget_rem = np.nan
        if isinstance(m, GrowingRBFNetPlastic):
            final_protos = proto_counts[m.name][-1]
            budget_rem = m.growth_gate._budget

        summary_data.append({
            "Policy": m.name,
            "Final Cumulative Reward": results[m.name]["cum_rewards"][-1],
            "Final Cumulative Regret": results[m.name]["cum_regrets"][-1],
            "Final #Prototypes": final_protos,
            "Growth Budget Remaining": budget_rem,
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df)

    prefix = config.plot_settings["filename_prefix"]
    summary_path = save_bandit_summary(summary_data, out_dir, f"{prefix}_summary.csv")
    print(f"\nSummary saved to {summary_path}")

    # Plots
    plot_settings_for_func = config.plot_settings.copy()
    plot_settings_for_func.pop("type", None)
    plot_bandit_results(results, out_dir, plot_settings=plot_settings_for_func)
    print(f"Plots saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run experiments for the Neural Organism project.")
    parser.add_argument(
        "experiment",
        type=str,
        choices=CONFIGS.keys(),
        help="The name of the experiment to run.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="neural_organism/results",
        help="Directory to save results.",
    )
    args = parser.parse_args()

    config = CONFIGS[args.experiment]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Running experiment: {config.name} (Seed: {config.seed})")

    if config.plot_settings["type"] == "classification":
        run_supervised(config, args.out_dir)
    elif config.plot_settings["type"] == "bandit":
        run_bandit(config, args.out_dir)
    else:
        raise ValueError(f"Unknown experiment type: {config.plot_settings['type']}")

if __name__ == "__main__":
    main()
