from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnvConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    name: str
    seed: int
    steps_per_phase: int
    num_phases: int
    env: EnvConfig
    models: List[ModelConfig]
    plot_settings: Dict[str, Any] = field(default_factory=dict)

# --- Base parameters ---

BASE_ENV_PARAMS = {
    "d": 2,
    "steps_per_phase": 2000,
    "num_phases": 6,
}

BASE_GRBF_PARAMS = {
    "sigma": 0.7, "lr_w": 0.25, "lr_c": 0.06, "min_phi_to_cover": 0.25,
    "max_prototypes": 90, "hebb_eta": 0.6, "hebb_decay": 0.94,
    "gate_beta": 1.2, "gate_top": 16, "key_lr": 0.03, "r_merge": 0.4,
    "merge_every": 600,
}

# --- Experiment-specific Configurations ---

# 1. Supervised Classification with Drift
supervised_drift_config = ExperimentConfig(
    name="supervised_drift",
    seed=21,
    steps_per_phase=BASE_ENV_PARAMS["steps_per_phase"],
    num_phases=BASE_ENV_PARAMS["num_phases"],
    env=EnvConfig(
        name="MixtureDriftGaussians",
        params={"d": 2, "k": 3, "n_components": 2, "sigma": 0.4},
    ),
    models=[
        ModelConfig(name="LogisticSGD", params={"d": 2, "k": 3, "lr": 0.05, "wd": 1e-4}),
        ModelConfig(name="MLPOnline", params={"d": 2, "k": 3, "h": 32, "lr": 0.01, "wd": 1e-5}),
        ModelConfig(name="GRBFN+Plastic", params={
            **BASE_GRBF_PARAMS,
            "d": 2, "k": 3, "sigma": 0.65, "r_merge": 0.35, "growth_gate_budget": None,
        }),
    ],
    plot_settings={
        "type": "classification",
        "proto_out_file": "classification_prototypes.png",
        "acc_out_file": "classification_accuracy.png",
    }
)

# 2. Contextual Bandit with Drift (vs. LinUCB)
bandit_drift_config = ExperimentConfig(
    name="bandit_drift",
    seed=33,
    steps_per_phase=BASE_ENV_PARAMS["steps_per_phase"],
    num_phases=BASE_ENV_PARAMS["num_phases"],
    env=EnvConfig(
        name="ContextualBanditDrift",
        params={"d": 2, "A": 3, "n_components": 2, "sigma": 0.5},
    ),
    models=[
        ModelConfig(name="LinUCB", params={"d": 2, "A": 3, "alpha": 0.8, "lam": 1.0}),
        ModelConfig(name="GRBFN+Plastic", params={
            **BASE_GRBF_PARAMS,
            "d": 2, "k": 3, "growth_gate_budget": 22,
        }),
    ],
    plot_settings={
        "type": "bandit",
        "filename_prefix": "bandit",
    }
)

# 3. Contextual Bandit with Drift (vs. LinUCB and MLP)
bandit_drift_plusmlp_config = ExperimentConfig(
    name="bandit_drift_plusmlp",
    seed=37,
    steps_per_phase=BASE_ENV_PARAMS["steps_per_phase"],
    num_phases=BASE_ENV_PARAMS["num_phases"],
    env=EnvConfig(
        name="ContextualBanditDrift",
        params={"d": 2, "A": 3, "n_components": 2, "sigma": 0.5},
    ),
    models=[
        ModelConfig(name="LinUCB", params={"d": 2, "A": 3, "alpha": 0.8, "lam": 1.0}),
        ModelConfig(name="GRBFN+Plastic", params={
            **BASE_GRBF_PARAMS,
            "d": 2, "k": 3, "growth_gate_budget": 22,
        }),
        ModelConfig(name="MLPOnline", params={"d": 2, "k": 3, "h": 32, "lr": 0.01, "wd": 1e-5}),
    ],
    plot_settings={
        "type": "bandit",
        "filename_prefix": "bandit_plusmlp",
    }
)

# --- Main Configuration Dictionary ---

CONFIGS = {
    "supervised": supervised_drift_config,
    "bandit": bandit_drift_config,
    "bandit_plusmlp": bandit_drift_plusmlp_config,
}
