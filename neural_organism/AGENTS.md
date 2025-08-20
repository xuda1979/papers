# Agent Instructions for the `neural_organism` Repository

Hello! This file provides guidance for working with the codebase in this repository. The code was recently refactored to be more modular and configuration-driven. Please follow these guidelines to maintain the structure.

## Codebase Architecture

The core logic is in the `src/` directory and is organized as follows:

-   `runner.py`: This is the **single entry point** for all experiments. Do not create new experiment scripts. Instead, add a new configuration to `config.py` and, if necessary, extend the `runner.py` to handle the new experiment type.
-   `config.py`: This file contains **all hyperparameters and experiment configurations**. When you need to add a new experiment or change parameters, this is the first place you should look. All experiments are defined as `ExperimentConfig` dataclasses.
-   `growing_rbf_net_plastic.py`: This is the core research model of the paper. It is a complex, stateful model.
-   `baselines.py`: Contains simpler baseline models used for comparison. If you add a new baseline, it should go here. Ensure it can be instantiated via a parameter dictionary, similar to the existing models.
-   `data_generators.py`: Contains the environments that produce data streams.
-   `experiment_utils.py`: Contains helper functions for plotting and saving results. If you add a new plot type, add a function here.

## How to Add a New Experiment

1.  **Define a Configuration:** Open `src/config.py`. Create a new `ExperimentConfig` instance for your experiment. Give it a unique name (e.g., `my_new_experiment_config`).
2.  **Add Models/Envs (if necessary):** If your experiment requires a new model or environment, add the class definition to `baselines.py` or `data_generators.py`, respectively.
3.  **Update Factories:** Make sure the `get_env` and `get_models` functions in `runner.py` can instantiate your new classes by adding them to the respective `_map` dictionaries.
4.  **Add to Main Config:** Add your new config object to the main `CONFIGS` dictionary at the bottom of `src/config.py`. The key you use here will be the command-line argument to run your experiment.
5.  **Extend Runner (if necessary):** If your experiment has a new *type* of logic (i.e., not "supervised" or "bandit"), you will need to add a new `run_...` function to `runner.py` and call it from the `main` function.

## How to Run Tests

The repository does not currently have a dedicated test suite. The primary method of verification is to run the experiments and check that the results are consistent.

To verify your changes, run all three standard experiments:

```bash
python -m neural_organism.src.runner supervised
python -m neural_organism.src.runner bandit
python -m neural_organism.src.runner bandit_plusmlp
```

Ensure that they run to completion and that the output results in `neural_organism/results/` are sensible.

By following these guidelines, you will help keep the codebase clean, organized, and easy for others to understand and build upon.
