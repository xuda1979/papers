# Setup and Usage Guide

## Environment Setup

To set up the Conda environment locally, run the provided batch script:

```cmd
setup_conda_env.bat
```

This will:
1. Create a new Conda environment named `rl_quantum_control` with Python 3.9.
2. Install the dependencies listed in `requirements.txt`.

## Using NPU Acceleration

The system now supports NPU acceleration (e.g., for Ascend NPUs or other devices supported by PyTorch).

To use the NPU, pass the `--npu` flag to the CLI commands.

### Examples

Train a model using NPU:
```bash
python cli.py train --task gate_synthesis --target H --agent dqn --npu
```

Evaluate a model using NPU:
```bash
python cli.py evaluate --model model.pt --task gate_synthesis --target H --npu
```

Run a single execution using NPU:
```bash
python cli.py run --model model.pt --task gate_synthesis --target H --npu
```

## Requirements

Ensure you have the necessary drivers and PyTorch version installed for your specific NPU hardware. The standard `requirements.txt` installs the CPU/CUDA version of PyTorch. You may need to install a specific version for NPU support.
