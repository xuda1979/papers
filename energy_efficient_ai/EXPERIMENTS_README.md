# SSA Experiments: Reproducibility Guide

## Overview

This document describes the experiments validating Spectral Sparse Attention (SSA).
The experiments are designed to run within the compute constraints of 20 Ascend 910B NPUs.

## Quick Start

```bash
# Test locally on CPU
python experiments_final.py --quick

# Run full experiments on CPU  
python experiments_final.py

# Run on Ascend NPU (requires MindSpore)
python experiments_final.py --npu

# Run on NVIDIA GPU (requires PyTorch)
python experiments_final.py --gpu
```

## Command Line Options

| Flag | Description |
|------|-------------|
| `--npu` | Use Ascend NPU with MindSpore backend |
| `--gpu` | Use NVIDIA GPU with PyTorch backend |
| `--quick` | Run quick sanity check only |
| `--output-dir DIR` | Specify output directory for results |

## Experimental Results Summary

### Validated Experiments (Fully Run)

| Experiment | Hardware | Time | Status |
|------------|----------|------|--------|
| Needle-in-Haystack Retrieval | CPU | ~5 min | ✅ Complete |
| Sparsity Analysis | CPU | ~2 min | ✅ Complete |
| Scalability (NumPy reference) | CPU | ~10 min | ✅ Complete |
| Output Quality Analysis | CPU | ~15 min | ✅ Complete |

### Projected Experiments (Require Full Training)

| Experiment | Hardware Required | Time Est. | Status |
|------------|-------------------|-----------|--------|
| WikiText-103 Perplexity | 4× Ascend 910B | ~8 hours | 📊 Projected |
| Long Range Arena | 8× Ascend 910B | ~12 hours | 📊 Projected |
| GPU Kernel Benchmarks | NVIDIA A100 | ~1 hour | 📊 Projected |

## Key Results

### 1. Needle-in-Haystack Retrieval (Table 4 in paper)

SSA achieves **perfect retrieval (1.000)** across all tested sequence lengths,
matching dense attention while local attention fails completely:

| N | Dense | SSA | Routing Trans. | Local-256 | Reformer |
|---|-------|-----|----------------|-----------|----------|
| 512 | 1.000 | **1.000** | 1.000 | 0.016 | 1.000 |
| 1024 | 1.000 | **1.000** | 1.000 | 0.008 | 1.000 |
| 2048 | 1.000 | **1.000** | 1.000 | -0.007 | 1.000 |
| 4096 | 1.000 | **1.000** | 1.000 | -0.018 | 1.000 |

### 2. Sparsity Analysis (Table 5 in paper)

SSA achieves 74-95% sparsity, with edges per query scaling as O(√N):

| N | Clusters k | Edges | Sparsity | Edges/Query |
|---|------------|-------|----------|-------------|
| 512 | 22 | 69,330 | 73.6% | 135 |
| 1024 | 32 | 161,836 | 84.6% | 158 |
| 2048 | 45 | 504,890 | 88.0% | 247 |
| 4096 | 64 | 1,249,389 | 92.5% | 305 |
| 8192 | 90 | 3,390,280 | 94.9% | 414 |

### 3. Output Quality

SSA output cosine similarity to dense attention:
- SSA (k=5): 0.65-0.75 depending on sequence length
- Better than Routing Transformer (0.27-0.35) and Local (0.25-0.65)
- Comparable to Reformer (0.68)

## Running the Experiments

### Prerequisites

```bash
pip install numpy matplotlib
```

### Run All Experiments

```bash
python experiments_final.py
```

This produces:
- `ssa_experiment_results.json`: Raw numerical results
- Console output with formatted tables

### Generate Plots

```bash
python generate_plots.py
```

This produces:
- `exp1_scalability.png`: Runtime scaling comparison
- `exp2_spectral.png`: Eigenvalue preservation
- `exp3_long_range.png`: Needle-in-haystack bar chart
- `exp4_energy.png`: Energy proxy scaling
- `exp5_ablation.png`: Cluster count ablation
- `exp6_pareto.png`: Accuracy-efficiency Pareto frontier

## Implementation Notes

### NumPy Reference vs Production

The NumPy implementation is a **reference implementation** for validating
theoretical properties. It has high Python overhead that makes SSA slower
than dense attention at small N. A production implementation would:

1. Use optimized GPU kernels (CUDA/Triton)
2. Implement block-sparse attention patterns
3. Use fused k-means + attention operations
4. Leverage hardware-specific optimizations (Ascend CANN, etc.)

### Why SSA is Slow in NumPy

The NumPy implementation loops over:
1. K-means iterations (10 iterations × N points)
2. Per-query cluster routing (N queries × k clusters)
3. Per-query sparse attention (N queries × variable keys)

With optimized GPU kernels, these become:
1. Single batched k-means kernel
2. Vectorized top-k cluster selection
3. Block-sparse attention (cuSPARSE or custom)

Expected speedup: 100-1000× over NumPy reference.

## Ascend 910B Deployment Guide

### For WikiText-103 Training (4 NPUs)

```python
# Pseudo-code for distributed training
import mindspore as ms
from mindspore import context
from mindspore.communication import init

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
init()

# Model: GPT-2 Small (125M params)
# Batch size: 8 per NPU × 4 NPUs = 32 effective
# Sequence length: 1024
# Training time: ~8 hours
```

### For LRA Benchmark (8 NPUs)

```python
# Focus on 3 tasks: Text, Retrieval, PathFinder
# These test long-range dependencies most directly
# Estimated time: ~12 hours total

TASKS = ['text', 'retrieval', 'pathfinder']
for task in TASKS:
    # Train 2-layer transformer with SSA attention
    # Sequence lengths: 1K-4K tokens
    pass
```

## Citation

If you use this code, please cite:

```bibtex
@article{ssa2024,
  title={Spectral Sparse Attention: Sub-Quadratic Attention via Cluster-Based Sparsification},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License
