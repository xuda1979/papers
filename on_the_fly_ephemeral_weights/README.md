# On-the-Fly Ephemeral Weights

This workspace develops an English research paper on dynamically generated transient weights for memory-bound large-model inference.

## Main Files

- `paper.tex`: paper draft
- `experiments/benchmark_ephemeral_weights.py`: first numerical benchmark scaffold
- `results/`: experiment outputs
- `figures/`: generated figures
- `scripts/`: Huanxin ai2 and S3 workflow helpers
- `browser-automation/`: local Huanxin browser shell bundle for direct Codex use

## Research Question

Can we trade excess compute for lower memory traffic by generating low-rank FFN weights on the fly, consuming them immediately, and never materializing the full matrix in HBM?

## First Experimental Program

1. Build a roofline-style analytical model for dense FFN vs. low-rank ephemeral FFN
2. Simulate bandwidth, arithmetic intensity, and latency under different model sizes
3. Evaluate sensitivity to rank, hidden dimension, batch size, sequence length, and generator cost
4. Produce break-even curves for next-generation compute-rich, bandwidth-constrained accelerators
