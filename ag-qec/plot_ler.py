#!/usr/bin/env python3
"""Re-generate logical error rate figure as a vector PDF with error bars."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from channel_models import wilson_interval

plt.rcParams["font.size"] = 8

BASE = Path(__file__).resolve().parent
data = pd.read_csv(BASE / "data/ag_qec_results.csv")
# Use Wilson interval with N=1e6 trials as in the paper
N = 1_000_000
failures = (data["p_L"] * N).astype(int)
lo, hi = zip(*(wilson_interval(f, N) for f in failures))

yerr_lower = data["p_L"] - pd.Series(lo)
yerr_upper = pd.Series(hi) - data["p_L"]

fig, ax = plt.subplots()
ax.errorbar(data["p_eff"], data["p_L"], yerr=[yerr_lower, yerr_upper],
            fmt="o", capsize=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Physical error rate $p$")
ax.set_ylabel("Logical error rate")
ax.grid(True, which="both", ls=":")
fig.tight_layout()
fig.savefig(BASE / "logical_error_rate.pdf")
