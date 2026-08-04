#!/usr/bin/env bash
# Submit LST training experiments to an Ascend NPU cluster.
#
# This script is a template for running the LST paper's experiments on a
# single Ascend NPU (per the user's "only 1 npu" constraint).  It does NOT
# run locally — it is meant to be submitted to a cluster scheduler (slurm /
# ModelArts / custom) or run directly on an NPU host.
#
# Prerequisites on the NPU host:
#   - Python 3.9+
#   - torch (matching the torch_npu version)
#   - torch_npu  (pip install torch-npu)
#   - numpy, matplotlib  (for figure generation)
#
# Usage on an NPU host:
#   chmod +x submit_npu_job.sh
#   ./submit_npu_job.sh
#
# Usage via slurm (if available):
#   sbatch --partition=npu --gres=npu:1 submit_npu_job.sh
#
# The script runs experiments 1, 4, 5, 6, 7, 9 (the main paper experiments)
# sequentially on a single NPU, writing results to code/results/.

set -euo pipefail
cd "$(dirname "$0")"

echo "=== LST NPU job — $(date) ==="
echo "Host: $(hostname)"
echo "Python: $(which python3)"
python3 -c "import torch; print('torch', torch.__version__)" || true
python3 -c "import torch_npu; print('torch_npu', torch_npu.__version__)" || true
echo

# Export single-NPU visibility (honours the "only 1 npu" constraint).
export ASCEND_VISIBLE_DEVICES=0
export NPU_VISIBLE_DEVICES=0

# Device check
python3 - <<'PY'
from device_utils import get_device
d = get_device()
print(f"Selected device: {d}")
assert d.type == "npu", f"Expected NPU device, got {d}. Is torch_npu installed?"
PY

echo "=== Running experiments on NPU ==="

# Main paper experiments (run in order; each writes to results/)
for exp in \
    exp1_mqar_scaling_v2 \
    exp4_k_ablation_v2 \
    exp5_gate_analysis_v2 \
    exp6_long_range_v2 \
    exp7_noise_v2 \
    exp9_content_adaptive \
    exp3_induction_copy_v2
do
    echo "--- $exp ---"
    python3 "$exp.py" || { echo "FAILED: $exp"; continue; }
done

echo "=== Regenerating figures ==="
python3 make_figures_v2.py || echo "figure regeneration failed"

echo "=== Done — $(date) ==="
