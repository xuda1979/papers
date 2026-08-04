#!/usr/bin/env bash
set -euo pipefail

# Entry script executed on the remote NPU host.
# Syncs code from S3, runs the experiment battery, pushes results back.
#
# Usage:
#   bash scripts/remote_dtfn_entry.sh --run-id <id> --device-list <ids> [--smoke]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID=""
DEVICE_LIST="0"
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --device-list) DEVICE_LIST="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) shift ;;
  esac
done

[[ -n "$RUN_ID" ]] || RUN_ID="dtfn-remote-$(date -u +%Y%m%dT%H%M%SZ)"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE_LIST"

echo "=== DTFN remote NPU run ==="
echo "RUN_ID=$RUN_ID"
echo "DEVICE_LIST=$DEVICE_LIST"
echo "SMOKE=$SMOKE"
echo "PWD=$(pwd)"
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"

# Verify torch_npu is available
python3 -c "import torch; import torch_npu; print('torch_npu OK, devices:', torch.npu.device_count())" || {
  echo "WARNING: torch_npu not available, falling back to CPU"
}

# Run the experiment battery
mkdir -p results

if [[ $SMOKE -eq 1 ]]; then
  echo "=== SMOKE TEST ==="
  python3 -m innovations.dtfn_experiments_v5 --exp exp1 exp2 --smoke --out results/remote_dtfn_smoke.json 2>&1 | tee results/remote_dtfn_log.txt
else
  echo "=== FULL RUN ==="
  python3 -m innovations.dtfn_experiments_v5 --exp all --out results/remote_dtfn_results.json 2>&1 | tee results/remote_dtfn_log.txt
fi

echo "=== Run complete. Results in results/ ==="
ls -la results/
