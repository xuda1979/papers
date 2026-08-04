#!/usr/bin/env bash
set -euo pipefail

# Submit DTFN experiments to ASI3 NPU cluster (1 NPU).
# Mirrors h_ds3m_paper/scripts/submit_asi3_one_npu.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
ITRF_ROOT="$WORKSPACE_ROOT/independent_papers/itrf_paper"
SUBMIT_JS="$ITRF_ROOT/browser-automation/huanxin_submit_task_run.js"

RUN_ID="${DTFN_RUN_ID:-dtfn-asi3-1npu-$(date -u +%Y%m%dT%H%M%SZ)}"
TASK_NAME="${DTFN_TASK_NAME:-dtfn-asi3-1npu}"
PROFILE_COPY_NAME="${HUANXIN_PROFILE_COPY_NAME:-dtfn-asi3-one-npu}"
DRY_RUN=0
SUBMIT=0

usage() {
  cat <<'EOF' >&2
Usage: bash scripts/submit_asi3_one_npu.sh [--dry-run] --submit [--run-id ID]
  --dry-run    Print the remote command without submitting
  --submit     Actually submit the task to the NPU cluster
  --run-id ID  Unique run identifier (default: dtfn-asi3-1npu-<timestamp>)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --submit) SUBMIT=1; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    *) usage ;;
  esac
done

if [[ $DRY_RUN -eq 1 ]]; then
  bash "$ROOT_DIR/scripts/remote_dtfn_run.sh" --dry-run --run-id "$RUN_ID" --device-list 0
  exit 0
fi

[[ $SUBMIT -eq 1 ]] || usage
[[ -f "$SUBMIT_JS" ]] || {
  echo "Huanxin submit helper not found: $SUBMIT_JS" >&2
  exit 2
}

cd "$WORKSPACE_ROOT"
export HUANXIN_HEADLESS="${HUANXIN_HEADLESS:-1}"
export HUANXIN_PROFILE_COPY_NAME="$PROFILE_COPY_NAME"

# Export DTFN-specific env for the remote launcher
export DTFN_RUN_ID="$RUN_ID"
export DTFN_REMOTE_ROOT="/root/software/dtfn"
export DTFN_S3_ROOT="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/dtfn-paper-main"
export DTFN_REMOTE_DEVICE_LIST=0
export DTFN_HUANXIN_ACCELERATOR_CARDS=1

node "$SUBMIT_JS" \
  --env ASI3 \
  --task-name "$TASK_NAME" \
  --accelerator-cards 1 \
  --cpu-cores 20 \
  --memory-gb 240 \
  --submit \
  --launcher-script dtfn/scripts/remote_dtfn_run.sh \
  --launcher-arg --run-id \
  --launcher-arg "$RUN_ID" \
  --launcher-arg --device-list \
  --launcher-arg 0 \
  --screenshot dtfn/results/huanxin-submit-task-run.png \
  --dump-html dtfn/results/huanxin-submit-task-run.html \
  --dump-json dtfn/results/huanxin-submit-task-run.json
