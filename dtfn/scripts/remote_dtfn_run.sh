#!/usr/bin/env bash
set -euo pipefail

# Remote launcher for DTFN experiments on Ascend NPU.
# Mirrors the h_ds3m_paper remote_ai_run.sh pattern.
#
# Usage:
#   bash scripts/remote_dtfn_run.sh --run-id <id> --device-list <ids>
#   bash scripts/remote_dtfn_run.sh --smoke --run-id <id> --device-list 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="${DTFN_REMOTE_ROOT:-/root/software/dtfn}"
S3_ROOT="${DTFN_S3_ROOT:-nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/dtfn-paper-main}"
HX_ENV_RAW="${DTFN_HUANXIN_ENV:-ASI3}"
REMOTE_CMD_BASE="${DTFN_REMOTE_CMD:-bash scripts/remote_dtfn_entry.sh}"

SMOKE=0
DRY_RUN=0
RUN_ID=""
DEVICE_LIST=""

usage() {
  cat >&2 <<EOF
Usage: scripts/remote_dtfn_run.sh [--smoke] [--dry-run] --run-id <id> --device-list <ids>
  --smoke          Quick smoke test (small config, 1 epoch)
  --dry-run        Print the payload that would be submitted, then exit
  --run-id <id>    Unique run identifier
  --device-list <ids>   Comma-separated NPU device ids (e.g. "0" or "0,1,2")
  --remote-cmd <cmd>    Override the remote entry command
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --device-list) DEVICE_LIST="$2"; shift 2 ;;
    --remote-cmd) REMOTE_CMD_BASE="$2"; shift 2 ;;
    *) usage ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  if [[ $SMOKE -eq 1 ]]; then
    RUN_ID="dtfn-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  else
    RUN_ID="dtfn-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
fi

normalize_env() {
  local v="$1"
  # Uppercase (bash 3 compatible: tr instead of ${v^^})
  v="$(printf '%s' "$v" | tr '[:lower:]' '[:upper:]')"
  case "$v" in
    AI|AI1|AI2|AI3|ASI1|ASI2|ASI3) echo "$v" ;;
    *) echo "ASI3" ;;
  esac
}

HX_ENV="$(normalize_env "$HX_ENV_RAW")"
cd "$ROOT_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  python3 - "$HX_ENV" "$RUN_ID" "$REMOTE_ROOT" "$S3_ROOT" "$REMOTE_CMD_BASE" \
    "${DEVICE_LIST:-}" "$SMOKE" <<'PY'
import json, sys
(env_name, run_id, remote_root, s3_root, remote_cmd, device_list, smoke) = sys.argv[1:]
accelerator_cards = len([d for d in device_list.split(",") if d.strip()]) if device_list else 1
cpu_cores = 20
memory_gb = 240
payload = {
    "env": env_name,
    "run_id": run_id,
    "smoke": smoke == "1",
    "remote_root": remote_root,
    "s3_root": s3_root,
    "remote_command": remote_cmd,
    "launcher_script": "scripts/remote_dtfn_entry.sh",
    "result_transport": "s3",
    "emitted_result_files": [
        "results/remote_dtfn_results.json",
        "results/remote_dtfn_log.txt",
    ],
    "instance_count": 1,
    "accelerator_cards": accelerator_cards,
    "cpu_cores": cpu_cores,
    "memory_gb": memory_gb,
    "preferred_accelerator_cards_per_instance": int(accelerator_cards),
    "preferred_cpu_per_instance": int(cpu_cores),
    "preferred_memory_gb_per_instance": int(memory_gb),
    "allow_quota_downshift": True,
    "supports_npu_fanout": True,
    "uses_all_visible_npus": True,
}
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
  exit 0
fi

# Delegate to the huanxin_shell.sh infrastructure (shared with h_ds3m_paper).
WORKSPACE_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
ITRF_ROOT="$WORKSPACE_ROOT/independent_papers/itrf_paper"
HUANXIN_SHELL="$ITRF_ROOT/scripts/huanxin_shell.sh"

if [[ ! -f "$HUANXIN_SHELL" ]]; then
  echo "huanxin_shell.sh not found at $HUANXIN_SHELL" >&2
  exit 2
fi

export DTFN_RUN_ID="$RUN_ID"
export DTFN_REMOTE_ROOT="$REMOTE_ROOT"
export DTFN_S3_ROOT="$S3_ROOT"
export DTFN_REMOTE_DEVICE_LIST="${DEVICE_LIST:-0}"
export DTFN_HUANXIN_ACCELERATOR_CARDS=1
export DTFN_SMOKE="$SMOKE"

# Build the remote command: sync code, run entry script, sync results back.
REMOTE_FULL_CMD="cd $REMOTE_ROOT && $REMOTE_CMD_BASE --run-id $RUN_ID --device-list ${DEVICE_LIST:-0} $([ $SMOKE -eq 1 ] && echo --smoke)"

bash "$HUANXIN_SHELL" "$HX_ENV" "$REMOTE_FULL_CMD"
