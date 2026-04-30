#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_ROOT="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/on-the-fly-ephemeral-weights"
REMOTE_ROOT="/root/root/work/on-the-fly-ephemeral-weights"
cd "$ROOT_DIR"

RCLONE_CMD="rclone sync '$S3_ROOT' '$REMOTE_ROOT' --exclude '.git/**' --exclude '__pycache__/**' --exclude '.pytest_cache/**' --exclude '.mypy_cache/**' --exclude '.ruff_cache/**' --exclude '*.pyc' --exclude 'node_modules/**' --exclude '.venv/**' --exclude 'venv/**' --exclude 'results/**' --exclude 'figures/**' --exclude 'artifacts/**' --exclude 'memory/**' --exclude 'logs/**' --exclude 'browser-automation/profile/**' --exclude 'browser-automation/*.png' --exclude 'browser-automation/*.html' --exclude 'browser-automation/*.json' --progress"
if [[ "${1:-}" == "--dry-run" ]]; then
  RCLONE_CMD+=" --dry-run"
fi

if [[ "$(uname -s)" == "Linux" && "$ROOT_DIR" == "$REMOTE_ROOT" ]]; then
  eval "$RCLONE_CMD"
  exit 0
fi

JSON_OUT="$(bash scripts/ai2_shell.sh "$RCLONE_CMD")"
python3 - <<'PY' "$JSON_OUT"
import json, sys
payload = json.loads(sys.argv[1])
text = payload.get('after', '') or ''
for marker in ('ERROR :', 'NOTICE: Failed', 'AccessDenied', 'Failed to'):
    if marker in text:
        raise SystemExit(text)
PY
