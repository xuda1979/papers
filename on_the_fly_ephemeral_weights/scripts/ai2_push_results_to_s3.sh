#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_ROOT="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/on-the-fly-ephemeral-weights"
REMOTE_ROOT="/root/root/work/on-the-fly-ephemeral-weights"
cd "$ROOT_DIR"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

REMOTE_PATHS=("$@")
if [[ ${#REMOTE_PATHS[@]} -eq 0 ]]; then
  REMOTE_PATHS=(results figures)
fi

for remote_path in "${REMOTE_PATHS[@]}"; do
  if [[ "$(uname -s)" == "Linux" && "$ROOT_DIR" == "$REMOTE_ROOT" ]]; then
    if [[ -e "$remote_path" ]]; then
      DIRECT_CMD=(rclone copy "$remote_path" "$S3_ROOT/$remote_path" --s3-no-check-bucket --progress)
      if [[ $DRY_RUN -eq 1 ]]; then
        DIRECT_CMD+=(--dry-run)
      fi
      "${DIRECT_CMD[@]}"
    else
      echo "skip missing: $remote_path"
    fi
    continue
  fi

  REMOTE_CMD="cd $REMOTE_ROOT && if [[ -e '$remote_path' ]]; then rclone copy '$remote_path' '$S3_ROOT/$remote_path' --s3-no-check-bucket --progress"
  if [[ $DRY_RUN -eq 1 ]]; then
    REMOTE_CMD+=" --dry-run"
  fi
  REMOTE_CMD+="; else echo 'skip missing: $remote_path'; fi"
  JSON_OUT="$(bash scripts/ai2_shell.sh "$REMOTE_CMD")"
  python3 - <<'PY' "$JSON_OUT"
import json, sys
payload = json.loads(sys.argv[1])
text = payload.get('after', '') or ''
for marker in ('ERROR :', 'NOTICE: Failed', 'AccessDenied', 'Failed to'):
    if marker in text:
        raise SystemExit(text)
PY
done
