#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_ROOT="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/on-the-fly-ephemeral-weights"
RCLONE_BIN="${RCLONE_BIN:-$(command -v rclone || true)}"
if [[ -z "$RCLONE_BIN" && -x /Users/daxu/homebrew/bin/rclone ]]; then
  RCLONE_BIN=/Users/daxu/homebrew/bin/rclone
fi

cd "$ROOT_DIR"

if [[ -z "$RCLONE_BIN" || ! -x "$RCLONE_BIN" ]]; then
  echo 'rclone not found. Set RCLONE_BIN or install rclone.' >&2
  exit 1
fi

DRY_RUN=0
ALLOW_BULKY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --all)
      ALLOW_BULKY=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

SOURCE_PATHS=("$@")
if [[ ${#SOURCE_PATHS[@]} -eq 0 ]]; then
  SOURCE_PATHS=(.)
fi

BASE_ARGS=(
  --exclude ".git/**"
  --exclude "__pycache__/**"
  --exclude ".pytest_cache/**"
  --exclude ".mypy_cache/**"
  --exclude ".ruff_cache/**"
  --exclude "*.pyc"
  --exclude "node_modules/**"
  --exclude ".venv/**"
  --exclude "venv/**"
  --progress
  --transfers 8
)

BULKY_ARGS=(
  --exclude "results/**"
  --exclude "figures/**"
  --exclude "artifacts/**"
  --exclude "memory/**"
  --exclude "logs/**"
  --exclude "*.pt"
  --exclude "*.pth"
  --exclude "*.bin"
  --exclude "*.safetensors"
  --exclude "*.ckpt"
  --exclude "*.tar"
  --exclude "*.zip"
)

if [[ $DRY_RUN -eq 1 ]]; then
  BASE_ARGS+=(--dry-run)
fi

for source_path in "${SOURCE_PATHS[@]}"; do
  [[ -e "$source_path" ]] || { echo "missing path: $source_path" >&2; exit 1; }
  rel_path="${source_path#./}"
  copy_args=("${BASE_ARGS[@]}")
  if [[ "$source_path" == "." && $ALLOW_BULKY -eq 0 ]]; then
    copy_args+=("${BULKY_ARGS[@]}")
  fi
  if [[ "$source_path" == "." ]]; then
    "$RCLONE_BIN" copy . "$S3_ROOT" "${copy_args[@]}"
  elif [[ -d "$source_path" ]]; then
    "$RCLONE_BIN" copy "$source_path" "$S3_ROOT/$rel_path" "${copy_args[@]}"
  else
    parent_dir="$(dirname "$rel_path")"
    if [[ "$parent_dir" == "." ]]; then
      dest="$S3_ROOT"
    else
      dest="$S3_ROOT/$parent_dir"
    fi
    "$RCLONE_BIN" copy "$source_path" "$dest" "${copy_args[@]}"
  fi
done
