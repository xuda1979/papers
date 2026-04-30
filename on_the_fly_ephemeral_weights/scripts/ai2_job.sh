#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/root/root/work/on-the-fly-ephemeral-weights"
SHELL_WRAPPER="$ROOT_DIR/scripts/ai2_shell.sh"
META_DIR="$ROOT_DIR/.huanxin_jobs"
TAIL_LINES_DEFAULT="${HUANXIN_JOB_TAIL_LINES:-40}"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/ai2_job.sh start <job-name> <log-path> "<remote command>"
  scripts/ai2_job.sh status <job-id> [tail-lines]
  scripts/ai2_job.sh logs <job-id> [tail-lines]
  scripts/ai2_job.sh list
EOF
  exit 1
}

combine_transcript() {
  python3 - <<'PY' "$1"
import json
import sys
payload = json.loads(sys.argv[1])
print("\n".join(part for part in [payload.get('output', ''), payload.get('after', ''), payload.get('before', '')] if part))
PY
}

load_meta_field() {
  python3 - <<'PY' "$1" "$2"
import json
import sys
with open(sys.argv[1], encoding='utf-8') as handle:
    payload = json.load(handle)
print(payload.get(sys.argv[2], ''))
PY
}

mkdir -p "$META_DIR"

case "${1:-}" in
  start)
    [[ $# -eq 4 ]] || usage
    job_name="$2"
    log_path="$3"
    remote_command="$4"
    safe_job_name="$(printf '%s' "$job_name" | tr -cs 'A-Za-z0-9._-' '-')"
    job_id="${safe_job_name}-$(date -u +%Y%m%dT%H%M%SZ)"
    meta_file="$META_DIR/${job_id}.json"
    escaped_remote_command="$(printf '%q' "$remote_command")"
    remote_launch="cd '$REMOTE_ROOT' && nohup bash -lc ${escaped_remote_command} > '$log_path' 2>&1 < /dev/null & pid=\$!; printf '__OPENCLAW_JOB__ id=$job_id pid=%s log=$log_path cwd=$REMOTE_ROOT\n' \"\$pid\""
    json_out="$($SHELL_WRAPPER "$remote_launch")"
    transcript="$(combine_transcript "$json_out")"
    python3 - <<'PY' "$meta_file" "$json_out" "$transcript" "$job_name" "$remote_command"
import json, re, sys
from datetime import datetime, timezone

meta_file, json_out, transcript, job_name, remote_command = sys.argv[1:6]
match = re.search(r"__OPENCLAW_JOB__ id=(\S+) pid=(\d+) log=(\S+) cwd=(\S+)", transcript)
if not match:
    raise SystemExit(f"Did not observe durable job marker.\n--- transcript ---\n{transcript}")
payload = json.loads(json_out)
job_payload = {
    'job_id': match.group(1),
    'name': job_name,
    'pid': int(match.group(2)),
    'log_path': match.group(3),
    'remote_root': match.group(4),
    'command': remote_command,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'transport': payload.get('transport'),
    'duration_ms': payload.get('durationMs'),
    'meta_file': meta_file,
}
with open(meta_file, 'w', encoding='utf-8') as handle:
    json.dump(job_payload, handle, indent=2)
print(json.dumps(job_payload, indent=2))
PY
    ;;
  status|logs)
    [[ $# -ge 2 && $# -le 3 ]] || usage
    job_id="$2"
    tail_lines="${3:-$TAIL_LINES_DEFAULT}"
    meta_file="$META_DIR/${job_id}.json"
    [[ -f "$meta_file" ]] || { echo "unknown job: $job_id" >&2; exit 1; }
    pid="$(load_meta_field "$meta_file" pid)"
    log_path="$(load_meta_field "$meta_file" log_path)"
    remote_root="$(load_meta_field "$meta_file" remote_root)"
    remote_status="cd '$remote_root' && if ps -p '$pid' -o pid=,stat=,etimes=,command= >/dev/null 2>&1; then echo '__OPENCLAW_JOB_STATUS__ running'; ps -p '$pid' -o pid=,stat=,etimes=,command=; else echo '__OPENCLAW_JOB_STATUS__ exited'; fi; if [[ -f '$log_path' ]]; then echo '__OPENCLAW_JOB_LOG__'; tail -n '$tail_lines' '$log_path'; else echo '__OPENCLAW_JOB_LOG_MISSING__'; fi"
    json_out="$($SHELL_WRAPPER "$remote_status")"
    transcript="$(combine_transcript "$json_out")"
    python3 - <<'PY' "$meta_file" "$json_out" "$transcript"
import json, sys
meta_file, json_out, transcript = sys.argv[1:4]
with open(meta_file, encoding='utf-8') as handle:
    meta = json.load(handle)
status = 'running' if '__OPENCLAW_JOB_STATUS__ running' in transcript else 'exited' if '__OPENCLAW_JOB_STATUS__ exited' in transcript else 'unknown'
ps_lines = []
capture = False
for line in transcript.splitlines():
    if line.startswith('__OPENCLAW_JOB_STATUS__'):
        capture = True
        continue
    if line.startswith('__OPENCLAW_JOB_LOG__') or line.startswith('__OPENCLAW_JOB_LOG_MISSING__'):
        capture = False
    if capture and line.strip():
        ps_lines.append(line)
log_tail = transcript.split('__OPENCLAW_JOB_LOG__\n', 1)[1].strip() if '__OPENCLAW_JOB_LOG__\n' in transcript else ''
result = {**meta, 'status': status, 'ps': ps_lines, 'log_tail': log_tail, 'transport': json.loads(json_out).get('transport'), 'duration_ms': json.loads(json_out).get('durationMs')}
print(json.dumps(result, indent=2))
PY
    ;;
  list)
    python3 - <<'PY' "$META_DIR"
import glob, json, os, sys
items = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    with open(path, encoding='utf-8') as handle:
        items.append(json.load(handle))
print(json.dumps(items, indent=2))
PY
    ;;
  *)
    usage
    ;;
esac