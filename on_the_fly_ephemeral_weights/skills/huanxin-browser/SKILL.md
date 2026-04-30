# Huanxin Browser Automation

Use Huanxin ai2 for heavier experiments.

## Environment

- Environment: `ai2`
- Remote workdir: `/root/root/work/on-the-fly-ephemeral-weights`

## Default Entry Point

Use:

```bash
scripts/ai2_shell.sh "cd /root/root/work/on-the-fly-ephemeral-weights && <command>"
```

This workspace includes a local `browser-automation/` bundle, so the shell wrapper is self-contained and does not depend on `~/.openclaw/workspace-quantum-rnd`.

For Codex or other local coding agents, this is the main automatic entrypoint.

The transfer helpers are also dual-mode:

- when run locally, they route through `scripts/ai2_shell.sh`
- when run inside `/root/root/work/on-the-fly-ephemeral-weights` on ai2, they execute `rclone` directly

## Long Jobs

Use background jobs and durable local handles:

```bash
scripts/ai2_job.sh start on-the-fly-exp /tmp/on-the-fly-exp.log "python3 experiments/benchmark_ephemeral_weights.py"
scripts/ai2_job.sh status on-the-fly-exp-20260324T000000Z
scripts/ai2_job.sh logs on-the-fly-exp-20260324T000000Z 120
scripts/ai2_job.sh list
```

Use `scripts/ai2_job.sh` instead of raw `nohup` when you need a durable local job handle. The plain shell wrapper only returns terminal snapshots.

## Rules

- ai2 only
- Prefer S3 relay for bulk file transfer
- Do not modify other bots' remote directories
