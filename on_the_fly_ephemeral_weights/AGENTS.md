# AGENTS.md - Workspace Rules

## Session Startup

Before doing anything else:

1. Read `SOUL.md`
2. Read `USER.md`
3. Read `IDENTITY.md`
4. Inspect `paper.tex`
5. Inspect `experiments/` and `results/`

Do not ask permission before starting technical work.

## Main Mission

This workspace is for one paper only: on-the-fly ephemeral weights for transforming the memory wall into a compute wall.

## Expectations

- Write in English
- Prefer measurable claims over rhetorical claims
- Keep experiments reproducible
- Update the paper when experiments change the story
- Do not leak into other workspaces or agents

## Huanxin Usage

- Use ai2 only
- Use local workspace scripts for transfer and shell execution
- Keep remote work inside `/root/root/work/on-the-fly-ephemeral-weights`
- This workspace now contains its own `browser-automation/` bundle copied locally, so Codex can use `scripts/ai2_shell.sh` directly without relying on OpenClaw-managed files in another workspace.
- Default shell entrypoint for Codex: `scripts/ai2_shell.sh "<cmd>"`
- Default long-job entrypoint: `scripts/ai2_job.sh start <name> <log-path> "<cmd>"`
- Use S3 relay scripts for code movement, not browser copy/paste.

## Safety

- No destructive commands unless necessary and justified
- Do not modify other agents, folders, or remote directories
- Use S3 relay for bulk transfers
