# S3 Transfer Skill

Use this skill to transfer files between the local paper workspace, S3, and Huanxin ai2.

## Current Setup

- Local workspace: `~/papers/on_the_fly_ephemeral_weights`
- S3 root: `nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/on-the-fly-ephemeral-weights`
- Remote ai2 workdir: `/root/root/work/on-the-fly-ephemeral-weights`

## Default Scripts

- Local -> S3: `scripts/push_to_s3.sh`
- S3 -> ai2: `scripts/ai2_sync_from_s3.sh`
- ai2 -> S3: `scripts/ai2_push_results_to_s3.sh`

The ai2 sync helpers are dual-mode:

- local invocation from this Mac uses the Huanxin browser shell automatically
- invocation from inside `/root/root/work/on-the-fly-ephemeral-weights` on ai2 runs `rclone` directly

Current defaults are intentionally code-first and exclude bulky artifacts unless you explicitly request them. Use `scripts/push_to_s3.sh --all` only for a truly broad workspace copy.

## Workflow

1. Validate locally
2. Push to S3
3. Sync to ai2
4. Run remote experiments
5. Push results back to S3
6. Pull results locally if needed

## Rules

- Use ai2 only
- Keep all remote work in `/root/root/work/on-the-fly-ephemeral-weights`
- Do not touch paths used by other bots
- Use `--dry-run` first for risky transfers
