# CPCS — Critical-Path Compute Scheduler (Spec)

**Goal.** Minimize expected makespan of a dynamic reasoning DAG under GPU/CPU/tool budgets.

## Interfaces
- `submit(task: Task) -> TaskId` with fields `{id, deps[], eta_hat, p_succ, cost_flops, priority_hint}`
- `update_estimates(task_id, eta_hat?, p_succ?, cost_flops?)`
- `tick(now)` recomputes expected critical path and (re)assigns resources
- `bind(task_id, resources)` where `resources` map to CUDA stream IDs, CPU affinities, cgroup shares
- `preempt(task_id)` issues stream priority drop and/or SIGSTOP to off-path processes

## Enforcement
- Uses `nvidia-cuda-mps` or per-stream priorities; Linux `cgroups.v2` for CPU quotas
- Preemption latency target: 50–200 ms

## Acceptance Tests
1. **Makespan Reduction vs FIFO:** ≥10% on synthetic DAGs with skewed ETAs
2. **Utilization:** GPU busy-time ≥90% under mixed tool/decoder load
3. **Stability:** No starvation for non-critical tasks beyond configurable bound
