# TTC — Transactional Tool Calls (Spec)

**Goal.** Prevent orphaned heavy runs and sunk compute when plans change.

## Interfaces
- `prepare(inputs, est_cost, consumers[]) -> call_id`
- `will_consume(call_id) -> bool` (coordinator arbitration)
- `commit(call_id) -> result`
- `abort(call_id)`

## Enforcement
- Tools must block heavy execution until `commit`
- Coordinator maintains a durable log: `PREPARE`, `COMMIT`, `ABORT`

## Acceptance Tests
1. **Orphan Rate:** ~0 across forced plan changes (≥100 randomized trials)
2. **Bounded Overhead:** `prepare` adds ≤2% latency vs direct call
3. **Auditability:** replay log reconstructs decisions exactly
