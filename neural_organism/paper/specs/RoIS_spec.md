# RoI-S — Region-of-Interest Splicing (Spec)

**Goal.** Repair only the erroneous span without losing prefix/suffix caches.

## Interfaces
- `mark_span(doc_id, start_token, end_token)` from verifier
- `partial_decode(model_id, kv_ckpt, prefix_tokens, target_end)` returns `repair_tokens`
- `splice(doc_id, start, end, repair_tokens)` updates document and KV-store

## Enforcement
- Decoder exposes KV slice/extend primitives; deterministic stitch verified by hash of (prefix, repair, suffix)
- Abort if splice invalidates verifier hash

## Acceptance Tests
1. **Token Savings:** ≥60% fewer regenerated tokens per fix vs full re-decode
2. **Determinism:** identical outputs across 5 runs with fixed RNG
3. **Latency:** ≤40% of full re-decode wall time for 256-token RoIs
