"""Synthetic task data generators for LST experiments.

Tasks:
  * MQAR (Multi-Query Associative Recall): the canonical SSM-vs-attention
    diagnostic. Sequence of (key, value) pairs followed by queries; the model
    must output the value associated with each queried key.
  * Induction: a 2-step induction-head task.
  * CopyMemory: a long-range copy task.
"""
from __future__ import annotations
import torch
from torch.utils.data import IterableDataset


# ---------------------------------------------------------------------------
# MQAR — Multi-Query Associative Recall
# ---------------------------------------------------------------------------
class MQARDataset(IterableDataset):
    """Multi-Query Associative Recall.

    Sequence layout (input tokens, length L = 2*n_kv + n_q):
      input  = [k1, v1, k2, v2, ..., kn, vn, q1, q2, ..., qm]
      target = [v1, k2, v2, k3, ..., vn, q1, a1, q2, a2, ..., q_{m-1}, a_m]
    where a_i is the value paired with key q_i in the preceding KV pairs.
    We score (with -100 ignore) only:
      * the value position in each KV pair (target = v_i at index 2*i+1)
      * the answer position after each query (target = a_i at index 2*n_kv+i+1)

    The "answer" positions are the MQAR score positions: the model sees q_i
    at position 2*n_kv+i and must predict a_i as the next token.

    Args:
      n_kv_pairs: number of (key, value) pairs to memorise
      n_queries:  number of query tokens at the end
      vocab_keys: number of distinct key tokens
      vocab_vals: number of distinct value tokens
      seq_len:    if set, pads with zeros to this length (else uses L)
      score_kv:   if True, also score the value positions in KV pairs (auxiliary)
    """

    def __init__(self, n_kv_pairs: int = 8, n_queries: int = 4,
                 vocab_keys: int = 16, vocab_vals: int = 16,
                 seq_len: int | None = None, seed: int = 0,
                 score_kv: bool = True):
        self.n_kv_pairs = n_kv_pairs
        self.n_queries = n_queries
        self.vocab_keys = vocab_keys
        self.vocab_vals = vocab_vals
        self.score_kv = score_kv
        self.effective_len = 2 * n_kv_pairs + n_queries
        self.seq_len = seq_len if seq_len is not None else self.effective_len
        self.vocab = 1 + vocab_keys + vocab_vals
        self.seed = seed
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def __iter__(self):
        while True:
            yield self._sample()

    def _sample(self):
        n_kv = self.n_kv_pairs
        n_q = self.n_queries
        # Distinct keys so the lookup is unambiguous
        keys = torch.randperm(self.vocab_keys, generator=self._gen)[:n_kv] + 1
        vals = torch.randint(self.vocab_keys + 1, self.vocab_keys + self.vocab_vals + 1,
                             (n_kv,), generator=self._gen)
        kv = torch.stack([keys, vals], dim=1).reshape(-1)              # (2*n_kv,)
        # Queries: with replacement so the same key may be queried multiple times
        q_idx = torch.randint(0, n_kv, (n_q,), generator=self._gen)
        queries = keys[q_idx]                                          # (n_q,)
        answers = vals[q_idx]                                          # (n_q,)
        seq = torch.cat([kv, queries])                                 # (2*n_kv + n_q,)
        # Pad if needed
        if seq.numel() < self.seq_len:
            pad = torch.zeros(self.seq_len - seq.numel(), dtype=torch.long)
            seq = torch.cat([seq, pad])
        else:
            seq = seq[:self.seq_len]
        # Targets: next-token prediction
        target = torch.full((self.seq_len,), -100, dtype=torch.long)
        # Score value positions in KV pairs: at index 2*i+1, target is v_i
        if self.score_kv:
            for i in range(n_kv):
                pos = 2 * i + 1
                if pos < self.seq_len:
                    target[pos] = vals[i].item()
        # Score answer positions: after each query, target is a_i
        for i in range(n_q):
            q_pos = 2 * n_kv + i
            ans_pos = q_pos + 1  # next-token after query
            if ans_pos < self.seq_len:
                target[ans_pos] = answers[i].item()
        return seq, target


# ---------------------------------------------------------------------------
# Induction task — cleaner 2-step induction
# ---------------------------------------------------------------------------
class InductionDataset(IterableDataset):
    """Induction-head task: [A, B, <noise>..., A] -> predict B at the last position.

    The model must use the second occurrence of A to look up what came after
    the first occurrence (B). This is the canonical induction-head task and
    is known to require attention-like retrieval.
    """

    def __init__(self, seq_len: int = 256, vocab: int = 32, seed: int = 0):
        self.seq_len = seq_len
        self.vocab = vocab
        self.seed = seed
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def __iter__(self):
        while True:
            yield self._sample()

    def _sample(self):
        # Use a larger vocab to avoid duplicates of `a` in the noise.
        # The induction task is only well-defined if `a` appears exactly twice.
        # We resample `a` until it does not collide with the noise tokens.
        a = torch.randint(1, 1 + self.vocab, (1,), generator=self._gen).item()
        b = torch.randint(1, 1 + self.vocab, (1,), generator=self._gen).item()
        # Sample noise that excludes `a` (so `a` appears exactly twice: at pos 0 and pos -1)
        noise = torch.randint(1, 1 + self.vocab, (self.seq_len - 3,), generator=self._gen)
        # Re-sample any noise element that equals `a`
        while (noise == a).any():
            mask = (noise == a)
            n_replace = mask.sum().item()
            noise[mask] = torch.randint(1, 1 + self.vocab, (n_replace,), generator=self._gen)
        seq = torch.empty(self.seq_len, dtype=torch.long)
        seq[0] = a
        seq[1] = b
        seq[2:-1] = noise
        seq[-1] = a
        target = torch.full((self.seq_len,), -100, dtype=torch.long)
        # Predict b at the position of the second 'a' (index seq_len-1)
        target[-1] = b
        return seq, target


# ---------------------------------------------------------------------------
# CopyMemory: long-range copy task
# ---------------------------------------------------------------------------
class CopyMemoryDataset(IterableDataset):
    """Copy a marker token's neighbour to the end.

    Sequence: [marker, v, <noise>..., END]
    Target at END: v  (the value right after the marker).
    """

    def __init__(self, seq_len: int = 256, vocab: int = 16, seed: int = 0):
        self.seq_len = seq_len
        self.vocab = vocab
        self.seed = seed
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def __iter__(self):
        while True:
            yield self._sample()

    def _sample(self):
        v = torch.randint(3, 3 + self.vocab, (1,), generator=self._gen).item()
        seq = torch.randint(3, 3 + self.vocab, (self.seq_len,), generator=self._gen)
        seq[0] = 1  # marker
        seq[1] = v
        seq[-1] = 2  # END
        target = torch.full((self.seq_len,), -100, dtype=torch.long)
        target[-1] = v
        return seq, target


def collate_fixed(batch):
    seqs, targets = zip(*batch)
    return torch.stack(seqs), torch.stack(targets)


# ---------------------------------------------------------------------------
# Induction task v2: multiple (key, value) pairs, multiple queries
# Tests in-context learning of an associative map via induction heads.
# Each sequence has P pairs, then P queries (shuffled). Scored at every
# query position. This gives P * batch_size scored tokens per batch,
# providing strong gradient signal.
# ---------------------------------------------------------------------------
class InductionPairsDataset(IterableDataset):
    """Multi-pair induction task.

    Sequence layout:
      input  = [k1, v1, k2, v2, ..., kP, vP, q1, q2, ..., qP]
      target = at each query position q_i, predict v_{j} where k_j == q_i.

    Uses a large key vocabulary so keys don't repeat in the noise.
    Scored positions: every query (P per sequence), giving strong gradients.
    """
    def __init__(self, n_pairs: int = 8, vocab_keys: int = 64, vocab_vals: int = 32,
                 seq_len: int | None = None, seed: int = 0):
        self.n_pairs = n_pairs
        self.vocab_keys = vocab_keys
        self.vocab_vals = vocab_vals
        self.effective_len = 2 * n_pairs + n_pairs + 1  # P pairs + P queries + 1 (for last target)
        self.seq_len = seq_len if seq_len is not None else self.effective_len
        self.seed = seed
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def __iter__(self):
        while True:
            yield self._sample()

    def _sample(self):
        P = self.n_pairs
        # Keys and values from disjoint vocab ranges (like MQAR)
        # Keys: 3 .. 3+vocab_keys-1, Values: 3+vocab_keys .. 3+vocab_keys+vocab_vals-1
        keys = torch.randperm(self.vocab_keys, generator=self._gen)[:P] + 3
        val_offset = 3 + self.vocab_keys
        vals = torch.randint(val_offset, val_offset + self.vocab_vals, (P,), generator=self._gen)
        # Interleave keys and values: [k1, v1, k2, v2, ..., kP, vP]
        kv = torch.stack([keys, vals], dim=1).reshape(-1)  # (2*P,)
        # Queries: shuffle the keys so each query matches exactly one pair
        query_order = torch.randperm(P, generator=self._gen)
        queries = keys[query_order]
        seq = torch.cat([kv, queries])
        # Pad if needed
        if seq.numel() < self.seq_len:
            pad = torch.zeros(self.seq_len - seq.numel(), dtype=torch.long)
            seq = torch.cat([seq, pad])
        else:
            seq = seq[:self.seq_len]
        # Targets: at each query position, predict the corresponding value
        target = torch.full((self.seq_len,), -100, dtype=torch.long)
        for i in range(P):
            q_pos = 2 * P + i
            ans_pos = q_pos + 1  # next-token after query
            if ans_pos < self.seq_len:
                # The query at position q_pos is queries[i] = keys[query_order[i]]
                # The answer is vals[query_order[i]]
                target[ans_pos] = vals[query_order[i]].item()
        return seq, target



# ---------------------------------------------------------------------------
# Copy task v2: copy a random segment to the end
# Tests long-range copying with multiple scored positions.
# ---------------------------------------------------------------------------
class CopySegmentDataset(IterableDataset):
    """Copy a random segment of length L to the end of the sequence.

    Sequence: [marker, s1, s2, ..., sL, noise..., marker, s1, s2, ..., sL]
    Target: at each position after the second marker, predict the corresponding
    segment token. This gives L scored positions per sequence.
    """
    def __init__(self, seq_len: int = 256, copy_len: int = 8, vocab: int = 16,
                 seed: int = 0):
        self.seq_len = seq_len
        self.copy_len = copy_len
        self.vocab = vocab
        self.seed = seed
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def __iter__(self):
        while True:
            yield self._sample()

    def _sample(self):
        L = self.copy_len
        marker = 1
        end_marker = 2
        # The segment to copy
        segment = torch.randint(3, 3 + self.vocab, (L,), generator=self._gen)
        # Noise fills the gap between the first segment and the second marker
        # Layout: [marker, segment (L), noise, end_marker, segment (L)]
        # Total length: 1 + L + noise_len + 1 + L = seq_len
        # So noise_len = seq_len - 2*L - 2
        noise_len = self.seq_len - 2 * L - 2
        if noise_len < 0:
            noise_len = 0
        noise = torch.randint(3, 3 + self.vocab, (noise_len,), generator=self._gen)
        seq = torch.cat([
            torch.tensor([marker], dtype=torch.long),
            segment,
            noise,
            torch.tensor([end_marker], dtype=torch.long),
            segment,
        ])
        if seq.numel() < self.seq_len:
            pad = torch.zeros(self.seq_len - seq.numel(), dtype=torch.long)
            seq = torch.cat([seq, pad])
        seq = seq[:self.seq_len]
        # Targets: at the end_marker position and each subsequent position,
        # predict the corresponding segment token (next-token prediction).
        target = torch.full((self.seq_len,), -100, dtype=torch.long)
        end_marker_pos = 1 + L + noise_len  # position of end_marker
        for i in range(L):
            # At position end_marker_pos + i, predict segment[i]
            pos = end_marker_pos + i
            if pos < self.seq_len:
                target[pos] = segment[i].item()
        return seq, target
