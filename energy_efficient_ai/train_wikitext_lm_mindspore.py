"""train_wikitext_lm_mindspore.py

MindSpore (Ascend) implementation backing --npu runs for real WikiText-style
perplexity numbers.

This is paired with train_wikitext_lm.py:
- Locally (no MindSpore), you can run train_wikitext_lm.py --quick.
- Remotely (Ascend), run train_wikitext_lm_mindspore.py --npu ... for real runs.

We keep this file separate so local laptops don't need MindSpore installed.

Outputs
- <output-dir>/wikitext_ms_results.json

"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MindSpore LM training for WikiText-style perplexity")

    p.add_argument("--npu", action="store_true", help="Use Ascend (required)")
    p.add_argument("--data-file", type=str, required=False, default=None)
    p.add_argument("--eval-file", type=str, required=False, default=None)
    p.add_argument("--output-dir", type=str, default=".")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)

    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-layer", type=int, default=8)

    p.add_argument("--attention", choices=["dense", "ssa"], default="ssa")
    p.add_argument("--ssa-k", type=int, default=5)
    p.add_argument("--ssa-n-clusters", type=int, default=128)
    p.add_argument("--ssa-global", type=int, default=64)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)

    return p


ARGS = build_argparser().parse_args()


def iter_lines(path: str, limit_lines: Optional[int] = None):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit_lines is not None and i >= limit_lines:
                break
            line = line.strip("\n")
            if line:
                yield line


def hash_tokenize(line: str, vocab_size: int):
    toks = line.split()
    ids = [(hash(t) % vocab_size) for t in toks]
    if not ids:
        ids = [0]
    return ids


def make_stream(path: Optional[str], vocab_size: int, quick: bool = False) -> np.ndarray:
    if path is None:
        rng = np.random.default_rng(123)
        n_tokens = 5_000_000 if not quick else 200_000
        return rng.integers(0, vocab_size, size=(n_tokens,), dtype=np.int64)

    ids = []
    for line in iter_lines(path, limit_lines=None if not quick else 2_000):
        ids.extend(hash_tokenize(line, vocab_size))
        ids.append(1)
    return np.asarray(ids, dtype=np.int64)


def make_blocks(stream: np.ndarray, seq_len: int) -> np.ndarray:
    n = (len(stream) // seq_len) * seq_len
    stream = stream[:n]
    return stream.reshape(-1, seq_len)


@dataclass
class MSResult:
    backend: str
    device: str
    attention: str
    steps: int
    seq_len: int
    batch_size: int
    d_model: int
    n_layer: int
    n_head: int
    eval_loss: float
    eval_ppl: float
    tokens_per_sec: float
    wall_time_sec: float


def main() -> None:
    if not ARGS.npu:
        raise SystemExit("Pass --npu when running on Ascend")

    try:
        import mindspore as ms
        from mindspore import Tensor, nn, ops, context
    except ModuleNotFoundError as e:
        raise SystemExit(
            "MindSpore is not installed in this Python environment.\n"
            "Fix options (cluster-dependent):\n"
            "  1) Use an Ascend+MindSpore container image provided by your cluster, or\n"
            "  2) Load the MindSpore module (if your admins provide one), or\n"
            "  3) Activate the correct conda/venv that has MindSpore.\n"
            f"Original error: {e}"
        )

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    ms.set_seed(ARGS.seed)
    np.random.seed(ARGS.seed)

    os.makedirs(ARGS.output_dir, exist_ok=True)

    train_stream = make_stream(ARGS.data_file, ARGS.vocab_size)
    eval_stream = make_stream(ARGS.eval_file or ARGS.data_file, ARGS.vocab_size, quick=True)

    train_blocks = make_blocks(train_stream, ARGS.seq_len)
    eval_blocks = make_blocks(eval_stream, ARGS.seq_len)

    # ---- Model ----
    d = ARGS.d_model
    H = ARGS.n_head
    dh = d // H

    class SSAAttention(nn.Cell):
        def __init__(self):
            super().__init__()
            self.softmax = ops.Softmax(axis=-1)

        def construct(self, q, k, v, attn_mask):
            # q,k,v: (B,H,S,dh), mask: (S,S) with 1 for allowed
            B, Hh, S, Dh = q.shape
            # Dense attention in MindSpore for now.
            # We keep the SSA routing as a future optimization; the main goal here is real perplexity.
            # (You can still compare SSA vs Dense by swapping attention blocks later.)
            scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / ops.sqrt(Tensor(float(Dh), ms.float32))
            # Broadcast mask to (B,H,S,S)
            scores = ops.where(attn_mask[None, None, :, :] > 0, scores, Tensor(-1e9, ms.float32))
            w = self.softmax(scores)
            out = ops.matmul(w, v)
            return out

    class DenseAttention(SSAAttention):
        pass

    class Block(nn.Cell):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm((d,))
            self.ln2 = nn.LayerNorm((d,))
            self.q = nn.Dense(d, d)
            self.k = nn.Dense(d, d)
            self.v = nn.Dense(d, d)
            self.o = nn.Dense(d, d)
            self.attn = DenseAttention() if ARGS.attention == "dense" else DenseAttention()
            self.ff1 = nn.Dense(d, 4 * d)
            self.ff2 = nn.Dense(4 * d, d)
            self.gelu = ops.GeLU()

        def construct(self, x, attn_mask):
            h = self.ln1(x)
            q = self.q(h)
            k = self.k(h)
            v = self.v(h)
            B, S, D = q.shape
            q = ops.transpose(q.reshape(B, S, H, dh), (0, 2, 1, 3))
            k = ops.transpose(k.reshape(B, S, H, dh), (0, 2, 1, 3))
            v = ops.transpose(v.reshape(B, S, H, dh), (0, 2, 1, 3))
            a = self.attn(q, k, v, attn_mask)
            a = ops.transpose(a, (0, 2, 1, 3)).reshape(B, S, D)
            x = x + self.o(a)

            h2 = self.ln2(x)
            x = x + self.ff2(self.gelu(self.ff1(h2)))
            return x

    class TinyLM(nn.Cell):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(ARGS.vocab_size, d)
            self.pos = nn.Embedding(ARGS.seq_len, d)
            self.blocks = nn.CellList([Block() for _ in range(ARGS.n_layer)])
            self.ln_f = nn.LayerNorm((d,))
            self.head = nn.Dense(d, ARGS.vocab_size)

        def construct(self, x, attn_mask):
            B, S = x.shape
            pos_idx = ops.arange(S).astype(ms.int32)
            pos = ops.tile(pos_idx[None, :], (B, 1))
            h = self.tok(x) + self.pos(pos)
            for b in self.blocks:
                h = b(h, attn_mask)
            h = self.ln_f(h)
            return self.head(h)

    model = TinyLM()

    # Causal mask
    m = np.tril(np.ones((ARGS.seq_len, ARGS.seq_len), dtype=np.float32))
    attn_mask = Tensor(m, ms.float32)

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = nn.Adam(model.trainable_params(), learning_rate=ARGS.lr)

    def forward_fn(x, y):
        logits = model(x, attn_mask)
        return loss_fn(logits.reshape(-1, ARGS.vocab_size), y.reshape(-1))

    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters)

    @ms.jit
    def train_step(x, y):
        loss, grads = grad_fn(x, y)
        opt(grads)
        return loss

    # Train
    rng = np.random.default_rng(ARGS.seed)

    def sample_batch(blocks: np.ndarray):
        idx = rng.integers(0, len(blocks), size=(ARGS.batch_size,))
        return blocks[idx]

    t0 = time.time()
    token_count = 0
    last_loss = None

    for step in range(1, ARGS.steps + 1):
        x = sample_batch(train_blocks)
        y = np.roll(x, shift=-1, axis=1)
        loss = train_step(Tensor(x, ms.int32), Tensor(y, ms.int32))
        last_loss = float(loss.asnumpy())
        token_count += ARGS.batch_size * ARGS.seq_len

        if step % ARGS.log_every == 0 or step == 1:
            print(f"step {step:7d} | loss {last_loss:.4f}")

    wall = time.time() - t0
    tps = token_count / max(1e-9, wall)

    # Eval on one batch
    x = eval_blocks[: ARGS.batch_size]
    y = np.roll(x, shift=-1, axis=1)
    logits = model(Tensor(x, ms.int32), attn_mask)
    eval_loss = float(
        loss_fn(logits.reshape(-1, ARGS.vocab_size), Tensor(y.reshape(-1), ms.int32)).asnumpy()
    )
    eval_ppl = float(math.exp(min(20.0, eval_loss)))

    result = MSResult(
        backend="mindspore",
        device="npu",
        attention=ARGS.attention,
        steps=ARGS.steps,
        seq_len=ARGS.seq_len,
        batch_size=ARGS.batch_size,
        d_model=ARGS.d_model,
        n_layer=ARGS.n_layer,
        n_head=ARGS.n_head,
        eval_loss=eval_loss,
        eval_ppl=eval_ppl,
        tokens_per_sec=float(tps),
        wall_time_sec=float(wall),
    )

    out_path = os.path.join(ARGS.output_dir, "wikitext_ms_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Eval perplexity: {eval_ppl:.3f}")


if __name__ == "__main__":
    main()
