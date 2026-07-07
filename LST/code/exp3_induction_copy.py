"""Experiment 3: Induction-head and CopyMemory tasks.

The induction-head task is the canonical diagnostic for in-context retrieval
(Olsson et al. 2022). CopyMemory tests long-range recall. We test whether LST
solves both at SSM-like cost.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LST, TransformerLM, SSMLM, LinformerLM
from lst.tasks import InductionDataset, CopyMemoryDataset
from lst.train import TrainConfig, train_model


def make_iter(ds, batch_size):
    it = iter(ds)
    def _next():
        nonlocal it
        batch = []
        for _ in range(batch_size):
            try: batch.append(next(it))
            except StopIteration:
                it = iter(ds); batch.append(next(it))
        seqs, tgts = zip(*batch)
        return torch.stack(seqs), torch.stack(tgts)
    return _next


def run_task(task_name: str, seq_len: int, vocab: int, models: list,
             steps: int, batch_size: int, device: str, seed: int = 42):
    if task_name == "induction":
        ds = InductionDataset(seq_len=seq_len, vocab=vocab, seed=0)
        ds_eval = InductionDataset(seq_len=seq_len, vocab=vocab, seed=999)
    elif task_name == "copy":
        ds = CopyMemoryDataset(seq_len=seq_len, vocab=vocab, seed=0)
        ds_eval = CopyMemoryDataset(seq_len=seq_len, vocab=vocab, seed=999)
    else:
        raise ValueError(task_name)
    # The vocab in these tasks is the *noise* vocab; the actual vocab size is
    # 1 + vocab + 3 (pad + marker + END + 1-indexed tokens)
    if task_name == "induction":
        full_vocab = 1 + vocab + 1  # 0=pad, 1..vocab+1 = real tokens
    else:
        full_vocab = 3 + vocab  # 0=pad, 1=marker, 2=END, 3..3+vocab = noise/values
    train_iter = make_iter(ds, batch_size)
    eval_iter = make_iter(ds_eval, batch_size)
    results = []
    for name, model in models:
        # Build a fresh model with the right vocab size
        if name == "LST":
            m = LST(vocab=full_vocab, d_model=128, n_layers=2, k_latent=32, n_heads=4)
        elif name == "Transformer":
            m = TransformerLM(vocab=full_vocab, d_model=128, n_layers=2, n_heads=4)
        elif name == "SSM":
            m = SSMLM(vocab=full_vocab, d_model=128, n_layers=2, d_state=16)
        elif name == "Linformer":
            m = LinformerLM(vocab=full_vocab, d_model=128, n_layers=2, n_heads=4,
                            k_latent=32, max_T=max(4096, seq_len * 2))
        print(f"\n--- {task_name}/{name} seq_len={seq_len} ---", flush=True)
        cfg = TrainConfig(epochs=1, steps_per_epoch=steps, batch_size=batch_size,
                          lr=3e-4, weight_decay=0.0, warmup_steps=100,
                          grad_clip=1.0, eval_every=steps // 4, eval_steps=50,
                          log_every=steps // 4, device=device, seed=seed)
        t0 = time.time()
        hist = train_model(m, train_iter, eval_iter, cfg, label=f"{task_name}/{name}")
        dt = time.time() - t0
        acc = hist['eval_acc'][-1][1]
        loss = hist['eval_loss'][-1][1]
        print(f"RESULT: {task_name}/{name} seq_len={seq_len} acc={acc:.4f} loss={loss:.4f} time={dt:.1f}s", flush=True)
        results.append({
            "task": task_name, "model": name, "seq_len": seq_len,
            "vocab": full_vocab, "params": sum(p.numel() for p in m.parameters()),
            "final_acc": acc, "final_loss": loss, "time_s": dt,
        })
    return results


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    all_results = []
    # Induction at multiple lengths
    for sl in [256, 512, 1024, 2048]:
        r = run_task("induction", seq_len=sl, vocab=32,
                     models=["Transformer", "LST", "SSM", "Linformer"],
                     steps=400, batch_size=64, device=device)
        all_results.extend(r)
        (out_dir / "exp3_partial.json").write_text(json.dumps(all_results, indent=2))
    # CopyMemory
    for sl in [256, 512, 1024, 2048]:
        r = run_task("copy", seq_len=sl, vocab=16,
                     models=["Transformer", "LST", "SSM", "Linformer"],
                     steps=400, batch_size=64, device=device)
        all_results.extend(r)
        (out_dir / "exp3_partial.json").write_text(json.dumps(all_results, indent=2))

    (out_dir / "exp3_induction_copy.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_dir / 'exp3_induction_copy.json'}")


if __name__ == "__main__":
    main()
