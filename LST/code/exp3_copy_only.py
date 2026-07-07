"""Run just the copy tasks for exp3 v2 (since induction is done)."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LST, LSTG, TransformerLM, SSMLM, LinformerLM
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
        ds_eval = InductionDataset(seq_len=seq_len, vocab=vocab, seed=1)
    else:
        ds = CopyMemoryDataset(seq_len=seq_len, vocab=vocab, seed=0)
        ds_eval = CopyMemoryDataset(seq_len=seq_len, vocab=vocab, seed=1)
    if task_name == "induction":
        full_vocab = 1 + vocab + 1
    else:
        full_vocab = 3 + vocab
    train_iter = make_iter(ds, batch_size)
    eval_iter = make_iter(ds_eval, batch_size)
    results = []
    for name in models:
        if name == "LST":
            m = LST(vocab=full_vocab, d_model=128, n_layers=2, k_latent=32, n_heads=4)
        elif name == "LSTG":
            m = LSTG(vocab=full_vocab, d_model=128, n_layers=2, k_latent=32, n_heads=4)
        elif name == "Transformer":
            m = TransformerLM(vocab=full_vocab, d_model=128, n_layers=2, n_heads=4)
        elif name == "SSM":
            m = SSMLM(vocab=full_vocab, d_model=128, n_layers=2, d_state=16)
        elif name == "Linformer":
            m = LinformerLM(vocab=full_vocab, d_model=128, n_layers=2,
                            k_latent=32, n_heads=4)
        else:
            raise ValueError(f"unknown model {name}")
        m = m.to(device)
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
        n_params = sum(p.numel() for p in m.parameters())
        print(f"RESULT: {task_name}/{name} seq={seq_len} acc={acc:.4f} loss={loss:.4f} time={dt:.1f}s params={n_params:,}", flush=True)
        results.append({"model": name, "task": task_name, "seq_len": seq_len,
                        "params": n_params, "final_acc": acc,
                        "final_loss": loss, "time_s": dt})
        del m
        if device == "mps":
            torch.mps.empty_cache()
    return results


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Load existing partial results (induction is done)
    partial_path = out_dir / "exp3_v2_partial.json"
    if partial_path.exists():
        all_results = json.loads(partial_path.read_text())
    else:
        all_results = []
    
    models = ["Transformer", "LST", "LSTG", "SSM", "Linformer"]
    # CopyMemory
    for sl in [256, 512, 1024, 2048]:
        # Check if already done
        existing = [r for r in all_results if r.get("task") == "copy" and r.get("seq_len") == sl]
        if len(existing) >= len(models):
            print(f"Skipping copy seq={sl} (already done)")
            continue
        r = run_task("copy", seq_len=sl, vocab=16,
                     models=models,
                     steps=400, batch_size=64, device=device)
        all_results.extend(r)
        partial_path.write_text(json.dumps(all_results, indent=2))

    (out_dir / "exp3_induction_copy_v2.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_dir / 'exp3_induction_copy_v2.json'}")


if __name__ == "__main__":
    main()
