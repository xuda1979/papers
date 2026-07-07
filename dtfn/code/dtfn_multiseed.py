"""Multi-seed experiments for statistical significance."""
from __future__ import annotations
import argparse, json, os, random, sys, time
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dtfn_core as dc
import dtfn_experiments_v2 as ex


def run_seeds(seeds: List[int], epochs: int, device_name: str, output: str) -> None:
    device = ex.get_device(device_name)
    train_ds = ex.DyckDataset(2000, min_pairs=4, max_pairs=32, num_bracket_types=1, seed=0)
    test_ds = ex.DyckDataset(500, min_pairs=4, max_pairs=32, num_bracket_types=1, seed=42)
    pad_id = train_ds.pad_id
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=lambda b: ex.collate(b, pad_id))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             collate_fn=lambda b: ex.collate(b, pad_id))

    models_factories = {
        'DTFN-U': lambda: dc.make_dtfn_unidirectional(vocab_size, d=64, K=3),
        'DTFN-B': lambda: dc.make_dtfn_bidirectional(vocab_size, d=64, K=3),
        'DTFN-MUSCL': lambda: dc.make_dtfn_muscl(vocab_size, d=64, K=3),
        'DTFN-FULL': lambda: dc.make_dtfn_full(vocab_size, d=64, K=3),
        'LSTM': lambda: ex.LSTMModel(vocab_size, d=64, hidden_dim=128),
        'Transformer': lambda: ex.TransformerModel(vocab_size, d=64, n_heads=4, n_layers=2, max_len=256),
    }

    all_results = {}
    for name, factory in models_factories.items():
        print(f"\n=== {name} ===")
        all_results[name] = []
        for seed in seeds:
            ex.set_seed(seed)
            model = factory()
            tr = ex.train_model(model, train_loader, epochs=epochs, device=device,
                                lr=1e-3, pad_id=pad_id, seed=seed, verbose=False)
            ev = ex.evaluate_model(model, test_loader, device, pad_id=pad_id)
            res = {'seed': seed, 'train': tr.__dict__, 'test': ev}
            all_results[name].append(res)
            print(f"  seed={seed}: test_loss={ev['loss']:.4f}  tk_acc={ev['token_acc']:.4f}  seq_acc={ev['seq_acc']:.4f}")

    # Aggregate
    print("\n=== Aggregated (mean ± std) ===")
    summary = {}
    for name, runs in all_results.items():
        losses = [r['test']['loss'] for r in runs]
        tk_accs = [r['test']['token_acc'] for r in runs]
        sq_accs = [r['test']['seq_acc'] for r in runs]
        summary[name] = {
            'loss_mean': float(np.mean(losses)), 'loss_std': float(np.std(losses)),
            'tk_acc_mean': float(np.mean(tk_accs)), 'tk_acc_std': float(np.std(tk_accs)),
            'seq_acc_mean': float(np.mean(sq_accs)), 'seq_acc_std': float(np.std(sq_accs)),
            'n_seeds': len(runs),
            'all_losses': losses, 'all_tk_accs': tk_accs, 'all_sq_accs': sq_accs,
        }
        print(f"{name:12s} loss={np.mean(losses):.4f}±{np.std(losses):.4f}  tk={np.mean(tk_accs):.4f}±{np.std(tk_accs):.4f}  seq={np.mean(sq_accs):.4f}±{np.std(sq_accs):.4f}")

    with open(output, 'w') as f:
        json.dump({'summary': summary, 'per_seed': all_results}, f, indent=2, default=str)
    print(f"\nSaved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="/Users/daxu/papers/dtfn/results/multiseed.json")
    args = parser.parse_args()
    run_seeds(args.seeds, args.epochs, args.device, args.output)


if __name__ == "__main__":
    main()
