"""Dyck-2 (two bracket types) experiment: a harder counting task."""
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


def run_dyck2(epochs: int, device_name: str, output: str, seed: int = 0) -> None:
    device = ex.get_device(device_name)
    # Dyck-2: 2 bracket types -> vocab_size = 2*2+3 = 7
    train_ds = ex.DyckDataset(3000, min_pairs=4, max_pairs=24, num_bracket_types=2, seed=0)
    test_ds = ex.DyckDataset(500, min_pairs=4, max_pairs=24, num_bracket_types=2, seed=42)
    pad_id = train_ds.pad_id
    vocab_size = train_ds.vocab_size
    print(f"vocab_size={vocab_size}, pad_id={pad_id}, train_size={len(train_ds)}, test_size={len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=lambda b: ex.collate(b, pad_id))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             collate_fn=lambda b: ex.collate(b, pad_id))

    models_factories = {
        'DTFN-U': lambda: dc.make_dtfn_unidirectional(vocab_size, d=64, K=4),
        'DTFN-B': lambda: dc.make_dtfn_bidirectional(vocab_size, d=64, K=4),
        'DTFN-FULL': lambda: dc.make_dtfn_full(vocab_size, d=64, K=4),
        'LSTM': lambda: ex.LSTMModel(vocab_size, d=64, hidden_dim=128),
        'Transformer': lambda: ex.TransformerModel(vocab_size, d=64, n_heads=4, n_layers=2, max_len=256),
    }

    all_results = {}
    for name, factory in models_factories.items():
        ex.set_seed(seed)
        model = factory()
        print(f"\nTraining {name} (params={ex.count_params(model)})")
        tr = ex.train_model(model, train_loader, epochs=epochs, device=device,
                            lr=1e-3, pad_id=pad_id, seed=seed, verbose=True)
        ev = ex.evaluate_model(model, test_loader, device, pad_id=pad_id)
        all_results[name] = {
            'train': tr.__dict__,
            'test': ev,
            'n_params': ex.count_params(model),
        }
        print(f"-> test loss={ev['loss']:.4f}  tk_acc={ev['token_acc']:.4f}  seq_acc={ev['seq_acc']:.4f}")

    with open(output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="/Users/daxu/papers/dtfn/results/dyck2.json")
    args = parser.parse_args()
    run_dyck2(args.epochs, args.device, args.output)


if __name__ == "__main__":
    main()
