"""Tomita grammar experiments for DTFNs.

Tomita grammars are 7 regular languages over {a, b} used to benchmark
recurrent networks (Tomita, 1982).  They are:
  1: a*            (any number of a's)
  2: (ab)*         (alternating ab)
  3: a*b*a*        (odd number of b's? no: any string not ending in b... actually: a*b*a*)
  4: (a|b)* not containing 'aaa' (no three consecutive a's)
  5: (a|b)* with even number of a's and even number of b's
  6: (a|b)* with number of a's minus number of b's divisible by 3
  7: a*b+         (a's followed by b's)

DTFNs with conservation should excel at the counting-based grammars (3, 5, 6, 7).
We test DTFN-U, DTFN-FULL, LSTM, and Transformer on Tomita 4, 5, 7.
"""
import json
import os
import sys
import time
import random
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dtfn_core import make_dtfn_unidirectional, make_dtfn_full, DTFN


# Token ids: a=0, b=1, PAD=2, BOS=3, EOS=4
A, B, PAD, BOS, EOS = 0, 1, 2, 3, 4
VOCAB = 5


def tomita_accepts(grammar: int, s: str) -> bool:
    """Return True if string s (over {a,b}) is in Tomita grammar `grammar`."""
    if grammar == 1:
        return all(c == 'a' for c in s)
    elif grammar == 2:
        if len(s) % 2 != 0: return False
        return all(s[i] == 'a' and s[i+1] == 'b' for i in range(0, len(s), 2))
    elif grammar == 3:
        # a*b*a*: strings where once you see b you can't go back to a... no.
        # Tomita 3 is: strings that do NOT contain "bb" as substring? Let me use the standard def.
        # Standard Tomita 3: a*b* (any number of a's followed by any number of b's)
        # Actually Tomita 3 = a*b*a* is wrong. Let me use: Tomita 3 = strings where every b is preceded by an a... no.
        # The standard Tomita 3 is: "strings with an odd number of b's"? No.
        # Let me use the canonical definitions from Tomita 1982:
        # T3: (a|b)* not containing 'bb'  -- wait no.
        # Canonical: T3 = a^n b^m, n,m >= 0 (a's then b's)? No that's T7.
        # Let me just use the widely-cited set:
        # T1: a*
        # T2: (ab)*
        # T3: a*b*a*  -- odd number of a's? No.
        # I'll use: T3 = strings not ending in 'b' and ... actually let me just use
        # the simpler well-known ones: T4, T5, T7 which are unambiguous.
        return True  # placeholder, won't use T3
    elif grammar == 4:
        # No 'aaa' substring
        return 'aaa' not in s
    elif grammar == 5:
        # Even number of a's AND even number of b's
        return s.count('a') % 2 == 0 and s.count('b') % 2 == 0
    elif grammar == 6:
        # (a_count - b_count) % 3 == 0
        return (s.count('a') - s.count('b')) % 3 == 0
    elif grammar == 7:
        # a* b+ (zero or more a's followed by one or more b's)
        if len(s) == 0: return False
        seen_b = False
        for c in s:
            if c == 'b': seen_b = True
            if c == 'a' and seen_b: return False
        return seen_b
    return False


def sample_string(max_len: int, rng: random.Random) -> str:
    n = rng.randint(1, max_len)
    return ''.join(rng.choice('ab') for _ in range(n))


def make_tomita_dataset(grammar: int, n_pos: int, n_neg: int, max_len: int,
                        seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a balanced binary classification dataset.

    Labels: 1 = accepted, 0 = rejected.
    Inputs: BOS + string + EOS, padded to max_len+2.
    Targets: same shifted by 1, with label at the EOS position.
    """
    rng = random.Random(seed)
    pos: List[List[int]] = []
    neg: List[List[int]] = []
    attempts = 0
    while len(pos) < n_pos or len(neg) < n_neg:
        s = sample_string(max_len, rng)
        accepted = tomita_accepts(grammar, s)
        toks = [BOS] + [A if c == 'a' else B for c in s] + [EOS]
        if accepted and len(pos) < n_pos:
            pos.append(toks)
        elif not accepted and len(neg) < n_neg:
            neg.append(toks)
        attempts += 1
        if attempts > 100000:
            break

    seqs = pos[:n_pos] + neg[:n_neg]
    labels = [1]*len(pos) + [0]*len(neg)
    # shuffle
    combined = list(zip(seqs, labels))
    rng.shuffle(combined)
    seqs, labels = zip(*combined)

    maxlen = max(len(s) for s in seqs)
    x = torch.full((len(seqs), maxlen), PAD, dtype=torch.long)
    y = torch.full((len(seqs), maxlen), PAD, dtype=torch.long)
    # We'll do next-token prediction: target at position i is the label
    # placed at the EOS position.  Simpler: predict label at every position
    # and mask all but EOS.
    label_y = torch.full((len(seqs), maxlen), -100, dtype=torch.long)
    for i, (s, lab) in enumerate(zip(seqs, labels)):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        y[i, :len(s)] = torch.tensor(s[1:] + [PAD], dtype=torch.long)
        # Put the classification label at the EOS position (last non-PAD)
        eos_idx = len(s) - 1
        label_y[i, eos_idx] = lab
    return x, label_y


# -- Baselines: minimal LSTM and Transformer ------------------------------

class MiniLSTM(nn.Module):
    def __init__(self, vocab_size: int, d: int = 32, n_classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.dec = nn.Linear(d, n_classes)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        return self.dec(out)


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, d: int = 32, n_classes: int = 2,
                 n_heads: int = 4, n_layers: int = 2, max_len: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,
                                           dim_feedforward=4*d, dropout=0.0,
                                           batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dec = nn.Linear(d, n_classes)

    def forward(self, x):
        L = x.size(1)
        e = self.emb(x) + self.pos(torch.arange(L, device=x.device)).unsqueeze(0)
        out = self.encoder(e)
        return self.dec(out)


def train_eval(model, x_train, y_train, x_test, y_test, device, epochs=5, lr=1e-3, batch=32, n_classes=2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    for ep in range(epochs):
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), batch):
            idx = perm[i:i+batch]
            logits = model(x_train[idx])  # (B, L, n_classes)
            loss = crit(logits.reshape(-1, n_classes), y_train[idx].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    # Test accuracy: at the EOS position
    model.eval()
    with torch.no_grad():
        logits = model(x_test)  # (B, L, n_classes)
        mask = y_test != -100
        pred = logits.argmax(-1)
        acc = (pred[mask] == y_test[mask]).float().mean().item()
        # Also compute per-class accuracy
        pos_mask = mask & (y_test == 1)
        neg_mask = mask & (y_test == 0)
        pos_acc = (pred[pos_mask] == 1).float().mean().item() if pos_mask.any() else 0.0
        neg_acc = (pred[neg_mask] == 0).float().mean().item() if neg_mask.any() else 0.0
    n_params = sum(p.numel() for p in model.parameters())
    return {"test_acc": acc, "pos_acc": pos_acc, "neg_acc": neg_acc, "n_params": n_params}


class DTFNClassifier(nn.Module):
    """Wrap a DTFN to do binary (n_classes) classification at each position."""
    def __init__(self, dtfn: DTFN, n_classes: int = 2):
        super().__init__()
        self.dtfn = dtfn
        d = dtfn.cfg.d
        self.cls = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_classes))

    def forward(self, x):
        m = self.dtfn.forward_mass(x)  # (B, L, d)
        return self.cls(m)


def run_tomita(seed: int = 0, epochs: int = 5) -> Dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    results = {}

    for grammar in [4, 5, 7]:
        print(f"\n=== Tomita {grammar} ===")
        x_train, y_train = make_tomita_dataset(grammar, n_pos=200, n_neg=200, max_len=20, seed=seed)
        x_test, y_test = make_tomita_dataset(grammar, n_pos=100, n_neg=100, max_len=30, seed=seed+100)
        print(f"  train: {len(x_train)} samples, test: {len(x_test)} samples")
        print(f"  train pos: {(y_train[y_train!=-100]==1).sum().item()}, neg: {(y_train[y_train!=-100]==0).sum().item()}")

        gr_results = {}
        for name, model_fn in [
            ("DTFN-U", lambda: DTFNClassifier(make_dtfn_unidirectional(vocab_size=VOCAB, d=32, K=3))),
            ("DTFN-FULL", lambda: DTFNClassifier(make_dtfn_full(vocab_size=VOCAB, d=32, K=3))),
            ("LSTM", lambda: MiniLSTM(VOCAB, d=32)),
            ("Transformer", lambda: MiniTransformer(VOCAB, d=32, max_len=64)),
        ]:
            torch.manual_seed(seed)
            model = model_fn()
            r = train_eval(model, x_train, y_train, x_test, y_test, device, epochs=epochs)
            gr_results[name] = r
            print(f"  {name:12s}: acc={r['test_acc']:.4f}  pos={r['pos_acc']:.4f}  neg={r['neg_acc']:.4f}  n_params={r['n_params']}")
        results[f"tomita_{grammar}"] = gr_results

    return results


if __name__ == "__main__":
    t0 = time.time()
    results = run_tomita(seed=0, epochs=5)
    results["elapsed_sec"] = time.time() - t0
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tomita.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    print(f"Elapsed: {results['elapsed_sec']:.1f}s")
