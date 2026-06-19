import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
import os
import psutil
from pathlib import Path
from lst_model import LSTModel, MinimalMamba
from tasks import get_dataloader

# --- Baselines ---

class PureTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, num_heads, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        # Create causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        logits = self.head(x)
        return logits

class PureMamba(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MinimalMamba(d_model) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            # Simple residual structure for baseline
            res = x
            x = layer(x)
            x = res + x
        x = self.norm_f(x)
        logits = self.head(x)
        return logits

# --- Runner ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    start_time = time.time()

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs) # (B, L, V)

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy on the last token only
        last_logits = outputs[:, -1, :]
        last_targets = targets[:, -1]
        preds = torch.argmax(last_logits, dim=1)
        correct += (preds == last_targets).sum().item()
        total += inputs.size(0)

    end_time = time.time()
    epoch_time = end_time - start_time

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy, epoch_time

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters - Reduced heavily for sandbox
    vocab_size = 50
    d_model = 32
    n_layers = 1 # reduced layers
    num_heads = 4
    num_epochs = 2 # minimum epochs
    batch_size = 8

    # Very short sequences just to prove the pipeline works
    seq_lengths = [32, 64, 128]

    results = []

    # Models to test
    model_types = ['LST', 'Transformer', 'Mamba']

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        train_loader = get_dataloader(vocab_size, seq_len, num_samples=32, batch_size=batch_size)

        for model_name in model_types:
            print(f"Training {model_name}...")

            if model_name == 'LST':
                model = LSTModel(vocab_size, d_model, n_layers, n_latent=8).to(device)
            elif model_name == 'Transformer':
                model = PureTransformer(vocab_size, d_model, n_layers, num_heads).to(device)
            elif model_name == 'Mamba':
                model = PureMamba(vocab_size, d_model, n_layers).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            mem_before = measure_memory()

            epoch_times = []
            accuracies = []

            for epoch in range(num_epochs):
                loss, acc, t = train_one_epoch(model, train_loader, criterion, optimizer, device)
                epoch_times.append(t)
                accuracies.append(acc)

            avg_time = sum(epoch_times) / len(epoch_times)
            final_acc = accuracies[-1]
            mem_after = measure_memory()
            mem_usage = mem_after - mem_before

            print(f"  Result: Acc {final_acc:.2f}, Time {avg_time:.3f}s")

            results.append({
                'model': model_name,
                'seq_len': seq_len,
                'accuracy': final_acc,
                'time_per_epoch': avg_time,
                'memory_usage': mem_usage
            })

    # Save results
    keys = results[0].keys()

    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'experiment_results.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f"\nResults saved to {output_dir / 'experiment_results.csv'}")

    # Additional Experiment: K ablation for LST
    print("\n--- Running K Ablation for LST ---")
    k_values = [4, 8]
    seq_len_fixed = 64
    ablation_results = []

    train_loader = get_dataloader(vocab_size, seq_len_fixed, num_samples=32, batch_size=batch_size)

    for k in k_values:
        print(f"Training LST with K={k}...")
        model = LSTModel(vocab_size, d_model, n_layers, n_latent=k).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            loss, acc, t = train_one_epoch(model, train_loader, criterion, optimizer, device)

        print(f"  Result: Acc {acc:.2f}")
        ablation_results.append({'k': k, 'accuracy': acc})

    with open(output_dir / 'k_ablation_results.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, ['k', 'accuracy'])
        dict_writer.writeheader()
        dict_writer.writerows(ablation_results)
    print("Ablation results saved.")

if __name__ == '__main__':
    run_experiment()
