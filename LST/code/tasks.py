import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class AssociativeRecallTask(Dataset):
    """
    Associative Recall Task:
    Input:  k1 v1 k2 v2 ... query_k
    Target: query_v

    The model must find the key 'query_k' in the history and predict the next token 'query_v'.
    """
    def __init__(self, vocab_size, sequence_length, num_samples):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random sequence of key-value pairs
        # We need pairs, so seq_len must be even (excluding the query)
        # Sequence: k1, v1, k2, v2, ..., kn, vn, query_k
        # Length: 2*n + 1

        # We want the total length to be self.sequence_length.
        # So 2*n + 1 = sequence_length => n = (L-1)/2.
        n_pairs = (self.sequence_length - 1) // 2

        # To avoid trivial learning, keys should be unique in the sequence if possible,
        # or we define the task as retrieving the *most recent* key.
        # Let's assume unique keys for simplicity in generation, or random is fine.

        # Random integers for the sequence
        seq = np.random.randint(0, self.vocab_size, size=2*n_pairs)

        # Pick a random pair index to query
        query_idx = np.random.randint(0, n_pairs)
        key = seq[2*query_idx]
        value = seq[2*query_idx + 1]

        # Construct input: seq + [key]
        input_seq = np.append(seq, key)

        # Target: shift input by 1 (standard AR) or just last token prediction?
        # Usually for AR task in papers, we care about the prediction at the very end.
        # But standard training does next-token prediction on the whole sequence.
        # We will return input and target for next-token prediction.

        # Input:  k1 v1 ... kn vn qk
        # Target: v1 k2 ... vn qk qv
        # The loss that matters is the one at the last position.

        target_seq = np.append(seq[1:], [key, value]) # Shifted by 1

        # If we use strict next-token prediction training, we need:
        # x: [0 ... L-1]
        # y: [1 ... L]
        # In our case, x has length 2*n+1. y has length 2*n+1.

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def get_dataloader(vocab_size=100, sequence_length=128, num_samples=1000, batch_size=32):
    dataset = AssociativeRecallTask(vocab_size, sequence_length, num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
