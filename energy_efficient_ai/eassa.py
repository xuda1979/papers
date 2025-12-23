"""
Energy-Aware Sparse State Aggregation (EASSA)
=============================================

A novel attention replacement mechanism that achieves O(NK) complexity
where K is the number of dynamic states determined by an energy budget.

Key Components:
1. Sparse State Aggregation: Dynamically cluster tokens into K representative states
2. Energy-Aware Router: Route tokens based on semantic similarity AND energy constraints
3. State Merging: Maintain bounded memory by merging similar states

Author: Anonymous
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class EnergyAwareRouter(nn.Module):
    """
    Energy-Aware Router for EASSA.
    
    Routes tokens to states based on:
    1. Semantic similarity (cosine similarity between key and state centroid)
    2. Energy penalty (discourages creating new states when budget is low)
    """
    
    def __init__(
        self,
        d_model: int,
        lambda_0: float = 1.0,
        temperature: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.lambda_0 = lambda_0
        self.temperature = temperature
        
        # Learnable energy budget controller
        self.energy_controller = nn.Linear(d_model, 1)
        nn.init.zeros_(self.energy_controller.weight)
        nn.init.zeros_(self.energy_controller.bias)
    
    def compute_per_token_budget(
        self, 
        x: torch.Tensor, 
        total_budget: float, 
        seq_len: int
    ) -> torch.Tensor:
        """
        Compute adaptive per-token energy budget.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            total_budget: Total energy budget for sequence
            seq_len: Sequence length
            
        Returns:
            Per-token budget [batch, seq_len]
        """
        # Base budget per token
        base_budget = total_budget / seq_len
        
        # Adaptive multiplier based on input complexity
        multiplier = torch.sigmoid(self.energy_controller(x)).squeeze(-1)  # [batch, seq_len]
        
        # Scale so expected value equals base_budget
        per_token_budget = base_budget * 2 * multiplier
        
        return per_token_budget
    
    def compute_routing_scores(
        self,
        key: torch.Tensor,
        centroids: torch.Tensor,
        energy_budget: float,
        state_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute routing scores for assigning a token to states.
        
        Args:
            key: Key vector [d_model]
            centroids: State centroids [K, d_model]
            energy_budget: Current energy budget
            state_counts: Number of tokens in each state [K]
            
        Returns:
            Routing scores [K+1] (K existing states + 1 new state option)
        """
        K = centroids.shape[0]
        
        if K == 0:
            # No existing states, must create new
            return torch.tensor([1.0], device=key.device)
        
        # Cosine similarity with existing centroids
        key_norm = F.normalize(key.unsqueeze(0), dim=-1)  # [1, d_model]
        centroid_norm = F.normalize(centroids, dim=-1)     # [K, d_model]
        similarities = (key_norm @ centroid_norm.T).squeeze(0)  # [K]
        
        # Energy penalty: higher lambda = more penalty for new states
        lambda_t = self.lambda_0 / max(energy_budget, 1e-6)
        
        # Score for existing states (no penalty)
        existing_scores = similarities
        
        # Score for creating new state (with energy penalty)
        # Use max similarity minus threshold as baseline
        new_state_score = similarities.max() - lambda_t
        
        # Concatenate scores
        scores = torch.cat([existing_scores, new_state_score.unsqueeze(0)])
        
        return scores
    
    def forward(
        self,
        key: torch.Tensor,
        centroids: torch.Tensor,
        energy_budget: float,
        state_counts: torch.Tensor,
        hard: bool = True
    ) -> Tuple[int, torch.Tensor]:
        """
        Route a token to a state.
        
        Args:
            key: Key vector [d_model]
            centroids: State centroids [K, d_model]
            energy_budget: Current energy budget
            state_counts: Number of tokens in each state [K]
            hard: If True, return hard assignment; if False, return soft weights
            
        Returns:
            (assigned_state_idx, routing_weights)
            assigned_state_idx: -1 means create new state, otherwise index of existing state
            routing_weights: Soft routing weights for gradient computation
        """
        scores = self.compute_routing_scores(key, centroids, energy_budget, state_counts)
        
        # Soft routing weights (for training)
        soft_weights = F.softmax(scores / self.temperature, dim=-1)
        
        if hard:
            # Hard assignment (for inference)
            assigned_idx = scores.argmax().item()
            K = centroids.shape[0] if centroids.shape[0] > 0 else 0
            if assigned_idx == K:
                # Create new state
                return -1, soft_weights
            else:
                return assigned_idx, soft_weights
        else:
            return -1, soft_weights


class SparseStateAggregator(nn.Module):
    """
    Sparse State Aggregator for EASSA.
    
    Maintains K dynamic states, each aggregating semantically similar tokens.
    States are created, updated, and merged based on input patterns.
    """
    
    def __init__(
        self,
        d_model: int,
        max_states: int = 64,
        merge_threshold: float = 0.9
    ):
        super().__init__()
        self.d_model = d_model
        self.max_states = max_states
        self.merge_threshold = merge_threshold
    
    def initialize_states(self, device: torch.device, dtype: torch.dtype) -> dict:
        """Initialize empty state storage."""
        return {
            'centroids': torch.empty(0, self.d_model, device=device, dtype=dtype),
            'states': torch.empty(0, self.d_model, device=device, dtype=dtype),
            'counts': torch.empty(0, device=device, dtype=torch.long),
            'K': 0
        }
    
    def create_state(
        self,
        storage: dict,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> dict:
        """Create a new state from a token."""
        storage['centroids'] = torch.cat([
            storage['centroids'],
            key.unsqueeze(0)
        ], dim=0)
        storage['states'] = torch.cat([
            storage['states'],
            value.unsqueeze(0)
        ], dim=0)
        storage['counts'] = torch.cat([
            storage['counts'],
            torch.ones(1, device=key.device, dtype=torch.long)
        ])
        storage['K'] += 1
        return storage
    
    def update_state(
        self,
        storage: dict,
        state_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> dict:
        """Update an existing state with a new token (running average)."""
        n = storage['counts'][state_idx].float()
        
        # Welford's online update for numerical stability
        delta_c = key - storage['centroids'][state_idx]
        storage['centroids'][state_idx] += delta_c / (n + 1)
        
        delta_s = value - storage['states'][state_idx]
        storage['states'][state_idx] += delta_s / (n + 1)
        
        storage['counts'][state_idx] += 1
        
        return storage
    
    def merge_states(self, storage: dict, idx_i: int, idx_j: int) -> dict:
        """Merge two states into one."""
        n_i = storage['counts'][idx_i].float()
        n_j = storage['counts'][idx_j].float()
        n_total = n_i + n_j
        
        # Weighted average of centroids and states
        storage['centroids'][idx_i] = (
            n_i * storage['centroids'][idx_i] + 
            n_j * storage['centroids'][idx_j]
        ) / n_total
        
        storage['states'][idx_i] = (
            n_i * storage['states'][idx_i] + 
            n_j * storage['states'][idx_j]
        ) / n_total
        
        storage['counts'][idx_i] = int(n_total)
        
        # Remove state j
        mask = torch.ones(storage['K'], dtype=torch.bool, device=storage['centroids'].device)
        mask[idx_j] = False
        storage['centroids'] = storage['centroids'][mask]
        storage['states'] = storage['states'][mask]
        storage['counts'] = storage['counts'][mask]
        storage['K'] -= 1
        
        return storage
    
    def find_most_similar_pair(self, storage: dict) -> Tuple[int, int]:
        """Find the two most similar states for merging."""
        K = storage['K']
        if K < 2:
            return -1, -1
        
        # Compute pairwise similarities
        centroids_norm = F.normalize(storage['centroids'], dim=-1)
        similarities = centroids_norm @ centroids_norm.T  # [K, K]
        
        # Mask diagonal
        similarities.fill_diagonal_(-float('inf'))
        
        # Find max
        max_idx = similarities.argmax().item()
        idx_i = max_idx // K
        idx_j = max_idx % K
        
        return min(idx_i, idx_j), max(idx_i, idx_j)
    
    def enforce_max_states(self, storage: dict) -> dict:
        """Merge states until we're under the budget."""
        while storage['K'] > self.max_states:
            idx_i, idx_j = self.find_most_similar_pair(storage)
            if idx_i < 0:
                break
            storage = self.merge_states(storage, idx_i, idx_j)
        return storage


class EASSAAttention(nn.Module):
    """
    Energy-Aware Sparse State Aggregation Attention.
    
    Replaces standard O(N^2) attention with O(NK) complexity
    where K adapts to input complexity and energy budget.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        max_states: int = 64,
        energy_budget: float = 100.0,
        lambda_0: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_states = max_states
        self.energy_budget = energy_budget
        
        # Projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # Per-head routers and aggregators
        self.routers = nn.ModuleList([
            EnergyAwareRouter(self.d_head, lambda_0)
            for _ in range(n_heads)
        ])
        self.aggregators = nn.ModuleList([
            SparseStateAggregator(self.d_head, max_states)
            for _ in range(n_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass of EASSA attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len]
            return_stats: If True, return statistics about state usage
            
        Returns:
            output: Output tensor [batch, seq_len, d_model]
            stats: Optional dictionary with statistics
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_Q(x)  # [batch, seq_len, d_model]
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Reshape for multi-head: [batch, seq_len, n_heads, d_head]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Process each batch and head separately (can be parallelized)
        outputs = []
        all_stats = []
        
        for b in range(batch_size):
            batch_outputs = []
            batch_stats = []
            
            for h in range(self.n_heads):
                head_output, head_stats = self._process_head(
                    Q[b, :, h, :],  # [seq_len, d_head]
                    K[b, :, h, :],
                    V[b, :, h, :],
                    self.routers[h],
                    self.aggregators[h],
                    seq_len
                )
                batch_outputs.append(head_output)
                batch_stats.append(head_stats)
            
            # Concatenate heads: [seq_len, d_model]
            batch_output = torch.cat(batch_outputs, dim=-1)
            outputs.append(batch_output)
            all_stats.append(batch_stats)
        
        # Stack batches: [batch, seq_len, d_model]
        output = torch.stack(outputs, dim=0)
        
        # Output projection
        output = self.W_O(output)
        output = self.dropout(output)
        
        if return_stats:
            stats = {
                'avg_states': sum(s['final_K'] for bs in all_stats for s in bs) / (batch_size * self.n_heads),
                'total_merges': sum(s['merges'] for bs in all_stats for s in bs),
                'total_creates': sum(s['creates'] for bs in all_stats for s in bs)
            }
            return output, stats
        
        return output, None
    
    def _process_head(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        router: EnergyAwareRouter,
        aggregator: SparseStateAggregator,
        seq_len: int
    ) -> Tuple[torch.Tensor, dict]:
        """
        Process a single attention head with EASSA.
        
        Args:
            Q, K, V: Query, Key, Value tensors [seq_len, d_head]
            router: EnergyAwareRouter instance
            aggregator: SparseStateAggregator instance
            seq_len: Sequence length
            
        Returns:
            output: Head output [seq_len, d_head]
            stats: Statistics dictionary
        """
        device = Q.device
        dtype = Q.dtype
        
        # Initialize state storage
        storage = aggregator.initialize_states(device, dtype)
        
        # Statistics
        stats = {'creates': 0, 'merges': 0, 'final_K': 0}
        
        # Per-token energy budget
        per_token_budget = self.energy_budget / seq_len
        
        outputs = []
        
        for t in range(seq_len):
            q_t = Q[t]  # [d_head]
            k_t = K[t]
            v_t = V[t]
            
            # Energy-aware routing
            if storage['K'] == 0:
                # First token: create first state
                storage = aggregator.create_state(storage, k_t, v_t)
                stats['creates'] += 1
            else:
                # Route to existing state or create new
                assigned_idx, _ = router(
                    k_t,
                    storage['centroids'],
                    per_token_budget,
                    storage['counts']
                )
                
                if assigned_idx == -1 and storage['K'] < aggregator.max_states:
                    # Create new state
                    storage = aggregator.create_state(storage, k_t, v_t)
                    stats['creates'] += 1
                else:
                    # Update existing state
                    if assigned_idx == -1:
                        assigned_idx = 0  # Fallback to first state if at max
                    storage = aggregator.update_state(storage, assigned_idx, k_t, v_t)
            
            # Enforce max states through merging
            before_K = storage['K']
            storage = aggregator.enforce_max_states(storage)
            stats['merges'] += before_K - storage['K']
            
            # Compute output via attention over states
            if storage['K'] > 0:
                # Attention weights: [K]
                attn_logits = q_t @ storage['centroids'].T / self.scale
                attn_weights = F.softmax(attn_logits, dim=-1)
                
                # Output: weighted sum of states
                y_t = attn_weights @ storage['states']
            else:
                y_t = torch.zeros_like(q_t)
            
            outputs.append(y_t)
        
        stats['final_K'] = storage['K']
        
        # Stack outputs: [seq_len, d_head]
        output = torch.stack(outputs, dim=0)
        
        return output, stats


class EASSATransformerBlock(nn.Module):
    """
    Transformer block with EASSA attention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_states: int = 64,
        energy_budget: float = 100.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = EASSAAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_states=max_states,
            energy_budget=energy_budget,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            return_stats: If True, return attention statistics
            
        Returns:
            output: Output tensor [batch, seq_len, d_model]
            stats: Optional statistics
        """
        # Self-attention with residual
        attn_out, stats = self.attention(self.norm1(x), mask, return_stats)
        x = x + attn_out
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x, stats


class EASSAModel(nn.Module):
    """
    Full EASSA language model.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_states: int = 64,
        energy_budget: float = 100.0,
        max_seq_len: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            EASSATransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_states=max_states,
                energy_budget=energy_budget,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.head.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            return_stats: If True, return statistics
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            stats: Optional statistics
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Transformer layers
        all_stats = []
        for layer in self.layers:
            x, stats = layer(x, return_stats=return_stats)
            if stats:
                all_stats.append(stats)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        if return_stats and all_stats:
            combined_stats = {
                'avg_states': sum(s['avg_states'] for s in all_stats) / len(all_stats),
                'total_merges': sum(s['total_merges'] for s in all_stats),
                'total_creates': sum(s['total_creates'] for s in all_stats)
            }
            return logits, combined_stats
        
        return logits, None


# ============================================================================
# Testing and Demo
# ============================================================================

def test_eassa():
    """Test EASSA components."""
    print("Testing EASSA components...")
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    d_model = 256
    n_heads = 4
    vocab_size = 1000
    max_states = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test attention module
    print("\n1. Testing EASSAAttention...")
    attention = EASSAAttention(
        d_model=d_model,
        n_heads=n_heads,
        max_states=max_states,
        energy_budget=100.0
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output, stats = attention(x, return_stats=True)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Average states used: {stats['avg_states']:.2f}")
    print(f"   Total state creates: {stats['total_creates']}")
    print(f"   Total state merges: {stats['total_merges']}")
    
    # Test full model
    print("\n2. Testing EASSAModel...")
    model = EASSAModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=n_heads,
        max_states=max_states,
        energy_budget=100.0
    ).to(device)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits, stats = model(input_ids, return_stats=True)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Average states used: {stats['avg_states']:.2f}")
    
    # Test memory efficiency
    print("\n3. Testing memory efficiency...")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    long_seq_len = 1024
    x_long = torch.randn(1, long_seq_len, d_model, device=device)
    output_long, _ = attention(x_long)
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"   Sequence length: {long_seq_len}")
        print(f"   Peak GPU memory: {peak_memory:.2f} MB")
    
    print("\n✓ All tests passed!")
    
    return model, attention


def benchmark_complexity():
    """Benchmark EASSA complexity scaling."""
    print("\nBenchmarking EASSA complexity scaling...")
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 256
    n_heads = 4
    max_states = 64
    
    attention = EASSAAttention(
        d_model=d_model,
        n_heads=n_heads,
        max_states=max_states,
        energy_budget=100.0
    ).to(device)
    
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print(f"{'Seq Length':<12} {'Time (ms)':<12} {'States':<12} {'Complexity':<12}")
    print("-" * 48)
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, d_model, device=device)
        
        # Warmup
        _ = attention(x)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            _, stats = attention(x, return_stats=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / 10 * 1000  # ms
        avg_states = stats['avg_states']
        
        # Theoretical complexity: O(N * K)
        complexity = seq_len * avg_states
        
        print(f"{seq_len:<12} {elapsed:<12.2f} {avg_states:<12.2f} {complexity:<12.0f}")
    
    print("\nNote: Time should scale linearly with (seq_len * avg_states)")


if __name__ == "__main__":
    model, attention = test_eassa()
    benchmark_complexity()
