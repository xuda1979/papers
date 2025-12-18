"""
Neural Flow Dynamics: Main Training Script
============================================

This script provides the complete training pipeline for Neural Flow Dynamics,
including:
- Data generation from quantum systems
- Flow Matching training loop
- Evaluation and benchmarking

Author: Neural Flow Dynamics Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from tqdm import tqdm

# Local imports
from flow_matching import (
    FlowConfig, VelocityNetwork, FlowMatchingLoss, 
    QuantumFidelityLoss, sample_gaussian_complex, normalize_quantum_state
)
from neural_ode import ODEConfig, NeuralODE
from quantum_systems import (
    TransverseFieldIsing, HeisenbergXXZ, FermiHubbard2D,
    random_state, computational_basis_state, entanglement_entropy
)


# ============================================
# Training Configuration
# ============================================

@dataclass
class TrainingConfig:
    """Configuration for training Neural Flow Dynamics."""
    
    # Model
    hidden_dim: int = 512
    num_layers: int = 6
    fourier_features: int = 128
    use_symplectic: bool = True
    
    # ODE Solver
    ode_method: str = "rk4"
    ode_steps_train: int = 20
    ode_steps_eval: int = 100
    
    # Training
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Data
    num_train_samples: int = 10000
    num_eval_samples: int = 1000
    
    # Quantum System
    system_type: str = "tfim"  # "tfim", "xxz", "hubbard"
    num_qubits: int = 8
    time_horizon: float = 5.0
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "outputs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# Dataset
# ============================================

class QuantumDynamicsDataset(Dataset):
    """
    Dataset of quantum state evolution pairs.
    
    Each sample contains:
    - z_0: Initial state (or prior sample)
    - z_1: Target state at time T
    - T: Evolution time
    """
    
    def __init__(
        self,
        system,
        num_samples: int,
        time_horizon: float = 5.0,
        seed: Optional[int] = None
    ):
        self.system = system
        self.num_samples = num_samples
        self.time_horizon = time_horizon
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Generate data
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Dict]:
        """Generate quantum state evolution data."""
        data = []
        dim = self.system.dim
        
        print("Generating quantum dynamics data...")
        for _ in tqdm(range(self.num_samples)):
            # Random initial state
            psi_0 = random_state(int(np.log2(dim)))
            
            # Random evolution time
            T = np.random.uniform(0.1, self.time_horizon)
            
            # Evolve state
            if hasattr(self.system, 'H'):  # PyTorch Hamiltonian
                psi_T = self.system.time_evolution(psi_0, T)
            else:  # NumPy/SciPy Hamiltonian
                psi_0_np = psi_0.numpy()
                psi_T_np = self.system.time_evolution(psi_0_np, T)
                psi_T = torch.from_numpy(psi_T_np)
            
            # Convert to real representation
            z_0 = torch.stack([psi_0.real, psi_0.imag], dim=-1).float()
            z_1 = torch.stack([psi_T.real, psi_T.imag], dim=-1).float()
            
            data.append({
                'z_0': z_0,
                'z_1': z_1,
                'T': torch.tensor([T], dtype=torch.float32)
            })
        
        return data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    return {
        'z_0': torch.stack([b['z_0'] for b in batch]),
        'z_1': torch.stack([b['z_1'] for b in batch]),
        'T': torch.stack([b['T'] for b in batch])
    }


# ============================================
# Neural Flow Dynamics Model
# ============================================

class NeuralFlowDynamics(nn.Module):
    """
    Complete Neural Flow Dynamics model.
    
    Combines:
    - Velocity network for flow matching
    - Neural ODE for integration
    - Conditioning on evolution time T
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Flow config
        flow_config = FlowConfig(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_qubits=config.num_qubits,
            fourier_features=config.fourier_features,
            use_symplectic=config.use_symplectic,
            device=config.device
        )
        
        # Velocity network
        self.velocity_net = VelocityNetwork(flow_config)
        
        # Time conditioning embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # ODE configs (different for training and evaluation)
        self.ode_config_train = ODEConfig(
            method=config.ode_method,
            num_steps=config.ode_steps_train
        )
        self.ode_config_eval = ODEConfig(
            method=config.ode_method,
            num_steps=config.ode_steps_eval
        )
        
        self.to(config.device)
    
    def forward(
        self,
        z_0: torch.Tensor,
        T: Optional[torch.Tensor] = None,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Generate state at time T from initial state z_0.
        
        Args:
            z_0: Initial state (batch, state_dim, 2)
            T: Target time (batch, 1); if None, T=1
            return_trajectory: If True, return full trajectory
        
        Returns:
            Predicted state z_T
        """
        # Use appropriate ODE config
        ode_config = self.ode_config_train if self.training else self.ode_config_eval
        
        # Create conditioned velocity function
        def velocity_fn(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.velocity_net(z, t)
        
        # Create Neural ODE
        ode = NeuralODE(velocity_fn, ode_config)
        
        # Integrate
        z_T = ode(z_0, t_span=(0.0, 1.0), return_trajectory=return_trajectory)
        
        return z_T
    
    def sample(
        self,
        num_samples: int,
        state_dim: int,
        T: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples by flowing from Gaussian prior.
        
        Args:
            num_samples: Number of samples to generate
            state_dim: Dimension of quantum state
            T: Target evolution time
        
        Returns:
            Generated quantum states
        """
        # Sample from prior
        z_0 = sample_gaussian_complex((num_samples, state_dim), device=self.config.device)
        
        # Flow to target
        z_T = self.forward(z_0, T)
        
        # Normalize
        z_T = normalize_quantum_state(z_T)
        
        return z_T


# ============================================
# Training Loop
# ============================================

class Trainer:
    """Training manager for Neural Flow Dynamics."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize quantum system
        self.system = self._create_system()
        
        # Create model
        self.model = NeuralFlowDynamics(config)
        
        # Loss functions
        self.fm_loss = FlowMatchingLoss()
        self.fidelity_loss = QuantumFidelityLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Metrics
        self.train_losses = []
        self.eval_metrics = []
    
    def _create_system(self):
        """Create quantum system based on config."""
        if self.config.system_type == "tfim":
            return TransverseFieldIsing(
                n_sites=self.config.num_qubits,
                J=1.0,
                h=0.5
            )
        elif self.config.system_type == "xxz":
            return HeisenbergXXZ(
                n_sites=self.config.num_qubits,
                J=1.0,
                delta=1.0
            )
        elif self.config.system_type == "hubbard":
            # For Hubbard, num_qubits refers to lattice sites
            L = int(np.sqrt(self.config.num_qubits))
            return FermiHubbard2D(Lx=L, Ly=L, t=1.0, U=4.0)
        else:
            raise ValueError(f"Unknown system type: {self.config.system_type}")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and evaluation dataloaders."""
        train_dataset = QuantumDynamicsDataset(
            self.system,
            num_samples=self.config.num_train_samples,
            time_horizon=self.config.time_horizon,
            seed=42
        )
        
        eval_dataset = QuantumDynamicsDataset(
            self.system,
            num_samples=self.config.num_eval_samples,
            time_horizon=self.config.time_horizon,
            seed=123
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return train_loader, eval_loader
    
    def train_step(self, batch: Dict) -> Dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        z_0 = batch['z_0'].to(self.device)
        z_1 = batch['z_1'].to(self.device)
        T = batch['T'].to(self.device)
        
        # Flow Matching loss
        loss = self.fm_loss(self.model.velocity_net, z_0, z_1)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip
        )
        
        # Update
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict:
        """Evaluate model on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_fidelity = 0.0
        num_batches = 0
        
        for batch in eval_loader:
            z_0 = batch['z_0'].to(self.device)
            z_1 = batch['z_1'].to(self.device)
            
            # Generate prediction
            z_pred = self.model(z_0)
            z_pred = normalize_quantum_state(z_pred)
            
            # Compute fidelity
            fidelity = 1 - self.fidelity_loss(z_pred, z_1).item()
            
            # Flow matching loss
            loss = self.fm_loss(self.model.velocity_net, z_0, z_1).item()
            
            total_loss += loss
            total_fidelity += fidelity
            num_batches += 1
        
        return {
            'eval_loss': total_loss / num_batches,
            'fidelity': total_fidelity / num_batches
        }
    
    def train(self):
        """Full training loop."""
        print(f"Training Neural Flow Dynamics on {self.device}")
        print(f"System: {self.config.system_type}, Qubits: {self.config.num_qubits}")
        print("=" * 60)
        
        # Create dataloaders
        train_loader, eval_loader = self.create_dataloaders()
        
        global_step = 0
        best_fidelity = 0.0
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in pbar:
                # Training step
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                
                # Logging
                if global_step % self.config.log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-100:])
                    pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
                
                # Evaluation
                if global_step % self.config.eval_interval == 0 and global_step > 0:
                    eval_metrics = self.evaluate(eval_loader)
                    self.eval_metrics.append({
                        'step': global_step,
                        **eval_metrics
                    })
                    
                    print(f"\n[Step {global_step}] "
                          f"Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                          f"Fidelity: {eval_metrics['fidelity']:.4f}")
                    
                    # Save best model
                    if eval_metrics['fidelity'] > best_fidelity:
                        best_fidelity = eval_metrics['fidelity']
                        self.save_checkpoint('best_model.pt')
                
                # Save checkpoint
                if global_step % self.config.save_interval == 0 and global_step > 0:
                    self.save_checkpoint(f'checkpoint_{global_step}.pt')
                
                global_step += 1
            
            # End of epoch
            self.train_losses.append(np.mean(epoch_losses))
            self.scheduler.step()
        
        # Final save
        self.save_checkpoint('final_model.pt')
        self.save_metrics()
        
        print("\n" + "=" * 60)
        print(f"Training complete! Best fidelity: {best_fidelity:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics = {
            'train_losses': self.train_losses,
            'eval_metrics': self.eval_metrics,
            'config': {
                k: v for k, v in self.config.__dict__.items()
                if not k.startswith('_')
            }
        }
        
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)


# ============================================
# Main
# ============================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Neural Flow Dynamics')
    parser.add_argument('--system', type=str, default='tfim',
                        choices=['tfim', 'xxz', 'hubbard'])
    parser.add_argument('--qubits', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        system_type=args.system,
        num_qubits=args.qubits,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device if args.device != 'auto' else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    )
    
    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
