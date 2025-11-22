"""
Training script for the Capsule Brain engine.

This script demonstrates how to perform a simple supervised training
loop using the CapsuleMemoryEngine on randomly generated data.  It
resets the recurrent state between sequences and updates the engine
parameters via stochastic gradient descent.  To use a real dataset,
replace the dummy data generation with your own loading logic.
"""

import argparse
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from capsule_full_project.engine import CapsuleMemoryEngine, EngineConfig


def generate_dummy_dataset(n_samples: int, obs_dim: int, num_classes: int) -> TensorDataset:
    """Generates a random dataset for classification."""
    x = torch.randn(n_samples, obs_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(x, y)


def train_supervised(
    cfg: EngineConfig,
    n_samples: int = 1000,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    device = torch.device(device)
    engine = CapsuleMemoryEngine(cfg).to(device)
    dataset = generate_dummy_dataset(n_samples, cfg.encoder.d_input, 10)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(engine.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for features, target in loader:
            features = {"obs": features.to(device)}
            target = target.to(device)
            engine.reset_state(batch_size=features["obs"].size(0), device=device)
            out = engine.forward_step(features)
            logits = out["action_logits"]
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")


def main(args: Dict[str, str]) -> None:
    cfg = EngineConfig()
    train_supervised(
        cfg,
        n_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Capsule Memory Engine")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()
    main(args)