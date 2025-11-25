"""
Training script for ToneNet.

This script provides a command line interface to train the ToneNet
audio processing module on a simple dataset.  The dataset is assumed
to consist of raw audio files in WAV format along with target
transcriptions or labels which are mapped to glyph IDs.  For the
purposes of this reference implementation, we provide a synthetic
dataset that generates random waveforms and assigns random target
glyph sequences.

Usage:

    python -m capsule_brain.tonenet.train_tonenet \
        --data_dir /path/to/audio_dataset \
        --epochs 5 \
        --batch_size 8 \
        --lr 1e-4 \
        --checkpoint checkpoints/tonenet.pt

For realistic training you should replace ``SyntheticToneDataset`` with
a dataset that loads real audio and transcripts.  ToneNet expects
raw audio sampled at the rate specified in ``ToneNetConfig``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .tonenet import ToneNet, ToneNetConfig


class SyntheticToneDataset(Dataset):
    """A simple dataset that generates random audio and labels.

    Each item consists of a random waveform of length ``sample_len`` and
    a random glyph index between 0 and ``vocab_size - 1``.  This is
    intended only for demonstration and testing.  Replace with a
    real dataset for meaningful training.
    """

    def __init__(self, num_samples: int, sample_len: int, vocab_size: int) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate a random waveform and random target glyph index
        wave = torch.randn(self.sample_len)
        target = torch.randint(0, self.vocab_size, (1,), dtype=torch.long)
        return wave, target


def train_tonenet(
    data_loader: DataLoader,
    tonenet: ToneNet,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint: Path | None = None,
) -> None:
    """Train ToneNet on the provided data loader.

    Args:
        data_loader: ``DataLoader`` yielding (waveform, target) pairs.
        tonenet: The ``ToneNet`` model to train.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        device: Device on which to run training.
        checkpoint: Optional path to save the trained model.
    """
    tonenet.to(device)
    optimizer = torch.optim.Adam(tonenet.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    tonenet.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for waves, targets in data_loader:
            waves = waves.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = tonenet(waves)  # [B, T, vocab_size]
            # Average over time dimension: treat as sequence classification
            logits_mean = logits.mean(dim=1)  # [B, vocab_size]
            loss = criterion(logits_mean, targets.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * waves.size(0)
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tonenet.state_dict(), checkpoint)
        print(f"Saved ToneNet to {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ToneNet")
    parser.add_argument("--data_dir", type=str, default="", help="Directory with audio data (unused in synthetic mode)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic samples to generate")
    parser.add_argument("--sample_len", type=int, default=16000, help="Length of each audio sample")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to save the trained model")
    args = parser.parse_args()
    # Initialize dataset (synthetic for now)
    cfg = ToneNetConfig()
    dataset = SyntheticToneDataset(
        num_samples=args.num_samples,
        sample_len=args.sample_len,
        vocab_size=cfg.n_mels,
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    tonenet = ToneNet(cfg)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    train_tonenet(data_loader, tonenet, args.epochs, args.lr, checkpoint=checkpoint_path)


if __name__ == "__main__":
    main()
