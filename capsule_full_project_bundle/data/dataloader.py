"""
Simple dataset loader utilities.

This module provides placeholders for loading custom datasets for the
Capsule Memory Engine.  Implement your own functions here to load
datasets and preprocess them into the expected feature format.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


def load_dummy_data(n_samples: int, obs_dim: int) -> DataLoader:
    """Returns a DataLoader of random data for prototyping."""
    x = torch.randn(n_samples, obs_dim)
    y = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)