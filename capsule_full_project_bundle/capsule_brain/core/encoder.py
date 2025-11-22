"""Generic encoders for perceptual inputs.

The Capsule Brain does not prescribe a fixed encoder architecture.
Instead this module provides a lightweight MLP encoder suitable for
tabular inputs as a baseline.  For more demanding tasks users can
substitute in vision transformers, convolutional neural networks or
language models depending on the modality.
"""

from typing import Dict, Any

import torch
import torch.nn as nn


class SimpleEncoder(nn.Module):
    """A minimal MLP encoder turning raw feature vectors into a latent state.

    ``SimpleEncoder`` expects a two‑dimensional input ``(B, F)``.  For
    multi‑dimensional perceptual inputs such as images or audio, use
    ``ImageEncoder`` or a custom encoder instead.  The network
    internally pads or truncates the input to match the configured
    dimensionality before applying two linear layers with a ReLU
    activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten everything except batch; if x has more than 2 dims
        # then treat it as a raw tensor and flatten.  This allows the
        # encoder to accept 4‑D images in degenerate cases but users
        # should prefer ``ImageEncoder`` for perceptual inputs.
        x_flat = x.view(x.size(0), -1)
        # Pad or truncate to fixed input dimension
        if x_flat.size(-1) < self.input_dim:
            pad = torch.zeros(x_flat.size(0), self.input_dim - x_flat.size(-1), device=x_flat.device)
            x_flat = torch.cat([x_flat, pad], dim=-1)
        elif x_flat.size(-1) > self.input_dim:
            x_flat = x_flat[:, : self.input_dim]
        return self.net(x_flat)


class ImageEncoder(nn.Module):
    """A simple convolutional encoder for RGB images.

    The ``ImageEncoder`` converts a batch of images of shape
    ``(B, C, H, W)`` into a latent vector of a fixed dimensionality.  A
    stack of convolutional layers reduces the spatial dimensions
    progressively before a linear projection produces the desired
    output dimensionality.  This encoder is suitable for images from
    MineRL/MineDojo (e.g. 3×64×64) and can be swapped for more
    sophisticated architectures as needed.
    """

    def __init__(self, output_dim: int, input_channels: int = 3) -> None:
        super().__init__()
        # Convolutional backbone: [B, C, 64, 64] -> [B, 64, 8, 8]
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Compute the number of flattened features assuming a 64×64 input
        # If other resolutions are used the view operation should be updated
        self.flatten_dim = 64 * 8 * 8  # after conv stack
        self.fc = nn.Linear(self.flatten_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (B, C, H, W)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out
