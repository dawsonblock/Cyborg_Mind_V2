"""
Simple global encoder for the Capsule Memory Engine.

This implementation uses a lightweight multilayer perceptron (MLP) to map
arbitrary input feature vectors into a lower‑dimensional global
representation.  The network is intentionally simple to keep the
architecture minimal; users are encouraged to swap this module out for
domain‑specific encoders such as BERT, ViT or audio models.
"""

from typing import Dict, Any

import torch
import torch.nn as nn

from ..config import EncoderConfig


class SimpleGlobalEncoder(nn.Module):
    """A minimal MLP encoder turning raw features into a global state."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.d_input, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.d_global),
        )

    def encode_state(self, features: Dict[str, Any]) -> torch.Tensor:
        """Extracts a single feature vector and encodes it."""
        # For demonstration, we concatenate all numeric features into a flat vector.
        if isinstance(features, torch.Tensor):
            x = features
        else:
            # Flatten dictionary values into a single tensor
            flat = []
            for v in features.values():
                if isinstance(v, torch.Tensor):
                    flat.append(v.view(v.size(0), -1))
                else:
                    flat.append(torch.tensor(v, dtype=torch.float32).view(-1, 1))
            x = torch.cat(flat, dim=-1)
        # Zero pad or truncate to expected input dimension
        if x.size(-1) < self.cfg.d_input:
            pad = torch.zeros(x.size(0), self.cfg.d_input - x.size(-1), device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif x.size(-1) > self.cfg.d_input:
            x = x[:, : self.cfg.d_input]
        return self.net(x)