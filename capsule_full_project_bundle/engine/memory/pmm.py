"""
Pseudo‑Mode Memory (PMM).

The PMM holds a fixed set of memory slots (or modes) and enables
retrieval via a simple dot‑product attention mechanism.  Upon each
query, the memory updates its modes via exponential moving average
to incorporate new information.
"""

from typing import Dict, Any

import torch
import torch.nn as nn

from ..config import PMMConfig


class StaticPseudoModeMemory(nn.Module):
    """Simple memory bank with exponential decay update."""

    def __init__(self, cfg: PMMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Memory slots: (num_modes, dim)
        self.register_buffer(
            "modes", torch.zeros(cfg.num_modes, cfg.dim), persistent=False
        )
        # Running counts for updates
        self.register_buffer(
            "counts", torch.zeros(cfg.num_modes), persistent=False
        )

    def forward(self, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Performs retrieval for the given batch of queries."""
        # query: (B, dim)
        # compute attention weights: (B, num_modes)
        sims = torch.matmul(query, self.modes.t())  # (B, num_modes)
        weights = torch.softmax(sims, dim=-1)
        # weighted sum: (B, dim)
        value = torch.matmul(weights, self.modes)
        return {"value": value, "weights": weights}

    def update(self, query: torch.Tensor) -> None:
        """Updates the memory modes using the new query via EMA."""
        # Attention chooses the most similar mode for each sample
        sims = torch.matmul(query, self.modes.t())  # (B, num_modes)
        idx = sims.argmax(dim=-1)
        for i in range(query.size(0)):
            j = idx[i].item()
            # Update mode j
            self.counts[j] = self.cfg.decay * self.counts[j] + 1
            lr = 1.0 / self.counts[j]
            self.modes[j] = (1 - lr) * self.modes[j] + lr * query[i]