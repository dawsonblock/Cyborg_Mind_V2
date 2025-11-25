"""Static pseudo‑mode memory.

This module implements a simple content‑addressable memory based on
cosine similarity.  Memory slots are updated via an exponential moving
average (EMA) with a decay factor specified in the configuration.
Each slot stores a high‑dimensional vector which can be retrieved
given a query.  The retrieval returns both the value and the
attention weights over the modes.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import PMMConfig


class StaticPseudoModeMemory(nn.Module):
    """Simple memory bank with exponential decay update.

    The static PMM maintains a fixed number of slots and updates them
    using a simple exponential moving average when a new query is
    written.  Retrieval is performed by computing cosine similarity
    between the query and memory keys and returning a weighted sum of
    values.
    """

    def __init__(self, cfg: PMMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Memory values: (num_modes, dim)
        self.register_buffer(
            "modes", torch.zeros(cfg.num_modes, cfg.dim), persistent=False
        )
        # Running counts for updates
        self.register_buffer(
            "counts", torch.zeros(cfg.num_modes), persistent=False
        )

    def forward(self, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Retrieve a weighted sum of memory values based on cosine similarity.

        Args:
            query: Query vectors of shape ``(B, dim)``.
        Returns:
            A dictionary with keys ``value`` (tensor of shape ``(B, dim)``)
            and ``weights`` (tensor of shape ``(B, num_modes)``) containing
            the attention distribution over slots.
        """
        # Normalise for cosine similarity
        q_norm = F.normalize(query, dim=-1)
        m_norm = F.normalize(self.modes, dim=-1)
        sims = torch.matmul(q_norm, m_norm.t())  # (B, num_modes)
        weights = torch.softmax(sims, dim=-1)
        value = torch.matmul(weights, self.modes)
        return {"value": value, "weights": weights}

    def update(self, query: torch.Tensor) -> None:
        """Write query vectors into the memory via EMA.

        The most similar slot for each query is selected and updated.
        Each slot tracks the number of times it has been updated to
        compute the effective learning rate.  Older values decay
        exponentially according to ``cfg.decay``.
        """
        # Compute cosine similarity
        q_norm = F.normalize(query, dim=-1)
        m_norm = F.normalize(self.modes, dim=-1) + 1e-8  # avoid zero division
        sims = torch.matmul(q_norm, m_norm.t())  # (B, num_modes)
        idx = sims.argmax(dim=-1)
        for i in range(query.size(0)):
            j = idx[i].item()
            # Update running counts with decay
            self.counts[j] = self.cfg.decay * self.counts[j] + 1
            lr = 1.0 / (self.counts[j] + 1e-8)
            self.modes[j] = (1 - lr) * self.modes[j] + lr * query[i]
