"""Dynamic pseudo‑mode memory with automatic expansion.

This module provides a GPU‑friendly dynamic memory inspired by the
fallback implementation used in ``brain_cyborg_mind``.  The memory
stores key/value pairs and computes attention using cosine similarity.
Additional metrics such as attention entropy and write density are
tracked to estimate memory pressure; when pressure exceeds a
threshold the number of slots doubles (up to a maximum specified in
the configuration).  Periodic garbage collection resets unused slots.
"""

from typing import Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import PMMConfig


class DynamicPseudoModeMemory(nn.Module):
    """Content‑addressable memory with dynamic expansion and GC.

    Each memory slot has an associated key and value vector.  Retrieval
    computes cosine similarity between the query and all keys and
    returns a weighted sum of values.  Update writes new key/value
    pairs into the least used slots and records write density.  The
    ``expand`` method doubles the number of slots while preserving
    existing contents when the estimated memory pressure is high.
    """

    def __init__(self, cfg: PMMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mem_slots = cfg.num_modes
        self.max_slots = cfg.max_modes or cfg.num_modes
        self.key_dim = cfg.dim
        self.value_dim = cfg.dim
        # Trainable key/value tensors
        self.keys = nn.Parameter(torch.randn(self.mem_slots, self.key_dim) * 0.1)
        self.values = nn.Parameter(torch.randn(self.mem_slots, self.value_dim) * 0.1)
        # Non‑trainable buffers
        self.register_buffer("usage", torch.zeros(self.mem_slots))
        self.register_buffer("write_count", torch.zeros(self.mem_slots))
        # Rolling history for pressure metrics
        self.register_buffer("attn_history", torch.zeros(100))
        self.register_buffer("write_history", torch.zeros(100))
        self.history_idx = 0
        # GC parameters
        self.gc_interval = 1000
        self.gc_threshold = 0.05
        self._step_counter = 0
        self._last_write_mask: Optional[torch.Tensor] = None

    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve values from memory and update pressure metrics.

        Args:
            query: Query vectors of shape ``(B, key_dim)``.
        Returns:
            A tuple ``(readout, weights)`` where ``readout`` has shape
            ``(B, value_dim)`` and ``weights`` has shape ``(B, mem_slots)``.
        """
        self._step_counter += 1
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(self.keys, dim=-1)
        sims = torch.matmul(q_norm, k_norm.t())  # [B, mem_slots]
        attn = F.softmax(sims, dim=-1)
        readout = torch.matmul(attn, self.values)
        # Update usage: decay and increment by current attention
        with torch.no_grad():
            self.usage.mul_(0.99).add_(attn.mean(dim=0) * 0.01)
            # Attention entropy
            probs = attn.clamp_min(1e-8)
            log_probs = probs.log()
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            max_entropy = math.log(self.mem_slots)
            ent_norm = float((entropy / max_entropy).clamp(0.0, 1.0))
            # Write density from last write
            write_density = 0.0
            if self._last_write_mask is not None:
                write_density = float(self._last_write_mask.float().mean())
            # Roll history
            self.attn_history[self.history_idx] = ent_norm
            self.write_history[self.history_idx] = write_density
            self.history_idx = (self.history_idx + 1) % self.attn_history.numel()
            # Trigger garbage collection periodically
            if self._step_counter % self.gc_interval == 0:
                self._garbage_collect()
        return readout, attn

    def update(self, query: torch.Tensor, value: torch.Tensor) -> None:
        """Write new key/value pairs into the least used slots.

        Args:
            query: Key vectors of shape ``(B, key_dim)``.
            value: Value vectors of shape ``(B, value_dim)``.
        """
        B = query.size(0)
        # Determine slots to write based on lowest usage
        _, indices = torch.topk(self.usage, k=B, largest=False)
        write_mask = torch.zeros_like(self.usage, dtype=torch.bool)
        write_mask[indices] = True
        with torch.no_grad():
            self._last_write_mask = write_mask.clone()
            # Expand memory if batch size exceeds current slot count or
            # if the estimated pressure is above a threshold and there
            # is room to expand.
            if B > self.mem_slots or (self.mem_slots < self.max_slots and self.compute_pressure() > 0.8):
                self.expand()
            for i, slot in enumerate(indices):
                j = slot.item()
                # If we expanded the memory the ``indices`` computed
                # earlier may be stale; clamp index to valid range.
                j = int(min(j, self.mem_slots - 1))
                self.keys.data[j] = query[i]
                self.values.data[j] = value[i]
                self.usage[j] = 1.0
                self.write_count[j] += 1

    def compute_pressure(self) -> float:
        """Compute a composite memory pressure metric.

        Pressure increases when attention entropy is low (queries are
        repetitive) or when write density is high (many writes per step).
        The metric is averaged over the rolling histories.
        """
        attn_ent = self.attn_history.mean().item()
        write_den = self.write_history.mean().item()
        # Combine entropic (low entropy means high pressure) and write
        # density contributions equally.
        pressure = (1.0 - attn_ent) * 0.5 + write_den * 0.5
        return pressure

    def expand(self) -> bool:
        """Double the number of memory slots if capacity allows.

        Returns ``True`` if expansion occurred, ``False`` otherwise.
        """
        if self.mem_slots >= self.max_slots:
            return False
        new_slots = min(self.mem_slots * 2, self.max_slots)
        # Allocate new key/value tensors
        new_keys = torch.randn(new_slots, self.key_dim, device=self.keys.device) * 0.1
        new_values = torch.randn(new_slots, self.value_dim, device=self.values.device) * 0.1
        new_usage = torch.zeros(new_slots, device=self.usage.device)
        new_write_count = torch.zeros(new_slots, device=self.write_count.device)
        # Copy existing contents
        new_keys[: self.mem_slots] = self.keys.data
        new_values[: self.mem_slots] = self.values.data
        new_usage[: self.mem_slots] = self.usage
        new_write_count[: self.mem_slots] = self.write_count
        # Replace parameters and buffers
        with torch.no_grad():
            self.keys = nn.Parameter(new_keys)
            self.values = nn.Parameter(new_values)
            self.register_buffer("usage", new_usage)
            self.register_buffer("write_count", new_write_count)
        self.mem_slots = new_slots
        return True

    def _garbage_collect(self) -> None:
        """Reset rarely used memory slots to free capacity."""
        # Mark slots with usage below threshold as free
        mask = self.usage < self.gc_threshold
        if mask.any():
            with torch.no_grad():
                idx = torch.nonzero(mask).squeeze(1)
                self.keys.data[idx] = torch.randn_like(self.keys.data[idx]) * 0.1
                self.values.data[idx] = torch.randn_like(self.values.data[idx]) * 0.1
                self.usage[idx] = 0.0
                self.write_count[idx] = 0.0
