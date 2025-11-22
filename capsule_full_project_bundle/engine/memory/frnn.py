"""
Fractional Recurrent Neural Network (FRNN) core.

In this simplified version we model the recurrent core as a standard
GRU network for demonstration purposes.  It accepts sequences of
embedding vectors and produces hidden states and outputs for each
time step.  The `reset_state` method allows the engine to maintain
state across multiple calls for streaming applications.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import FRNNConfig


class FRNNCore(nn.Module):
    """Wrapper around PyTorch GRU for stateful sequence processing."""

    def __init__(self, cfg: FRNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        # Linear layer to map hidden state to output
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (B, T, input_dim).
            h0: Initial hidden state (num_layers, B, hidden_dim), optional.
        Returns:
            outputs: Projected outputs (B, T, output_dim).
            h_n: Final hidden state (num_layers, B, hidden_dim).
        """
        out, h_n = self.gru(x, h0)
        proj = self.output_proj(out)
        return proj, h_n

    def reset_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Initializes a zero hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(
            self.cfg.num_layers, batch_size, self.cfg.hidden_dim, device=device
        )