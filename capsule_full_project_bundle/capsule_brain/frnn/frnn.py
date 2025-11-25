"""Fractional Recurrent Neural Network (FRNN).

An FRNN is a fully recurrent neural network in which every unit is
connected to every other unit.  This general architecture is the most
flexible recurrent topology; other RNNs can be derived by masking
connections【131464440980276†L620-L630】.  For practical reasons we implement
the FRNN using a gated recurrent unit (GRU) followed by a linear
projection to the desired output dimensionality.  The class provides a
``reset_state`` method to initialise hidden states between sequences.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..core.config import FRNNConfig


class FRNNCore(nn.Module):
    """Wrapper around a GRU to provide stateful sequence processing."""

    def __init__(self, cfg: FRNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence and return outputs and final hidden state.

        Args:
            x: Input sequence of shape ``(B, T, input_dim)``.
            h0: Optional initial hidden state of shape
                ``(num_layers, B, hidden_dim)``.  If ``None`` a zero state
                is used.
        Returns:
            outputs: Projected outputs of shape ``(B, T, output_dim)``.
            h_n: Final hidden state of shape ``(num_layers, B, hidden_dim)``.
        """
        out, h_n = self.gru(x, h0)
        proj = self.output_proj(out)
        return proj, h_n

    def reset_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Initialise a zero hidden state for the GRU."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(
            self.cfg.num_layers, batch_size, self.cfg.hidden_dim, device=device
        )
