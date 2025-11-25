"""Global Workspace Engine.

The global workspace is inspired by the Global Workspace Theory of
consciousness【609993358319586†L382-L399】.  It acts as a central bottleneck
through which different subsystems of the Capsule Brain can exchange
information.  In this implementation the workspace is updated by a
gated recurrent unit (GRU) that takes as input a concatenated vector
of the current perceptual embedding, the previous workspace, the
emotion vector and any retrieved memory.  The output of the GRU is
projected back to the workspace dimensionality and fed back on the
next timestep.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..core.config import WorkspaceConfig


class GlobalWorkspaceEngine(nn.Module):
    """Recurrent module that maintains the global workspace vector."""

    def __init__(self, cfg: WorkspaceConfig, input_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        # Project GRU hidden state to workspace dimensionality
        self.proj = nn.Linear(cfg.hidden_dim, cfg.dim)

    def forward(
        self,
        x: torch.Tensor,
        prev_workspace: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the workspace given new inputs.

        Args:
            x: Input features of shape ``(B, input_dim)``.
            prev_workspace: Previous workspace vector of shape ``(B, dim)``.
            hidden: Optional hidden state ``(num_layers, B, hidden_dim)``.
        Returns:
            workspace: Updated workspace of shape ``(B, dim)``.
            hidden_state: Updated hidden state for the GRU.
        """
        # Concatenate input and previous workspace
        inp = torch.cat([x, prev_workspace], dim=-1).unsqueeze(1)  # (B,1,*)
        out, h_n = self.gru(inp, hidden)
        workspace = self.proj(out[:, 0, :])
        return workspace, h_n
