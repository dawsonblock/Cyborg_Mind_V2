"""Minecraft skill module.

This skill wraps an actor–critic network trained to play
Minecraft‑like environments.  It consumes the global workspace
representation and outputs logits over discrete actions along with a
state value estimate.  During deployment the skill is invoked by the
Capsule Brain via the ``ActionCapsule``.
"""

from typing import Dict

import torch
import torch.nn as nn

from ...core.config import CapsuleBrainConfig


class Skill(nn.Module):
    """Simple actor–critic head for Minecraft."""

    def __init__(self, cfg: CapsuleBrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.workspace.dim
        hidden_dim = 256
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, 20)  # Example discrete actions
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, workspace: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.policy(workspace)
        logits = self.action_head(x)
        value = self.value_head(x)
        return {"action_logits": logits, "value": value}
