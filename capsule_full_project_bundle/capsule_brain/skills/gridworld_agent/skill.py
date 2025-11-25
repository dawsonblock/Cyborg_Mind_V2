"""Gridworld skill for the Capsule Brain.

This skill provides a minimal actor‑critic head for a toy gridworld
environment.  The agent receives the refined workspace vector and
outputs logits over four actions corresponding to moving in the four
cardinal directions.  A value estimate is also produced.
"""

from typing import Dict

import torch
import torch.nn as nn

from ...core.config import CapsuleBrainConfig


class Skill(nn.Module):
    """A gridworld navigation skill.

    The skill implements a two‑layer MLP policy and value network.
    Output actions correspond to (0=up, 1=down, 2=left, 3=right).
    """

    def __init__(self, cfg: CapsuleBrainConfig) -> None:
        super().__init__()
        input_dim = cfg.workspace.dim
        hidden_dim = 256
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, 4)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, workspace: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.policy(workspace)
        logits = self.action_head(x)
        value = self.value_head(x)
        return {"action_logits": logits, "value": value}
