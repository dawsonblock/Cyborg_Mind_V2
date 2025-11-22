"""Coding skill for the Capsule Brain.

The coding skill is a placeholder illustrating how a language or code
generation module could be plugged into the Capsule Brain.  Given
the refined workspace, it outputs logits over two actions:

1. ``generate`` – produce a code snippet in response to a task or
   instruction.  In a full implementation this would delegate to a
   specialised language model.
2. ``noop`` – take no action.

It also provides a scalar value estimate for reinforcement learning.
"""

from typing import Dict, Any

import torch
import torch.nn as nn

from ...core.config import CapsuleBrainConfig


class Skill(nn.Module):
    """A minimal coding skill."""

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
        # Two actions: generate or noop
        self.action_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, workspace: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.policy(workspace)
        logits = self.action_head(x)
        value = self.value_head(x)
        # Generate a simple code snippet based on the mean of the workspace.
        mean_val = workspace.mean().item()
        if mean_val > 0:
            code = "print('Hello, world!')"
        else:
            code = "# No operation needed"
        return {"action_logits": logits, "value": value, "generated_code": code}
