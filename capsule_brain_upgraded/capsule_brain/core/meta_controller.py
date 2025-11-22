"""
Meta controller for multi‑skill selection.

This module implements a simple meta‑controller that examines the
workspace representation along with optional context inputs (such
as goals or memory pressure) and outputs a probability distribution
over available skills.  The meta‑controller can be trained to
select the appropriate skill based on high‑level cues, enabling
multi‑task agents that can seamlessly switch between skills.

For now the implementation is intentionally simple: a two‑layer
feed‑forward network producing logits for each skill.  It can be
replaced with a more sophisticated architecture (e.g. attention
mechanisms or reinforcement learning policies) without changing
the rest of the codebase.
"""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaController(nn.Module):
    """Simple meta controller to choose among multiple skills.

    The meta controller takes as input the current workspace vector
    and optional context (goal vector, memory pressure, etc.) and
    outputs a set of logits corresponding to each available skill.
    A higher logit indicates a higher preference for that skill.
    """

    def __init__(self, workspace_dim: int, num_skills: int, context_dim: int = 0,
                 hidden_dim: int = 128) -> None:
        """
        Args:
            workspace_dim: Dimension of the workspace vector produced by the
                Global Workspace Engine.
            num_skills: Number of skills to choose from.
            context_dim: Dimension of optional context vector to concatenate.
            hidden_dim: Hidden dimension used inside the MLP.
        """
        super().__init__()
        self.num_skills = num_skills
        self.input_dim = workspace_dim + context_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills),
        )

    def forward(self, workspace: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Produce skill logits from workspace and optional context.

        Args:
            workspace: Tensor of shape [B, workspace_dim] representing the
                current workspace state.
            context: Optional tensor of shape [B, context_dim] containing
                additional context features (e.g. goals, memory usage).  If
                provided, it will be concatenated with the workspace.

        Returns:
            Logits tensor of shape [B, num_skills].  Higher logits
                correspond to a higher preference for the skill.
        """
        if context is not None:
            x = torch.cat([workspace, context], dim=-1)
        else:
            x = workspace
        logits = self.mlp(x)
        return logits
