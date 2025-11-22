"""
PPO‑based capacity controller.

This module defines a simple actor‑critic network that outputs a discrete
distribution over cognitive capacities and a scalar value estimate.  It is
not a full PPO training implementation; rather it is designed to
integrate into the Capsule Memory Engine by providing the ability to
sample capacities based on the current state vector.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PPOConfig


class CapacityActorCritic(nn.Module):
    """Simple actor‑critic network for selecting discrete capacities."""

    def __init__(self, state_dim: int, cfg: PPOConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.actor = nn.Sequential(
            nn.Linear(state_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.actor(state)
        value = self.critic(state).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs, "value": value}

    def act(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Samples an action (capacity index) given the state vector and
        returns the action index along with the distribution and value.
        """
        out = self.forward(state)
        dist = torch.distributions.Categorical(probs=out["probs"])
        action = dist.sample()
        # Map discrete actions to meaningful capacities; e.g. [4, 6, 9, 12]
        capacities = torch.tensor([4, 6, 9, 12], device=state.device)
        selected = capacities[action]
        return {
            "capacity": selected,
            "action": action,
            "log_prob": dist.log_prob(action),
            "value": out["value"],
        }