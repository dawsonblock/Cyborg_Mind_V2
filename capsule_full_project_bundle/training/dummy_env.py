"""
Dummy environment for training.

This environment produces random feature vectors and returns a scalar
reward based on how well the engine's action logits match a randomly
selected target.  It is designed solely for testing the integration of
the CapsuleMemoryEngine with reinforcement learning algorithms.
"""

from typing import Dict, Any

import torch


class DummyEnv:
    """Simple environment producing random observations and rewards."""

    def __init__(self, obs_dim: int = 128, num_actions: int = 10) -> None:
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.target_action = torch.randint(0, num_actions, (1,), dtype=torch.long)

    def reset(self) -> Dict[str, Any]:
        """Resets the environment and returns an initial observation."""
        self.target_action = torch.randint(0, self.num_actions, (1,), dtype=torch.long)
        obs = torch.randn(1, self.obs_dim)
        return {"obs": obs}

    def step(self, action_logits: torch.Tensor) -> (Dict[str, Any], float):
        """Takes a step given action logits and returns next obs and reward."""
        # Determine predicted action
        pred_action = action_logits.argmax(dim=-1)
        reward = 1.0 if pred_action.item() == self.target_action.item() else -0.1
        # Next observation
        obs = torch.randn(1, self.obs_dim)
        # With small probability change the target action
        if torch.rand(1).item() < 0.1:
            self.target_action = torch.randint(0, self.num_actions, (1,), dtype=torch.long)
        return {"obs": obs}, reward