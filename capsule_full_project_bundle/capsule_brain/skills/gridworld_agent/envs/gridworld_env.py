"""Toy gridworld environment adapter.

This environment defines a simple deterministic gridworld of size
``N×N``.  The agent starts in the centre of the grid and can move
up, down, left or right.  The observation space matches the format
expected by the Capsule Brain: an RGB image, scalar vector and goal
vector.  For simplicity the ``pixels`` observation is a blank image
and only the scalar coordinates are provided.  Rewards are zero and
the task never terminates (``done`` is always ``False``).
"""

from typing import Dict, Tuple, Any

import numpy as np


class GridworldEnvAdapter:
    """Adapter for a toy gridworld environment."""

    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.reset()

    def reset(self) -> Dict[str, Any]:
        # Start agent at centre
        self.pos = [self.size // 2, self.size // 2]
        return self._convert_obs()

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # Move agent based on discrete action
        if action_idx == 0 and self.pos[0] > 0:
            self.pos[0] -= 1  # up
        elif action_idx == 1 and self.pos[0] < self.size - 1:
            self.pos[0] += 1  # down
        elif action_idx == 2 and self.pos[1] > 0:
            self.pos[1] -= 1  # left
        elif action_idx == 3 and self.pos[1] < self.size - 1:
            self.pos[1] += 1  # right
        # Constant reward and no termination
        obs = self._convert_obs()
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        return obs, reward, done, info

    def _convert_obs(self) -> Dict[str, Any]:
        # Return a blank image and the agent's position as scalars
        pixels = np.zeros((3, 64, 64), dtype=np.float32)
        scalars = np.array(self.pos, dtype=np.float32)  # row, col
        goal = np.zeros_like(scalars)
        return {"pixels": pixels, "scalars": scalars, "goal": goal}
