"""Adapter for MineDojo environments.

MineDojo is a research platform built on top of Minecraft for
open‑ended embodied agents.  This adapter normalises its complex
observations into the format expected by the Capsule Brain.  The
implementation is intentionally high level; users should customise
the conversion to suit their tasks.
"""

from typing import Dict, Tuple, Any

import numpy as np

try:
    import minedojo
except ImportError:
    minedojo = None


class MineDojoEnvAdapter:
    """Adapter for MineDojo environments."""

    def __init__(self, task_id: str = "house-building-v0") -> None:
        if minedojo is None:
            raise RuntimeError("minedojo is not installed. Please install MineDojo to use this adapter.")
        self.env = minedojo.make(task_id)

    def reset(self) -> Dict[str, Any]:
        obs = self.env.reset()
        return self._convert_obs(obs)

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return self._convert_obs(obs), reward, done, info

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        pixels = obs["rgb"].astype(np.float32) / 255.0  # assume key exists
        pixels = pixels.transpose(2, 0, 1)
        scalars = np.array([], dtype=np.float32)
        goal = np.array([], dtype=np.float32)
        return {"pixels": pixels, "scalars": scalars, "goal": goal}
