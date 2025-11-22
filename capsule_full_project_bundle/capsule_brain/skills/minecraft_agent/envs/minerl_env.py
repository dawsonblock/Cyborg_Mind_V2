"""Wrapper around MineRL environments.

The MineRL environment provides a simple interface to interactive
Minecraft tasks.  This adapter normalises observations into a format
expected by the Capsule Brain: an RGB image, a scalar vector and a
goal vector.  Additional fields from the native MineRL observation
space can be added as required.
"""

from typing import Dict, Tuple, Any

import numpy as np

try:
    import minerl
except ImportError:
    minerl = None


class MineRLEnvAdapter:
    """Adapter for MineRL environments."""

    def __init__(self, env_name: str = "MineRLTreechop-v0") -> None:
        if minerl is None:
            raise RuntimeError("minerl is not installed. Please install MineRL to use this adapter.")
        self.env = minerl.make(env_name)
        # Define a discrete action mapping.  Each entry corresponds to an
        # action index used by the Capsule Brain skill.  These are
        # deliberately simple and may need to be tailored to your task.
        # See MineRL documentation for the full action space.  Unused
        # fields default to 0 or None.
        self.action_map = [
            {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0.0, 0.0]},
            {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0.0, 0.0]},
            {"forward": 0, "back": 1, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0.0, 0.0]},
            {"forward": 0, "back": 0, "left": 1, "right": 0, "jump": 0, "attack": 0, "camera": [0.0, 0.0]},
            {"forward": 0, "back": 0, "left": 0, "right": 1, "jump": 0, "attack": 0, "camera": [0.0, 0.0]},
            {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 1, "attack": 0, "camera": [0.0, 0.0]},
            {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 1, "camera": [0.0, 0.0]},
            {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 1, "camera": [0.0, 0.0]},
            {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 1, "attack": 1, "camera": [0.0, 0.0]},
            {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [5.0, 0.0]},
            {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [-5.0, 0.0]},
            {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0.0, 5.0]},
            {"forward": 1, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0.0, -5.0]},
            {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0.0, 5.0]},
            {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 0, "camera": [0.0, -5.0]},
            {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 1, "camera": [5.0, 0.0]},
            {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 1, "camera": [-5.0, 0.0]},
            {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 1, "camera": [0.0, 5.0]},
            {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0, "attack": 1, "camera": [0.0, -5.0]},
        ]

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        return self._convert_obs(obs)

    def step(self, action_idx: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Perform one step in the MineRL environment using a discrete
        action index.  The integer ``action_idx`` is mapped to a
        high‑dimensional action dictionary via ``self.action_map``.
        """
        # Clip the index to the valid range
        idx = int(max(0, min(action_idx, len(self.action_map) - 1)))
        action_dict = self.action_map[idx]
        obs, reward, done, info = self.env.step(action_dict)
        return self._convert_obs(obs), reward, done, info

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # Normalise image to range [0,1]
        pixels = obs["pov"].astype(np.float32) / 255.0  # (64,64,3)
        pixels = pixels.transpose(2, 0, 1)  # (3,64,64)
        # Flatten some scalar features; here we extract inventory counts if present
        scalars = []
        if "inventory" in obs:
            inv = obs["inventory"]
            for key in sorted(inv.keys()):
                scalars.append(float(inv[key]))
        scalars = np.array(scalars, dtype=np.float32)
        # Goal vector can encode desired number of logs; default to zeros
        goal = np.zeros_like(scalars)
        return {"pixels": pixels, "scalars": scalars, "goal": goal}
