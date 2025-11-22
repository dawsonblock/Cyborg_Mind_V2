"""PPO training script for the Minecraft skill.

This script trains the actor–critic head of the Minecraft skill using
Proximal Policy Optimisation (PPO) from ``stable_baselines3``.  The
Capsule Brain provides a feature extractor that maps raw
observations into a latent space.  For simplicity we treat the
Minecraft task as a standard reinforcement learning problem and let
SB3 handle the optimisation.
"""

import argparse
from typing import Any, Dict, Optional

try:
    import gym
except ImportError:  # pragma: no cover - fallback for modern Farama stack
    import gymnasium as gym  # type: ignore[assignment]
import numpy as np
import torch
import torch.nn as nn
from typing import Dict

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
except ImportError:
    PPO = None  # type: ignore

from ....core.brain import CapsuleBrain
from ....core.config import CapsuleBrainConfig
from ..envs import MineRLEnvAdapter


class CapsuleFeaturesExtractor(BaseFeaturesExtractor):
    """Extract features using the Capsule Brain encoder and workspace."""

    def __init__(self, observation_space: gym.Space, brain: CapsuleBrain):
        # Determine output size equal to workspace dimension
        super().__init__(observation_space, features_dim=brain.cfg.workspace.dim)
        self.brain = brain.eval()

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        """
        Extract latent features from raw observations using the Capsule Brain
        without invoking the skill head.  The brain is run in evaluation
        mode with no previous state so that it always resets its recurrent
        modules between batches.
        """
        pixels = observations["pixels"]  # shape (B,C,H,W)
        device = pixels.device
        with torch.no_grad():
            out = self.brain(
                pixels.to(device),
                scalars=None,
                goals=None,
                audio=None,
                env=None,
                prev_state=None,
            )
        # Return the workspace representation as the feature vector
        return out["workspace"]


def make_env(env_name: str) -> gym.Env:
    adapter = MineRLEnvAdapter(env_name)
    # Wrap adapter into a Gym environment
    class AdapterEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            # Define observation and action spaces matching the adapter
            self.observation_space = gym.spaces.Dict(
                {
                    "pixels": gym.spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
                    "scalars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32),
                    "goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32),
                }
            )
            # The action space is discrete with one entry per action in the adapter
            self.action_space = gym.spaces.Discrete(len(adapter.action_map))

        def reset(self) -> Dict[str, np.ndarray]:
            return adapter.reset()

        def step(self, action: int):
            obs, reward, done, info = adapter.step(action)
            return obs, reward, done, info
    return AdapterEnv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Minecraft skill using PPO")
    parser.add_argument("--env", type=str, default="MineRLTreechop-v0", help="MineRL environment")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="minecraft_skill.zip", help="Path to save trained policy")
    parser.add_argument(
        "--logdir",
        type=str,
        default="",
        help="Directory for TensorBoard logs.  If empty, logging is disabled.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=0,
        help="Evaluate the policy every N timesteps on a separate environment. 0 to disable.",
    )
    args = parser.parse_args()
    if PPO is None:
        raise RuntimeError("stable-baselines3 is required to train with PPO. Install via pip install stable-baselines3")
    cfg = CapsuleBrainConfig()
    brain = CapsuleBrain(cfg)
    env = make_env(args.env)
    policy_kwargs = dict(
        features_extractor_class=CapsuleFeaturesExtractor,
        features_extractor_kwargs=dict(brain=brain),
    )
    # Optional TensorBoard logging
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None
    if args.logdir:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        writer = SummaryWriter(args.logdir)

    class TensorboardCallback(BaseCallback):
        """Callback for logging training metrics to TensorBoard."""

        def __init__(self, eval_env: Optional[gym.Env] = None, eval_interval: int = 0) -> None:
            super().__init__()
            self.eval_env = eval_env
            self.eval_interval = eval_interval
            self.num_calls = 0

        def _on_step(self) -> bool:
            if writer is not None:
                # Log training reward and episode length if available
                if "episode" in self.locals:
                    ep_info = self.locals["episode"]
                    if "r" in ep_info:
                        writer.add_scalar("train/episode_reward", ep_info["r"], self.num_calls)
                    if "l" in ep_info:
                        writer.add_scalar("train/episode_length", ep_info["l"], self.num_calls)
            self.num_calls += 1
            # Evaluate periodically
            if self.eval_env is not None and self.eval_interval > 0:
                if self.num_calls % self.eval_interval == 0:
                    mean_reward, std_reward = evaluate_policy(
                        self.model,
                        self.eval_env,
                        n_eval_episodes=5,
                        deterministic=True,
                        render=False,
                    )
                    if writer is not None:
                        writer.add_scalar("eval/mean_reward", mean_reward, self.num_calls)
                        writer.add_scalar("eval/std_reward", std_reward, self.num_calls)
            return True

    # Create evaluation environment if requested
    eval_env = None
    if args.eval_interval > 0:
        eval_env = make_env(args.env)

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.lr,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    callback: Optional[BaseCallback] = None
    if writer is not None or eval_env is not None:
        callback = TensorboardCallback(eval_env=eval_env, eval_interval=args.eval_interval)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.output)
    # Close writer
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
