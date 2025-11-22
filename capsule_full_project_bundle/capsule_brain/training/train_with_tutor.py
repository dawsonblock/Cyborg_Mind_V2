"""
Training script that uses the external AI Tutor to provide hints.

This script illustrates how the TutorBridge can be used during
reinforcement learning to provide additional feedback.  It runs
episodes in a specified environment, collects a simple textual
summary of events, and periodically queries the tutor for advice.
The tutor’s response is printed to the console and could be
incorporated into the agent’s memory or training loop for
curriculum shaping.

Example usage:

    python -m capsule_brain.training.train_with_tutor \
        --env MineRLTreechop-v0 --episodes 3 --tutor_model gpt-4 --api_key sk-...
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

try:
    import minerl  # type: ignore
except ImportError:
    minerl = None  # type: ignore

import numpy as np
import torch

from ..core.brain import CapsuleBrain
from ..core.config import CapsuleBrainConfig
from ..api.tutor_bridge import TutorBridge
from ..skills.minecraft_agent.envs import MineRLEnvAdapter


def run_episode(env_name: str, brain: CapsuleBrain) -> List[str]:
    """Run one episode and return a textual log of events.

    The returned log is a list of strings describing each time step.
    """
    logs: List[str] = []
    if env_name == "Gridworld":
        # Use gridworld adapter
        from ..skills.gridworld_agent.envs.gridworld_env import GridworldEnv
        env = GridworldEnv()
    elif env_name == "Chat":
        # Chat tasks have no environment; produce dummy log
        logs.append("Chat session started.")
        return logs
    else:
        # Default to MineRL environment
        try:
            env = MineRLEnvAdapter(env_name)
        except Exception:
            # Fallback: no real environment available
            logs.append(f"[Warning] Environment {env_name} unavailable; skipping episode.")
            return logs
    obs = env.reset()
    done = False
    prev_state = None
    step = 0
    while not done and step < 10:
        # Prepare tensors for brain
        pixels = torch.tensor(obs["pixels"]).unsqueeze(0).float()
        out = brain(pixels, env=env_name, prev_state=prev_state)
        # Select action
        action_logits = out["action_logits"][0]
        action = int(torch.argmax(action_logits).item())
        obs, reward, done, info = env.step(action)
        prev_state = out
        logs.append(
            f"Step {step}: action={action}, reward={reward}, done={done}"  # type: ignore
        )
        step += 1
    logs.append("Episode finished.")
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with Tutor guidance")
    parser.add_argument("--env", type=str, default="MineRLTreechop-v0", help="Environment name")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--tutor_model", type=str, default="gpt-4", help="Tutor model name")
    parser.add_argument("--api_key", type=str, default="", help="API key for the tutor")
    args = parser.parse_args()
    cfg = CapsuleBrainConfig()
    brain = CapsuleBrain(cfg)
    tutor = TutorBridge(model=args.tutor_model, api_key=args.api_key if args.api_key else None)
    for ep in range(args.episodes):
        log = run_episode(args.env, brain)
        # Compose a prompt summarising the episode
        summary = "\n".join(log)
        prompt = f"Episode log:\n{summary}\n\nPlease suggest improvements or hints for the next run."
        response = tutor.ask(prompt)
        print(f"Tutor suggestion after episode {ep+1}:")
        print(response)
        print("---")


if __name__ == "__main__":
    main()