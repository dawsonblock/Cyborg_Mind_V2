"""
Train the meta‑controller for multi‑skill selection.

This script performs supervised training of the meta controller by
presenting it with a variety of environments (or tasks) and
teaching it to select the appropriate skill for each one.  The
capsule brain’s encoder, PMM and FRNN are frozen so that only the
meta controller’s parameters are updated.  This provides a way to
bootstrap multi‑skill behaviour without having to perform full
reinforcement learning across multiple tasks.

Example usage:

    python -m capsule_brain.training.train_multi_skill \
        --envs MineRLTreechop-v0 Gridworld Chat
        --steps 1000
        --lr 1e-4
        --save meta_controller.pt

Notes:
    * This training script uses dummy observations for tasks that
      cannot be run in the current environment (e.g. Chat tasks).
    * For RL‑based training of the meta controller you should
      integrate the meta controller into the PPO trainer and allow
      gradients to flow through it.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..core.brain import CapsuleBrain
from ..core.config import CapsuleBrainConfig


class TaskDataset(Dataset):
    """Dataset of tasks with ground‑truth skill indices.

    Each item consists of a dummy observation and a label indicating
    which skill should be chosen.  Observations are represented by
    random pixel tensors; in practice you may want to load real
    observations from each environment.
    """

    def __init__(self, tasks: List[str], skill_map: Dict[str, int], length: int = 1000) -> None:
        self.tasks = tasks
        self.skill_map = skill_map
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random dummy observation: 3×64×64
        pixels = torch.randn(3, 64, 64)
        # Select a random task
        task = np.random.choice(self.tasks)
        label = torch.tensor(self.skill_map[task], dtype=torch.long)
        return {"pixels": pixels, "task": task, "label": label}


def train_meta_controller(
    brain: CapsuleBrain,
    tasks: List[str],
    lr: float,
    steps: int,
    save_path: Optional[str] = None,
) -> None:
    """Train the brain’s meta controller on a list of tasks.

    Args:
        brain: A ``CapsuleBrain`` instance with multiple skills.
        tasks: A list of task names corresponding to allowed skills.
        lr: Learning rate for Adam.
        steps: Number of training steps.
        save_path: Optional path to save the trained meta controller.
    """
    # Build mapping from task name to skill index
    skill_names = list(brain.skills.keys())
    skill_map = {task: skill_names.index(task) if task in skill_names else 0 for task in tasks}
    dataset = TaskDataset(tasks, skill_map, length=steps)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    # Freeze all parameters except meta controller
    for param in brain.parameters():
        param.requires_grad = False
    for param in brain.meta_controller.parameters():
        param.requires_grad = True
    brain.train()
    optimizer = torch.optim.Adam(brain.meta_controller.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain.to(device)
    step_count = 0
    while step_count < steps:
        for batch in loader:
            if step_count >= steps:
                break
            pixels = batch["pixels"].to(device)
            labels = batch["label"].to(device)
            # Forward pass through brain to obtain workspace and meta logits
            out = brain(pixels, env=None)
            # Meta controller logits already combined with action capsule; use
            # meta controller directly for supervision to avoid skill bias
            with torch.no_grad():
                workspace = out["workspace"]
            logits = brain.meta_controller(workspace)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_count += pixels.size(0)
        print(f"Step {step_count}/{steps}, loss: {loss.item():.4f}")
    if save_path is not None:
        torch.save(brain.meta_controller.state_dict(), save_path)
        print(f"Meta controller saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the meta controller for multi‑skill selection")
    parser.add_argument("--envs", nargs="+", default=["MineRLTreechop-v0", "Gridworld", "Chat"], help="List of tasks/environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save", type=str, default="", help="Optional path to save the meta controller state")
    args = parser.parse_args()
    cfg = CapsuleBrainConfig()
    brain = CapsuleBrain(cfg)
    train_meta_controller(brain, args.envs, args.lr, args.steps, save_path=args.save if args.save else None)


if __name__ == "__main__":
    main()