"""
Evaluate the quality of a student encoder distilled from CLIP.

This script compares the embeddings produced by the student encoder
against those produced by a pre‑trained CLIP model on a set of
Minecraft frames.  The mean squared error (MSE) or cosine distance
between the two embeddings provides a simple measure of how well the
student approximates the teacher.

Usage:

    python -m capsule_brain.skills.minecraft_agent.teacher.evaluate_distill \
        --env MineRLTreechop-v0 --frames 100 --student_path path/to/student.pt

If the ``minerl`` package is available this script will sample
observations from the specified environment using the env adapter.
Otherwise it will fall back to random image tensors.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import minerl  # type: ignore
except ImportError:
    minerl = None  # type: ignore

from .clip_teacher import RealTeacher, StudentEncoder
from ..envs.minerl_env import MineRLEnvAdapter


def sample_frames(env_name: str, num_frames: int) -> torch.Tensor:
    """Collect a batch of frames from a MineRL environment.

    If MineRL is not installed this generates random images.
    """
    frames = []
    if minerl is None:
        # Generate random images (64×64 RGB) if minerl unavailable
        for _ in range(num_frames):
            frames.append(torch.rand(3, 64, 64))
    else:
        env = MineRLEnvAdapter(env_name)
        obs = env.reset()
        for _ in range(num_frames):
            frames.append(obs["pixels"])  # type: ignore
            # Take a random action to change environment state
            action = np.random.randint(0, env.action_space.n)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
    return torch.stack(frames, dim=0)


def evaluate(student: StudentEncoder, teacher: RealTeacher, frames: torch.Tensor, device: str = "cpu") -> float:
    """Compute the mean squared error between student and teacher embeddings.
    Args:
        student: The student encoder.
        teacher: The teacher model.
        frames: Tensor of shape [N, 3, H, W] containing image observations.
        device: Device for inference.
    Returns:
        Mean squared error between student and teacher embeddings.
    """
    student.to(device)
    teacher.to(device)
    student.eval()
    teacher.eval()
    with torch.no_grad():
        frames = frames.to(device)
        student_emb = student(frames)
        teacher_emb = teacher(frames)
        mse = F.mse_loss(student_emb, teacher_emb).item()
    return mse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP distillation")
    parser.add_argument("--env", type=str, default="MineRLTreechop-v0", help="Name of the MineRL environment")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to sample")
    parser.add_argument("--student_path", type=str, required=True, help="Path to the trained student encoder (.pt)")
    args = parser.parse_args()
    # Load teacher and student
    teacher = RealTeacher(device="cpu")
    student = StudentEncoder(output_dim=teacher.output_dim)
    student.load_state_dict(torch.load(args.student_path, map_location="cpu"))
    # Sample frames
    frames = sample_frames(args.env, args.frames)
    mse = evaluate(student, teacher, frames)
    print(f"Distillation MSE: {mse:.6f}")


if __name__ == "__main__":
    main()
