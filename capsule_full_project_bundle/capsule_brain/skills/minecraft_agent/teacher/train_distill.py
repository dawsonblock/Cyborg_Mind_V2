"""Command line script to distill a student model from the CLIP teacher.

This script samples frames from a MineRL environment and uses the
``RealTeacher`` to generate target embeddings.  A student encoder is
trained to regress these embeddings using mean squared error.  The
resulting student can be used within the Capsule Brain to provide
compact visual representations.
"""

import argparse
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import minerl
except ImportError:
    minerl = None

from .clip_teacher import RealTeacher, distill_student


class StudentEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return h / h.norm(dim=-1, keepdim=True)


def collect_frames(env_name: str, num_frames: int) -> torch.Tensor:
    if minerl is None:
        raise RuntimeError("minerl must be installed to collect frames")
    env = minerl.make(env_name)
    obs = env.reset()
    frames = []
    for _ in range(num_frames):
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        frame = obs["pov"].astype("float32") / 255.0
        frame = torch.from_numpy(frame.transpose(2, 0, 1))
        frames.append(frame)
        if done:
            obs = env.reset()
    env.close()
    return torch.stack(frames)


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = RealTeacher(device=device)
    student = StudentEncoder(embedding_dim=512)
    # Collect frames
    frames = collect_frames(args.env, args.frames)
    dataset = TensorDataset(frames, torch.zeros(len(frames)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    student = distill_student(
        teacher,
        student,
        dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    torch.save(student.state_dict(), args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill CLIP teacher into student encoder")
    parser.add_argument("--env", type=str, default="MineRLTreechop-v0", help="MineRL environment name")
    parser.add_argument("--frames", type=int, default=5000, help="Number of frames to collect")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="student_encoder.pth", help="Path to save student weights")
    args = parser.parse_args()
    main(args)
