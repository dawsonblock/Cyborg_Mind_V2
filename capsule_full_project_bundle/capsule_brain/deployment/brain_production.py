"""Production wrapper for Capsule Brain.

This wrapper performs just‑in‑time scripting and optional dynamic
quantisation on linear layers to improve inference performance.
Use ``ProductionBrain`` to load a trained brain in deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.brain import CapsuleBrain
from ..core.config import CapsuleBrainConfig


class ProductionBrain(nn.Module):
    """Optimised inference‑only Capsule Brain."""

    def __init__(self, ckpt_path: str | None = None, device: str = "cuda") -> None:
        super().__init__()
        # Determine device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Instantiate and load brain
        brain = CapsuleBrain(CapsuleBrainConfig()).to(self.device)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=self.device)
            brain.load_state_dict(state, strict=False)
        brain.eval()
        # Script the brain
        scripted = torch.jit.script(brain)
        # Apply dynamic quantisation on linear layers
        self.brain = torch.quantization.quantize_dynamic(
            scripted, {nn.Linear}, dtype=torch.qint8
        )
        self.brain.eval()

    @torch.no_grad()
    def forward(
        self,
        pixels: torch.Tensor,
        scalars: torch.Tensor | None = None,
        goals: torch.Tensor | None = None,
        env: str | None = None,
    ) -> dict[str, torch.Tensor]:
        B = pixels.size(0)
        return self.brain(pixels, scalars, goals, env=env)
