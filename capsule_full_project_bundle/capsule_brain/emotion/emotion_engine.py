"""Continuous emotion engine.

This module implements a simple feed‑forward network that updates
continuous emotion vectors in the Valence–Arousal–Dominance (VAD)
space【93681059298509†L51-L83】.  Given a set of input features (for example
the current workspace, perceptual embeddings and retrieved memory)
the network predicts a delta to the emotion vector.  The updated
emotion is clamped to the range ``[-1, 1]``.
"""

from typing import Optional

import torch
import torch.nn as nn

from ..core.config import EmotionConfig


class EmotionEngine(nn.Module):
    """Multi‑layer perceptron producing emotion updates."""

    def __init__(self, cfg: EmotionConfig, input_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.num_channels),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, prev_emotion: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict the updated emotion vector.

        Args:
            x: Input features of shape ``(B, input_dim)``.
            prev_emotion: Previous emotion vector of shape ``(B, num_channels)``.
        Returns:
            Updated emotion in the range ``[-1, 1]``.
        """
        delta = self.net(x)
        if prev_emotion is None:
            prev_emotion = torch.zeros_like(delta)
        emotion = prev_emotion + delta
        return torch.clamp(emotion, -1.0, 1.0)
