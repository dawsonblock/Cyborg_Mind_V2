"""RealTeacher for Minecraft skill based on CLIP.

The RealTeacher uses a pre‑trained CLIP model to produce dense
representations of visual observations.  These representations are
treated as targets for a student network during the initial
distillation stage.  The teacher does not update its parameters.
"""

from typing import Iterable, Tuple

import torch
import torch.nn as nn

try:
    import clip
except ImportError:
    clip = None


class RealTeacher(nn.Module):
    """Wrapper around a pre‑trained CLIP model."""

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        if clip is None:
            raise RuntimeError(
                "The 'clip' package is required. Install it via pip install git+https://github.com/openai/CLIP.git"
            )
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into CLIP embeddings.

        Args:
            frames: Tensor of shape ``(B, C, H, W)`` in range [0,1].
        Returns:
            Normalised embeddings of shape ``(B, D)``.
        """
        # CLIP expects inputs in range [0,1] and normalised using its own preprocessing.
        with torch.no_grad():
            # CLIP preprocess transforms PIL images; here we assume frames are already torch tensors
            # and simply rescale to 0–1 and normalize using the CLIP mean/std.
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=frames.device).view(1,3,1,1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=frames.device).view(1,3,1,1)
            frames_norm = (frames - mean) / std
            image_features = self.model.encode_image(frames_norm)
            return image_features / image_features.norm(dim=-1, keepdim=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(frames)


def distill_student(
    teacher: RealTeacher,
    student: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
) -> nn.Module:
    """Train a student network to mimic the CLIP teacher.

    Args:
        teacher: Instance of ``RealTeacher``.
        student: Neural network taking images and producing embeddings.
        dataloader: Iterable yielding batches of images and dummy labels.
        epochs: Number of epochs to train.
        lr: Learning rate.
        device: Device to run training on.
    Returns:
        The trained student network.
    """
    device = torch.device(device)
    teacher.eval()
    student.to(device)
    optim = torch.optim.Adam(student.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for frames, _ in dataloader:
            frames = frames.to(device)
            with torch.no_grad():
                target = teacher(frames)
            pred = student(frames)
            loss = loss_fn(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return student
