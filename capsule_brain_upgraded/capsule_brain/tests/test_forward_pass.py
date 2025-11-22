"""Forward pass test for the Capsule Brain.

This test instantiates a CapsuleBrain with default configuration,
generates a dummy image batch of shape (B,3,64,64) and performs a
forward pass.  The test asserts that required keys are present in the
output and that no exceptions are raised.
"""

import torch

from capsule_brain.core.brain import CapsuleBrain
from capsule_brain.core.config import CapsuleBrainConfig


def test_forward_pass() -> None:
    cfg = CapsuleBrainConfig()
    brain = CapsuleBrain(cfg)
    # Use CPU for test
    device = torch.device("cpu")
    brain.to(device)
    B = 2
    pixels = torch.randn(B, 3, 64, 64, device=device)
    out = brain(pixels, env=None)
    # Check keys
    assert "action_logits" in out
    assert "workspace" in out
    assert "emotion" in out
    # Shapes
    assert out["action_logits"].shape[0] == B
    assert out["workspace"].shape[0] == B