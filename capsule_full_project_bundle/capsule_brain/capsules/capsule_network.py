"""Capsule network modules.

Capsule networks (CapsNets) are neural architectures that model
hierarchical relationships between objects in data【909429444129561†L134-L151】.  Each
capsule represents an entity or part and outputs a vector whose length
encodes the probability of the entity’s presence while its orientation
encodes instantiation parameters.  Dynamic routing by agreement allows
higher‑level capsules to selectively aggregate outputs from lower
levels based on the agreement between their poses.

In this implementation we provide a minimal capsule layer and a
processor network suitable for integration into the Capsule Brain.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import CapsuleConfig


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Squash activation used in capsule networks.

    This nonlinearity scales vectors to have length in the range (0,1).
    Args:
        s: Input tensor.
        dim: Dimension along which to compute the norm.
        eps: Small constant to avoid division by zero.
    Returns:
        Squashed tensor with the same shape as ``s``.
    """
    squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * s / torch.sqrt(squared_norm + eps)


class CapsuleLayer(nn.Module):
    """A single capsule layer with dynamic routing.

    This layer transforms a set of input capsules into a set of output
    capsules using learned linear mappings and dynamic routing by
    agreement.  Input capsules are expected to have shape
    ``(B, in_capsules, in_dim)`` and outputs have shape
    ``(B, out_capsules, out_dim)``.  The number of routing iterations
    controls the sharpness of the coupling coefficients.
    """

    def __init__(
        self,
        in_capsules: int,
        in_dim: int,
        out_capsules: int,
        out_dim: int,
        num_routes: int = 3,
    ) -> None:
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_routes = num_routes
        # Transformation weights W_{i,o} of shape (in_capsules, out_capsules, in_dim, out_dim)
        # Each input capsule i has a weight matrix for each output capsule o.
        self.W = nn.Parameter(0.1 * torch.randn(in_capsules, out_capsules, in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_capsules, in_dim)
        B = x.size(0)
        # Compute predicted output capsule poses for each input capsule
        # x_hat[b,i,o,d] = sum_k x[b,i,k] * W[i,o,k,d]
        # Use einsum to perform batch matrix multiplication
        # x: b i k; W: i o k d -> b i o d
        x_hat = torch.einsum("bik, iokd -> bio d", x, self.W)
        # Routing logits initialised to zero
        b = torch.zeros(B, self.in_capsules, self.out_capsules, device=x.device)
        for _ in range(self.num_routes):
            # Coupling coefficients via softmax over output capsules
            c = torch.softmax(b, dim=-1)  # (B, in_capsules, out_capsules)
            # Aggregate votes from input capsules
            # s[b,o,d] = sum_i c[b,i,o] * x_hat[b,i,o,d]
            s = (c.unsqueeze(-1) * x_hat).sum(dim=1)
            # Squash to get output capsule vectors
            v = squash(s, dim=-1)
            # Update logits based on agreement
            # agreement a[b,i,o] = dot(x_hat[b,i,o], v[b,o])
            a = (x_hat * v.unsqueeze(1)).sum(dim=-1)
            b = b + a
        return v


class CapsuleProcessorNetwork(nn.Module):
    """High level capsule processor network.

    The processor network takes a flat input and reshapes it into a set
    of primary capsules, applies one or more capsule layers and
    produces a flattened output vector.  This component is used
    downstream of the emotion and workspace integration to refine
    representations before skill selection.
    """

    def __init__(self, cfg: CapsuleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Primary capsules: map input to a set of vector capsules
        self.primary = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.num_capsules * cfg.capsule_dim),
            nn.ReLU(),
        )
        self.capsule_layer = CapsuleLayer(
            in_capsules=cfg.num_capsules,
            in_dim=cfg.capsule_dim,
            out_capsules=cfg.num_capsules,
            out_dim=cfg.capsule_dim,
            num_routes=cfg.num_routes,
        )
        # Output projection to flatten capsule poses
        self.flatten = nn.Linear(cfg.num_capsules * cfg.capsule_dim, cfg.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        prim = self.primary(x)  # (B, num_capsules * capsule_dim)
        prim = prim.view(B, self.cfg.num_capsules, self.cfg.capsule_dim)
        poses = self.capsule_layer(prim)
        out = poses.view(B, -1)
        return self.flatten(out)
