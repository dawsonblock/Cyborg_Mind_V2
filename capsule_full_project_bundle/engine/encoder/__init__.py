"""
Global encoder subpackage.

Implements a simple feed‑forward network to produce a fixed‑dimension
representation from raw input features.  In production this can be
replaced with a pretrained model (e.g. a Transformer, CNN or multimodal
encoder).
"""

from .global_encoder import SimpleGlobalEncoder  # noqa: F401