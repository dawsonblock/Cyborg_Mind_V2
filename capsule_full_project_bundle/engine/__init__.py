"""
Capsule Brain engine package.

This package exposes the high‑level `CapsuleMemoryEngine` and its configuration.
It wires together the global encoder, pseudo‑mode memory, recurrent memory and
metacognitive capacity controller.
"""

from .config import EngineConfig  # noqa: F401
from .core import CapsuleMemoryEngine  # noqa: F401