"""Pseudo‑mode memory implementations.

The pseudo‑mode memory (PMM) acts as a content addressable store for
high‑level embeddings.  The simplest implementation uses a fixed set
of memory slots updated via an exponential moving average.  When the
memory becomes saturated a dynamic variant grows the number of slots
and implements garbage collection.  Memory pressure can be monitored
via attention entropy and write density metrics.
"""

from .static_pmm import StaticPseudoModeMemory  # noqa: F401
from .dynamic_pmm import DynamicPseudoModeMemory  # noqa: F401
