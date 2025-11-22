"""
Memory components for the Capsule Memory Engine.

This subpackage includes the static pseudo‑mode memory (PMM) used for
longer‑term storage of global states and the FRNN recurrent core which
maintains a stateful representation over sequences.
"""

from .pmm import StaticPseudoModeMemory  # noqa: F401
from .frnn import FRNNCore  # noqa: F401