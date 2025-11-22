"""
Capsule Brain AGI Framework
===========================

This package contains a modular AGI architecture built on top of a number
of loosely coupled subsystems.  Each subsystem is responsible for a
different aspect of cognition such as memory, recurrent processing,
workspace integration, emotional modulation, symbolic reasoning and
specialised skill execution.  The goal of the Capsule Brain is to unify
perception, memory and action under a common interface while allowing
additional skills to be plugged in on demand.

The high level architecture follows the guidance of Global Workspace
Theory – a small bottleneck (the ``workspace``) mediates the flow of
information between distributed specialists【609993358319586†L382-L399】.  Memory is
implemented using a pseudo‑mode memory (PMM) with content addressable
slots and dynamic expansion, while temporal coherence is maintained
through a Fractional Recurrent Neural Network (FRNN)【131464440980276†L620-L630】.  An
emotion engine tracks continuous valence–arousal–dominance channels
【93681059298509†L51-L83】, allowing the brain to model affective state.

Skills (such as playing Minecraft) live under ``capsule_brain.skills``
and implement a simple interface of ``forward(workspace) -> action``.
The brain selects which skill to invoke via an ``ActionCapsule`` that
consults the current workspace contents.
"""

from .core.brain import CapsuleBrain  # noqa: F401