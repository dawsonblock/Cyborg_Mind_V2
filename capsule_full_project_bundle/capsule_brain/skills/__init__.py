"""Skill registry.

Skills are modular sub‑brains that implement task specific behaviour.
Each subpackage under ``capsule_brain.skills`` should expose a
``Skill`` class derived from ``torch.nn.Module`` with a ``forward``
method taking a workspace vector and returning a dictionary with
``action_logits`` and optionally other keys (e.g. value estimates).
"""

__all__ = []  # skills are loaded dynamically by CapsuleBrain
