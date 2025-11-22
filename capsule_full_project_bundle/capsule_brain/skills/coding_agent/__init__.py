"""Coding skill package.

This package defines a minimal coding skill that can be used as a
placeholder for future text‑to‑code or code execution abilities.  The
current implementation is a simple actor–critic head that outputs a
small set of abstract actions (e.g. generate code, no‑op) and a value
estimate.
"""

from .skill import Skill  # noqa: F401
