"""
Metacognitive control components.

Includes the PPO actor‑critic network used to dynamically choose the
cognitive capacity (number of FRNN modes) based on the current task
context, recent accuracy and energy budget.
"""

from .ppo_controller import CapacityActorCritic  # noqa: F401