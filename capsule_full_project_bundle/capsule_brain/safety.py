"""Safety and self‑model modules.

This module defines stub classes for safety checks and self‑modeling.
In a complete AGI system these components would enforce policy
constraints, ethical considerations and introspective reasoning over
the agent’s own state.  Here they serve as placeholders and examples
of how such modules might be integrated.
"""

from typing import Any, Dict, List, Optional

import json
from pathlib import Path

import torch


class SafetyCapsule:
    """Safety capsule enforcing environment–skill policies.

    The safety capsule monitors the logits over skills produced by the
    brain and masks those that are forbidden in the current context.
    Policy rules are read from a JSON file ``safety_rules.json`` in
    the ``capsule_brain`` package directory.  The rules file should
    contain a mapping of environment names to lists of allowed
    skills.  For example:

    ``{
        "allowed_skills": {
            "MineRLTreechop-v0": ["minecraft_agent"],
            "Gridworld": ["gridworld_agent"],
            "Chat": ["chat_agent", "coding_agent"]
        }
    }``

    If no rules are found for the current environment the logits are
    returned unchanged.  This mechanism can be extended to enforce
    finer grained constraints (e.g. mask individual actions within a
    skill) by modifying the rules file and corresponding logic.
    """

    def __init__(self, rules_path: Optional[str] = None) -> None:
        # Load rules from the specified path or default to the package
        # directory.
        if rules_path is None:
            # Determine the default location relative to this file
            package_dir = Path(__file__).resolve().parent
            rules_file = package_dir / "safety_rules.json"
        else:
            rules_file = Path(rules_path)
        if rules_file.is_file():
            try:
                with open(rules_file, "r", encoding="utf-8") as f:
                    self.rules = json.load(f)
            except Exception:
                self.rules = {}
        else:
            self.rules = {}

    def filter_action(self, skill_logits: torch.Tensor, env: Optional[str] = None) -> torch.Tensor:
        """Mask skill logits based on the current environment.

        Args:
            skill_logits: Tensor of shape [B, num_skills] representing
                logits for each skill.
            env: Optional name of the current environment.  If provided
                and rules exist for this environment, skills not in the
                allowed list will have their logits set to a very
                negative value.
        Returns:
            Modified logits tensor.
        """
        if env is None or "allowed_skills" not in self.rules:
            return skill_logits
        allowed_map: Dict[str, List[str]] = self.rules.get("allowed_skills", {})  # type: ignore
        allowed = allowed_map.get(env)
        if allowed is None:
            return skill_logits
        # Determine which skills are allowed; assume skills are ordered
        # consistently with the brain's skill registry.  We mask
        # disallowed skills by setting logits to a large negative value.
        mask = torch.full_like(skill_logits, float("-inf"))
        for idx, skill_name in enumerate(self.skill_names):  # type: ignore[attr-defined]
            if skill_name in allowed:
                mask[:, idx] = 0.0
        return skill_logits + mask


class SelfModel:
    """Placeholder for a self‑model.

    A self‑model predicts the agent’s own future states and evaluates
    the consequences of candidate actions.  In this stub it simply
    stores and echoes state information.
    """

    def __init__(self) -> None:
        self.state: Any = None

    def update(self, new_state: Any) -> None:
        """Update the internal model of self.

        Args:
            new_state: Arbitrary state representation.
        """
        self.state = new_state

    def predict(self) -> Any:
        """Predict the next state.  In this stub returns the last state."""
        return self.state
