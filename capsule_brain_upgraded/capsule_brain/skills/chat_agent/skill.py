"""Chat skill for the Capsule Brain.

This skill represents a very simple text interaction module.  Given
the refined workspace from the Capsule Brain it outputs logits over a
small vocabulary of actions such as "ask", "respond" and "end
conversation".  In a production system this module would interface
with a language model or retrieval engine and maintain a dialogue
context, but here it serves as a placeholder for integrating
non‑visual skills.
"""

from typing import Dict, Any

import torch
import torch.nn as nn

from ...core.config import CapsuleBrainConfig
from ...api.tutor_bridge import TutorBridge


class Skill(nn.Module):
    """A minimal chat skill.

    The skill processes the workspace vector and produces action logits
    representing high‑level dialogue acts.  It also outputs a value
    estimate for reinforcement learning.
    """

    def __init__(self, cfg: CapsuleBrainConfig) -> None:
        super().__init__()
        input_dim = cfg.workspace.dim
        hidden_dim = 256
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Define three actions: 0=ask, 1=respond, 2=end conversation
        self.action_head = nn.Linear(hidden_dim, 3)
        self.value_head = nn.Linear(hidden_dim, 1)
        # Initialise a tutor bridge.  If API key is not provided
        # (likely in offline mode) the bridge will return a placeholder
        # response.  In production you should pass an API key via
        # environment variable or configuration.
        self.tutor = TutorBridge()

    def forward(self, workspace: torch.Tensor) -> Dict[str, Any]:
        """Generate chat actions and optionally produce a tutor response.

        Args:
            workspace: Capsule Brain workspace vector of shape [B, D].

        Returns:
            A dictionary containing action logits, value estimates and a
            textual response from the tutor.  The tutor is queried
            with a simple prompt based on the mean of the workspace.
        """
        x = self.policy(workspace)
        logits = self.action_head(x)
        value = self.value_head(x)
        # Query the tutor for a quick response.  Use the mean of the
        # workspace vector as a crude summary of the agent’s state.
        try:
            summary = workspace.mean().item()
            prompt = f"The agent’s current internal state has mean value {summary:.3f}.  "
            prompt += "Provide a short encouraging message."
            response = self.tutor.ask(prompt)
        except Exception:
            response = "[No tutor response]"
        return {"action_logits": logits, "value": value, "text_response": response}
