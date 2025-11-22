"""
Capsule Memory Engine core.

This module defines the `CapsuleMemoryEngine` class which orchestrates
encoding, memory retrieval/update, metacognitive capacity selection and
recurrent processing.  It exposes a `forward_step` method for
incrementally processing inputs with stateful recurrence and a
`reset_state` method to clear the recurrent state.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import EngineConfig
from .encoder.global_encoder import SimpleGlobalEncoder
from .memory.pmm import StaticPseudoModeMemory
from .memory.frnn import FRNNCore
from .metacog.ppo_controller import CapacityActorCritic


class CapsuleMemoryEngine(nn.Module):
    """
    Unified Capsule Brain Memory Engine.

    Wiring:
        raw input -> GlobalEncoder -> PMM -> FRNN -> action / thought
                                   \-> PPO (capacity control)
    """

    def __init__(self, cfg: Optional[EngineConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or EngineConfig()
        self.encoder = SimpleGlobalEncoder(self.cfg.encoder)
        self.pmm = StaticPseudoModeMemory(self.cfg.pmm)
        self.frnn = FRNNCore(self.cfg.frnn)
        # PPO state vector: global (d_global) + 3 scalars (difficulty, recent accuracy, energy)
        self.ppo_state_dim = self.cfg.encoder.d_global + 3
        self.ppo = CapacityActorCritic(self.ppo_state_dim, self.cfg.ppo)
        # Output head from FRNN hidden state to generic action logits
        self.action_head = nn.Linear(self.cfg.frnn.output_dim, 10)
        # Buffers for performance feedback
        self.register_buffer("recent_accuracy", torch.tensor(1.0))
        self.register_buffer("energy_budget", torch.tensor(1.0))
        # Hidden state holder
        self._frnn_state: Optional[torch.Tensor] = None

    def encode(self, features: Dict[str, Any]) -> torch.Tensor:
        return self.encoder.encode_state(features)

    def select_capacity(
        self, global_repr: torch.Tensor, difficulty: float
    ) -> Dict[str, torch.Tensor]:
        B = global_repr.size(0)
        device = global_repr.device
        difficulty_vec = torch.full((B, 1), float(difficulty), device=device)
        acc_vec = self.recent_accuracy.expand(B, 1)
        energy_vec = self.energy_budget.expand(B, 1)
        state_vec = torch.cat([global_repr, difficulty_vec, acc_vec, energy_vec], dim=-1)
        return self.ppo.act(state_vec)

    def forward_step(
        self,
        features: Dict[str, Any],
        difficulty: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        global_repr = self.encode(features).to(device)
        # Memory retrieval
        pmm_out = self.pmm(global_repr)
        # Update memory
        self.pmm.update(global_repr)
        # Capacity selection
        ppo_out = self.select_capacity(global_repr, difficulty)
        capacity = ppo_out["capacity"]
        # FRNN step
        x_seq = pmm_out["value"].unsqueeze(1)  # (B, 1, dim)
        frnn_out, self._frnn_state = self.frnn(x_seq, self._frnn_state)
        last_state = frnn_out[:, -1, :]
        action_logits = self.action_head(last_state)
        return {
            "global_repr": global_repr,
            "pmm_value": pmm_out["value"],
            "pmm_weights": pmm_out["weights"],
            "capacity": capacity,
            "action_logits": action_logits,
        }

    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = next(self.parameters()).device
        self._frnn_state = self.frnn.reset_state(batch_size, device)