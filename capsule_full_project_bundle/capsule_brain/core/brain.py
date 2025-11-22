"""Top level Capsule Brain class.

The ``CapsuleBrain`` orchestrates all submodules and provides a single
entry point for inference and training.  Given multimodal inputs it
performs the following steps:

1. Encodes the perceptual input into a compact embedding using
   ``SimpleEncoder`` or a user supplied encoder.
2. Retrieves content from the pseudo‑mode memory using the current
   query and updates the memory with the new embedding.
3. Processes sequential dynamics via an FRNN core.
4. Updates the global workspace by combining the FRNN output, the
   retrieved memory and the previous workspace vector.
5. Updates the emotion vector based on the new workspace.
6. Refines the workspace via a capsule processor network.
7. Selects and invokes an appropriate skill module via the
   ``ActionCapsule``.

This design decouples general cognition from task specific skills and
allows new skills to be added simply by registering them under
``capsule_brain.skills``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import importlib
import pkgutil
import torch
import torch.nn as nn

from .config import (
    CapsuleBrainConfig,
    PMMConfig,
    FRNNConfig,
    WorkspaceConfig,
    EmotionConfig,
    CapsuleConfig,
    ToneNetConfig,
)
from .encoder import ImageEncoder, SimpleEncoder
from ..pmm import DynamicPseudoModeMemory
from ..frnn import FRNNCore
from ..workspace import GlobalWorkspaceEngine
from ..emotion import EmotionEngine
from ..capsules import CapsuleProcessorNetwork
from .meta_controller import MetaController
from ..symbolic import SymbolicReasoner
from ..tonenet import ToneNet


class ActionCapsule(nn.Module):
    """Maps the workspace to a distribution over registered skills."""

    def __init__(self, workspace_dim: int, skill_names: list[str]) -> None:
        super().__init__()
        self.skill_names = skill_names
        self.linear = nn.Linear(workspace_dim, len(skill_names))

    def forward(self, workspace: torch.Tensor) -> torch.Tensor:
        # Returns logits over skills
        return self.linear(workspace)


class CapsuleBrain(nn.Module):
    """Aggregate brain combining memory, recurrent core and skills."""

    def __init__(self, cfg: CapsuleBrainConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = CapsuleBrainConfig()
        self.cfg = cfg
        # Instantiate encoder; use an image encoder for pixel observations.
        # The ImageEncoder maps RGB images (B,C,H,W) to a latent vector of
        # dimension ``cfg.frnn.input_dim``.  For non‑image inputs you can
        # replace this with a custom encoder by assigning to
        # ``self.encoder`` before using the brain.
        self.encoder = ImageEncoder(output_dim=cfg.frnn.input_dim, input_channels=3)
        # Memory (dynamic) with key/value dimension equal to FRNN input
        self.pmm = DynamicPseudoModeMemory(cfg.pmm)
        # FRNN core
        self.frnn = FRNNCore(cfg.frnn)
        # ToneNet: instantiate early so we can inspect its vocabulary size.
        self.tonenet = ToneNet(cfg.tonenet)
        # Determine the dimensionality of the inputs to the global workspace.
        # The workspace receives a concatenation of the FRNN output,
        # retrieved memory, emotion vector and optional ToneNet output.
        tone_dim = getattr(self.tonenet, "vocab_size", 0)
        self._tone_dim = tone_dim
        gw_in_dim = (
            cfg.frnn.output_dim + cfg.pmm.dim + cfg.emotion.num_channels + tone_dim
        )
        # Global workspace engine: input_dim accounts for previous workspace
        self.workspace_engine = GlobalWorkspaceEngine(
            cfg.workspace, input_dim=gw_in_dim + cfg.workspace.dim
        )
        # Emotion engine input: workspace and FRNN output
        self.emotion_engine = EmotionEngine(
            cfg.emotion, input_dim=cfg.workspace.dim + cfg.frnn.output_dim
        )
        # Capsule processor network
        self.capsule_net = CapsuleProcessorNetwork(cfg.capsule)
        # Symbolic reasoner
        self.reasoner = SymbolicReasoner()
        # Safety and self‑model modules
        from ..safety import SafetyCapsule, SelfModel
        self.safety = SafetyCapsule()
        self.self_model = SelfModel()
        # Register skills
        self.skills: Dict[str, nn.Module] = {}
        self._load_skills()
        # Action capsule for skill selection
        self.action_capsule = ActionCapsule(cfg.workspace.dim, list(self.skills.keys()))
        # Meta controller for multi‑skill gating.  This learns to bias skill
        # selection based on high‑level cues.  The context dimension can be
        # extended if you wish to include goal vectors or other features.
        self.meta_controller = MetaController(
            workspace_dim=cfg.workspace.dim,
            num_skills=len(self.skills),
            context_dim=0,
            hidden_dim=128,
        )
        # After skills are registered, update safety capsule with skill names
        # so that it can apply environment‑specific masking
        if hasattr(self.safety, "__dict__"):
            # setattr to avoid mypy complaining about attribute defined outside init
            try:
                self.safety.skill_names = list(self.skills.keys())  # type: ignore[attr-defined]
            except Exception:
                pass

    def _load_skills(self) -> None:
        """Dynamically import all skills under ``capsule_brain.skills``."""
        # The ``skills`` package may contain multiple skills.  Each skill
        # should expose a ``Skill`` class with a ``forward`` method.
        try:
            import capsule_brain.skills as skills_pkg  # type: ignore
        except ImportError:
            return
        for finder, name, ispkg in pkgutil.iter_modules(skills_pkg.__path__):  # type: ignore
            if not ispkg:
                continue
            try:
                mod = importlib.import_module(f"capsule_brain.skills.{name}.skill")
                skill_class = getattr(mod, "Skill", None)
                if skill_class is not None:
                    self.skills[name] = skill_class(self.cfg)
            except Exception:
                continue

    def forward(
        self,
        pixels: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        goals: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        env: Optional[str] = None,
        prev_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run one inference step.

        Args:
            pixels: Image observations of shape ``(B, C, H, W)``.
            scalars: Optional scalar inputs ``(B, S)``.
            goals: Optional goal vectors ``(B, G)``.
            audio: Optional raw audio waveforms ``(B, T)`` for ToneNet.
            env: Optional name of the current environment; used to bias skill selection.
            prev_state: Optional dictionary containing previous hidden states
                (``h``, ``workspace``, ``emotion``).
        Returns:
            Dictionary with keys ``action_logits``, ``workspace``, ``emotion``,
            ``memory_pressure``, and skill‑specific outputs.
        """
        B = pixels.size(0)
        device = pixels.device
        # Initialise state
        if prev_state is None:
            prev_state = {}
        h = prev_state.get("h")
        workspace = prev_state.get(
            "workspace", torch.zeros(B, self.cfg.workspace.dim, device=device)
        )
        emotion = prev_state.get(
            "emotion", torch.zeros(B, self.cfg.emotion.num_channels, device=device)
        )
        # Encode pixels using the image encoder
        embed = self.encoder(pixels)
        # Retrieve memory
        mem_value, attn = self.pmm(embed)
        # Update FRNN
        frnn_input = embed.unsqueeze(1)  # (B,1,input_dim)
        frnn_out, h_n = self.frnn(frnn_input, h)
        frnn_vec = frnn_out[:, 0, :]  # (B, output_dim)
        # Update workspace
        # Compute ToneNet features if audio is provided
        if audio is not None:
            # audio: (B, T) – produce glyph logits (B, N, vocab_size)
            glyph_logits = self.tonenet(audio)
            # Summarise by mean over time dimension to get (B, vocab_size)
            tone_vec = glyph_logits.mean(dim=1)
        else:
            tone_vec = torch.zeros(B, self._tone_dim, device=device)
        gw_in = torch.cat([frnn_vec, mem_value, emotion, tone_vec], dim=-1)
        workspace, h_ws = self.workspace_engine(gw_in, workspace, prev_state.get("h_ws"))
        # Update emotion
        emo_in = torch.cat([workspace, frnn_vec], dim=-1)
        emotion = self.emotion_engine(emo_in, emotion)
        # Refine workspace via capsules
        workspace_refined = self.capsule_net(workspace)
        # Compute raw skill logits from the action capsule
        raw_skill_logits = self.action_capsule(workspace_refined)
        # Compute meta controller logits; currently no extra context
        meta_logits = self.meta_controller(workspace_refined, None)
        # Combine logits from action capsule and meta controller.  This
        # simple addition encourages skills preferred by both modules.  You
        # can experiment with alternative combination strategies (e.g.
        # multiplication, gating via softmax) to adjust how strongly the
        # meta controller influences selection.
        combined_logits = raw_skill_logits + meta_logits
        # Apply safety filtering on the combined logits, passing the environment
        skill_logits = self.safety.filter_action(combined_logits, env)
        # Determine which skill to execute.  If an explicit environment is
        # provided (e.g. from the API), we force the selection to that
        # skill.  Otherwise select the index with the largest logit.
        if env is not None and env in self.skills:
            target_idx = list(self.skills.keys()).index(env)
            skill_idx = torch.full(
                (B,), target_idx, dtype=torch.long, device=workspace.device
            )
        else:
            skill_idx = torch.argmax(skill_logits, dim=-1)
        # For each batch element call the appropriate skill
        actions = []
        skill_outputs = {}
        for b in range(B):
            sidx = skill_idx[b].item()
            skill_name = list(self.skills.keys())[sidx]
            skill = self.skills[skill_name]
            out = skill(workspace_refined[b : b + 1])
            if isinstance(out, dict):
                # Expect at least 'action_logits'
                actions.append(out.get("action_logits"))
                # Collect other outputs keyed by skill name; only keep tensor values
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        skill_outputs.setdefault(k, []).append(v)
            else:
                actions.append(out)
        # Stack actions
        action_logits = torch.cat(actions, dim=0)
        # Memory update
        with torch.no_grad():
            self.pmm.update(embed, embed)
        # Update self‑model with the current workspace
        self.self_model.update(workspace)
        return {
            "action_logits": action_logits,
            "workspace": workspace,
            "emotion": emotion,
            "h": h_n,
            "h_ws": h_ws,
            "memory_pressure": torch.tensor(self.pmm.compute_pressure()),
            **{k: torch.cat(v, dim=0) for k, v in skill_outputs.items()},
        }
