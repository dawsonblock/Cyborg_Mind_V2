"""
Configuration schemas for the Capsule Memory Engine.

The `EngineConfig` dataclass aggregates nested configurations for the encoder,
pseudo‑mode memory (PMM), recurrent neural network (FRNN) and the PPO
controller.  Each sub‑configuration is itself a dataclass to aid in type
checking and clarity.
"""

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """Configuration for the global encoder."""

    d_input: int = 128
    d_global: int = 60
    hidden_dim: int = 128


@dataclass
class PMMConfig:
    """Configuration for the pseudo‑mode memory."""

    dim: int = 60
    num_modes: int = 32
    decay: float = 0.99


@dataclass
class FRNNConfig:
    """Configuration for the recurrent neural network core."""

    input_dim: int = 60
    hidden_dim: int = 64
    output_dim: int = 64
    num_layers: int = 1


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO controller."""

    hidden_dim: int = 128
    num_actions: int = 4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4


@dataclass
class EngineConfig:
    """Top‑level engine configuration aggregating all sub‑configs."""

    encoder: EncoderConfig = EncoderConfig()
    pmm: PMMConfig = PMMConfig()
    frnn: FRNNConfig = FRNNConfig()
    ppo: PPOConfig = PPOConfig()