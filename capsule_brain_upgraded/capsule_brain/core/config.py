"""
Configuration dataclasses for the Capsule Brain subsystems.

The Capsule Brain is composed of many independent modules; each module
exposes a configuration dataclass that controls its behaviour.  Central
configuration objects may be composed into a single top‑level
``CapsuleBrainConfig`` to simplify instantiation of the full system.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PMMConfig:
    """Hyperparameters for the pseudo‑mode memory.

    :param num_modes: Number of memory slots available at initialisation.
    :param dim: Dimensionality of each memory slot.
    :param decay: Exponential decay factor applied to running counts when
        updating memory.  A value in ``(0,1)``; smaller values favour
        faster forgetting.
    :param max_modes: Maximum number of slots allowed when the memory
        automatically expands.  When ``None`` the memory is fixed.
    """

    num_modes: int = 64
    dim: int = 128
    decay: float = 0.9
    max_modes: Optional[int] = 1024


@dataclass
class FRNNConfig:
    """Configuration for the Fractional Recurrent Neural Network (FRNN).

    The FRNN is implemented using a standard GRU; these parameters
    correspond to the input dimensionality, hidden state size and number
    of layers.
    """

    input_dim: int = 256
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 1


@dataclass
class WorkspaceConfig:
    """Hyperparameters for the Global Workspace Engine.

    :param dim: Size of the workspace vector.
    :param hidden_dim: Hidden dimension used in the GRU that updates
        workspace state.
    :param num_layers: Number of GRU layers.
    """

    dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1


@dataclass
class EmotionConfig:
    """Configuration for the emotion engine.

    Emotions are modelled in a continuous valence–arousal–dominance
    (VAD) space【93681059298509†L51-L83】.  The emotion engine predicts updates
    to this VAD vector given the current workspace and other features.
    """

    num_channels: int = 3  # VAD
    hidden_dim: int = 64


@dataclass
class CapsuleConfig:
    """Configuration for the capsule processor network.

    :param input_dim: Dimensionality of the input to the capsule layer.
    :param num_capsules: Number of capsules in the layer.
    :param capsule_dim: Dimensionality of each capsule.
    :param num_routes: Number of dynamic routing iterations.
    """

    input_dim: int = 256
    num_capsules: int = 8
    capsule_dim: int = 32
    num_routes: int = 3


@dataclass
class ToneNetConfig:
    """Configuration for the ToneNet audio processing network.

    :param sample_rate: Sampling rate of input audio.
    :param n_mels: Number of Mel filterbank channels used to compute
        spectrograms.  Mel spectrograms are used as intermediate
        representations before feeding into a neural network to
        produce glyphs.
    """

    sample_rate: int = 16000
    n_mels: int = 80


@dataclass
class CapsuleBrainConfig:
    """Top level configuration bundling all subcomponent configs.

    This object is passed to ``CapsuleBrain`` to build the complete
    system.  Default values provide a reasonable starting point for
    research; they can be overridden via keyword arguments or by
    manually constructing subconfigs.
    """

    pmm: PMMConfig = field(default_factory=PMMConfig)
    frnn: FRNNConfig = field(default_factory=FRNNConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    capsule: CapsuleConfig = field(default_factory=CapsuleConfig)
    tonenet: ToneNetConfig = field(default_factory=ToneNetConfig)
