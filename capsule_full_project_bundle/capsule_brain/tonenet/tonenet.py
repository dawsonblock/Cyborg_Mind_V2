"""ToneNet audio → glyph conversion.

The ToneNet converts raw audio waveforms into a sequence of high level
glyphs suitable for symbolic processing.  It first computes a
Mel‑spectrogram and then projects the time–frequency representation
into a discrete vocabulary using a simple convolutional encoder.

This module serves as a placeholder; for production use the encoder
should be replaced with a pre‑trained speech representation model such
as wav2vec or whisper.  The output glyphs can be fed into the
Capsule Brain symbolic subsystem or used to update the workspace.
"""

from typing import List

import torch
import torch.nn as nn

try:
    import torchaudio
except ImportError:
    torchaudio = None

from ..core.config import ToneNetConfig


class ToneNet(nn.Module):
    """Audio feature extractor and glyph encoder."""

    def __init__(self, cfg: ToneNetConfig, vocab_size: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        # Simple convolutional encoder mapping spectrograms to logits
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, vocab_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to a sequence of glyph logits.

        Args:
            audio: Tensor of shape ``(B, T)`` containing raw waveforms.
        Returns:
            Logits over the glyph vocabulary of shape ``(B, N, vocab_size)``.
        """
        if torchaudio is None:
            raise RuntimeError("torchaudio is required for ToneNet but not installed")
        # Compute Mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_mels=self.cfg.n_mels,
        )(audio)  # (B, n_mels, time)
        mel_spec = mel_spec.unsqueeze(1)  # (B,1,n_mels,time)
        feats = self.encoder(mel_spec)  # (B,C,H,W)
        # Pool spatial dimensions to get per‑frame features
        feats = feats.mean(dim=[2])  # (B,C,W)
        logits = self.classifier(feats.transpose(1, 2))  # (B,W,vocab_size)
        return logits

    def decode(self, logits: torch.Tensor) -> List[List[int]]:
        """Convert glyph logits to integer sequences using argmax decoding."""
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist()
