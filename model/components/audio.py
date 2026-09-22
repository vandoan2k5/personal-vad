from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torchaudio


@dataclass
class FrontendState:
    samples: Tensor


class LogMelFrontend(nn.Module):
    """Chunk-invariant causal log-Mel frontend."""

    def __init__(self, sample_rate: int = 16_000, n_mels: int = 80) -> None:
        super().__init__()
        self.sample_rate, self.hop_length = sample_rate, 160
        self.window_length = 400
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=400, win_length=400, hop_length=160, n_mels=n_mels, f_min=20.0, f_max=sample_rate / 2, power=2.0, center=False)
        self.norm = nn.LayerNorm(n_mels)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> FrontendState:
        return FrontendState(torch.empty(batch_size, 0, device=device, dtype=dtype))

    def forward(self, waveforms: Tensor, state: FrontendState | None = None) -> tuple[Tensor, FrontendState]:
        if waveforms.ndim == 3:
            waveforms = waveforms.mean(dim=1)
        if state is None:
            state = self.init_state(waveforms.size(0), waveforms.device, waveforms.dtype)
        samples = torch.cat((state.samples.to(waveforms), waveforms), dim=1)
        if samples.size(1) < self.window_length:
            return waveforms.new_empty(waveforms.size(0), 0, self.norm.normalized_shape[0]), FrontendState(samples)
        frame_count = 1 + (samples.size(1) - self.window_length) // self.hop_length
        framed = samples[:, : frame_count * self.hop_length + self.window_length - self.hop_length]
        features = self.mel(framed).clamp_min(1e-10).log().transpose(1, 2)
        remainder = samples[:, frame_count * self.hop_length :]
        return self.norm(features), FrontendState(remainder)

    def forward_offline(self, waveforms: Tensor) -> Tensor:
        if waveforms.ndim == 3:
            waveforms = waveforms.mean(dim=1)
        features = self.mel(waveforms).clamp_min(1e-10).log().transpose(1, 2)
        return self.norm(features)
