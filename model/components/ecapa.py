from __future__ import annotations

from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as functional
import torchaudio

# Must match ``lin_neurons`` in ``model/pretrained/ecapa/hyperparams.yaml``.
ECAPA_EMBEDDING_DIM = 192


class ECAPAProfileEncoder(nn.Module):
    """Frozen SpeechBrain ECAPA-TDNN encoder used once for enrollment.

    Pipeline mirrors ``model/pretrained/ecapa/hyperparams.yaml``:
    log-Mel features -> sentence mean norm (no std) -> ECAPA-TDNN ->
    global embedding mean norm (no std) -> L2 normalize.
    """

    def __init__(self, model_dir: str | Path, device: str | torch.device = "cpu") -> None:
        super().__init__()
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

        model_dir = Path(model_dir)
        self.device_name = str(device)
        self.encoder = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=ECAPA_EMBEDDING_DIM,
        )
        checkpoint = torch.load(model_dir / "embedding_model.ckpt", map_location=device, weights_only=True)
        self.encoder.load_state_dict(checkpoint, strict=True)
        self.encoder.to(device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        norm_checkpoint = torch.load(model_dir / "mean_var_norm_emb.ckpt", map_location=device, weights_only=False)
        self.register_buffer("emb_mean", torch.as_tensor(norm_checkpoint["glob_mean"], dtype=torch.float32))
        self.embedding_dim = ECAPA_EMBEDDING_DIM
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16_000, n_fft=400, win_length=400, hop_length=160, n_mels=80, f_min=20.0, f_max=8_000.0, power=2.0, center=False).to(device)

    @torch.inference_mode()
    def encode(self, waveforms: Tensor) -> Tensor:
        if waveforms.ndim == 3:
            waveforms = waveforms.mean(dim=1)
        waveforms = waveforms.to(self.device_name)
        if waveforms.size(1) < 400:
            waveforms = functional.pad(waveforms, (0, 400 - waveforms.size(1)))
        features = self.mel(waveforms).clamp_min(1e-10).log().transpose(1, 2)
        features = features - features.mean(dim=1, keepdim=True)
        embeddings = self.encoder(features).squeeze(1)
        embeddings = embeddings - self.emb_mean.to(embeddings.device)
        return functional.normalize(embeddings, dim=-1)
