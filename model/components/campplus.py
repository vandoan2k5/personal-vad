from __future__ import annotations

from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as functional
import torchaudio


class CAMPPlusProfileEncoder(nn.Module):
    """Frozen local CAM++ encoder used once for enrollment."""

    def __init__(self, model_dir: str | Path, device: str | torch.device = "cpu") -> None:
        super().__init__()
        import yaml
        from model.pretrained.campplus import CAMPPlus

        model_dir = Path(model_dir)
        with (model_dir / "config.yaml").open() as handle:
            model_config = yaml.safe_load(handle)["model_conf"]
        self.device_name = str(device)
        self.encoder = CAMPPlus(**{key: value for key, value in model_config.items() if key != "output_level"})
        checkpoint = torch.load(model_dir / "campplus_cn_common.bin", map_location=device, weights_only=True)
        self.encoder.load_state_dict(checkpoint, strict=True)
        self.encoder.to(device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.embedding_dim = int(model_config["embedding_size"])
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16_000, n_fft=400, win_length=400, hop_length=160, n_mels=80, f_min=20.0, f_max=8_000.0, power=2.0, center=False).to(device)

    @torch.inference_mode()
    def encode(self, waveforms: Tensor) -> Tensor:
        if waveforms.ndim == 3:
            waveforms = waveforms.mean(dim=1)
        waveforms = waveforms.to(self.device_name)
        if waveforms.size(1) < 400:
            waveforms = functional.pad(waveforms, (0, 400 - waveforms.size(1)))
        features = self.mel(waveforms).clamp_min(1e-10).log().transpose(1, 2)
        embeddings = self.encoder(features)
        return functional.normalize(embeddings, dim=-1)
