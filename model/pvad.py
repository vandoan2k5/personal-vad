from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as functional
from .components import CausalSpeakerPrenet, CAMPPlusProfileEncoder, ECAPAProfileEncoder, LogMelFrontend, SpeakerFiLM, StreamingGRUBackbone

TSS, NTSS, NS = 0, 1, 2
CLASS_NAMES = ("tss", "ntss", "ns")


@dataclass
class PVADConfig:
    sample_rate: int = 16_000
    n_mels: int = 80
    hidden_dim: int = 192
    speaker_prenet_hidden_dim: int = 96
    gru_layers: int = 2
    # "campplus" -> model/pretrained/campplus, "ecapa" -> model/pretrained/ecapa.
    speaker_encoder: str = "campplus"
    speaker_dir: str = "model/pretrained/campplus"


def build_speaker_encoder(name: str, model_dir: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Select the frozen enrollment encoder by name."""
    if name == "campplus":
        return CAMPPlusProfileEncoder(model_dir, device)
    if name == "ecapa":
        return ECAPAProfileEncoder(model_dir, device)
    raise ValueError(f"unknown speaker_encoder={name!r}, expected 'campplus' or 'ecapa'")


class PersonalVAD(nn.Module):
    """Causal ternary PVAD with CAM++ enrollment and speaker pre-net matching."""

    def __init__(self, config: PVADConfig = PVADConfig(), device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.config = config
        self.frontend = LogMelFrontend(config.sample_rate, config.n_mels)
        self.speaker_encoder = build_speaker_encoder(config.speaker_encoder, config.speaker_dir, device)
        self.backbone = StreamingGRUBackbone(config.n_mels, config.hidden_dim, config.gru_layers)
        self.speaker_prenet = CausalSpeakerPrenet(config.n_mels, config.speaker_prenet_hidden_dim, self.speaker_encoder.embedding_dim)
        self.film = SpeakerFiLM(config.hidden_dim, self.speaker_encoder.embedding_dim)
        self.classifier = nn.Linear(config.hidden_dim, 3)
        # Auxiliary heads (all causal, per-frame linear):
        # - overlap_head reads the conditioned features (same input as classifier).
        # - vad_head reads the pre-FiLM acoustic features (speaker-independent VAD).
        self.overlap_head = nn.Linear(config.hidden_dim, 1)
        self.vad_head = nn.Linear(config.hidden_dim, 1)

    @torch.inference_mode()
    def enroll(self, enrollment: Tensor) -> Tensor:
        return self.speaker_encoder.encode(enrollment)

    def init_stream(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> dict[str, object]:
        return {
            "frontend": self.frontend.init_state(batch_size, device, dtype),
            "backbone": None,
            "speaker_prenet": None,
        }

    def stream_step(self, stream: Tensor, target_embedding: Tensor, state: dict[str, object]) -> tuple[dict[str, Tensor], dict[str, object]]:
        features, frontend_state = self.frontend(stream, state["frontend"])
        next_state = dict(state)
        next_state["frontend"] = frontend_state
        if features.size(1) == 0:
            empty = stream.new_empty(stream.size(0), 0)
            return {"logits": stream.new_empty(stream.size(0), 0, 3), "cosine": empty, "overlap_logits": empty, "vad_logits": empty}, next_state
        speaker_frames, speaker_state = self.speaker_prenet(features, state["speaker_prenet"])
        acoustic, backbone_state = self.backbone(features, state["backbone"])
        scores = functional.cosine_similarity(speaker_frames, target_embedding.unsqueeze(1), dim=-1)
        conditioned = self.film(acoustic, scores, target_embedding)
        logits = self.classifier(conditioned)
        overlap_logits = self.overlap_head(conditioned).squeeze(-1)
        vad_logits = self.vad_head(acoustic).squeeze(-1)
        next_state["speaker_prenet"] = speaker_state
        next_state["backbone"] = backbone_state
        return {"logits": logits, "cosine": scores, "overlap_logits": overlap_logits, "vad_logits": vad_logits}, next_state

    def forward(self, stream: Tensor, target_embedding: Tensor, state: Tensor | None = None) -> dict[str, Tensor]:
        features = self.frontend.forward_offline(stream)
        speaker_frames, speaker_state = self.speaker_prenet(features, None)
        acoustic, backbone_state = self.backbone(features, state)
        scores = functional.cosine_similarity(speaker_frames, target_embedding.unsqueeze(1), dim=-1)
        conditioned = self.film(acoustic, scores, target_embedding)
        return {"logits": self.classifier(conditioned), "overlap_logits": self.overlap_head(conditioned).squeeze(-1), "vad_logits": self.vad_head(acoustic).squeeze(-1), "cosine": scores, "state": backbone_state, "speaker_state": speaker_state}

    def train(self, mode: bool = True) -> "PersonalVAD":
        super().train(mode)
        self.speaker_encoder.eval()
        return self

    def checkpoint(self) -> dict:
        state_dict = {name: value for name, value in self.state_dict().items() if not name.startswith("speaker_encoder.")}
        return {"config": asdict(self.config), "state_dict": state_dict}

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "PersonalVAD":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(PVADConfig(**checkpoint["config"]), device=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        return model
