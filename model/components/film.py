from __future__ import annotations

import torch
from torch import Tensor, nn


class SpeakerFiLM(nn.Module):
    """Identity-conditioned scale/shift, initialized as identity."""

    def __init__(self, feature_dim: int, speaker_dim: int) -> None:
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(speaker_dim + 1, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim * 2))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, features: Tensor, cosine_scores: Tensor, target_embedding: Tensor) -> Tensor:
        target = target_embedding.unsqueeze(1).expand(-1, features.size(1), -1)
        scale, shift = self.condition(torch.cat((cosine_scores.unsqueeze(-1), target), dim=-1)).chunk(2, dim=-1)
        return (1.0 + scale) * features + shift


class CausalSpeakerPrenet(nn.Module):
    """Causal frame-level speaker representation."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.gru = nn.GRU(hidden_dim, embedding_dim, batch_first=True)

    def forward(self, features: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        encoded, state = self.gru(self.input(features), state)
        return torch.nn.functional.normalize(encoded, dim=-1), state
