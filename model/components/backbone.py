from __future__ import annotations

from torch import Tensor, nn


class StreamingGRUBackbone(nn.Module):
    """Small causal acoustic encoder; caller owns its recurrent state."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 192, layers: int = 2) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        encoded, state = self.gru(self.input_projection(features), state)
        return self.output_norm(encoded), state
