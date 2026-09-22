from __future__ import annotations

from torch import Tensor, nn
import torch
import torch.nn.functional as functional


class WeightedPairwiseLoss(nn.Module):
    """Cost-sensitive CE where tss/ntss confusion receives the strongest penalty."""

    def __init__(self, tss_ntss_weight: float = 1.0, ns_ntss_weight: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("pair_weights", torch.tensor([[0.0, tss_ntss_weight, 0.5], [tss_ntss_weight, 0.0, ns_ntss_weight], [0.5, ns_ntss_weight, 0.0]]))

    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        flat_logits, flat_targets = logits.reshape(-1, 3), targets.reshape(-1)
        if mask is not None:
            keep = mask.reshape(-1).bool()
            flat_logits, flat_targets = flat_logits[keep], flat_targets[keep]
        ce = functional.cross_entropy(flat_logits, flat_targets, reduction="none")
        penalty = (self.pair_weights[flat_targets] * flat_logits.softmax(dim=-1)).sum(dim=-1)
        return ((1.0 + penalty) * ce).mean()
