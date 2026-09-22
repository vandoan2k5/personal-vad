from __future__ import annotations

from torch import Tensor, nn
import torch
import torch.nn.functional as functional


class WeightedPairwiseLoss(nn.Module):
    """Cost-sensitive CE where tss/ntss confusion receives the strongest penalty.

    Supports hard labels ``[B, T]`` (long) and hierarchical soft labels
    ``[B, T, 3]`` (float, rows sum to 1). Soft CE is the expected CE under the
    label distribution; the pairwise term is the expected pairwise cost.
    """

    def __init__(self, tss_ntss_weight: float = 1.0, ns_ntss_weight: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("pair_weights", torch.tensor([[0.0, tss_ntss_weight, 0.5], [tss_ntss_weight, 0.0, ns_ntss_weight], [0.5, ns_ntss_weight, 0.0]]))

    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        flat_logits = logits.reshape(-1, 3)
        if targets.dtype == torch.long and targets.dim() == logits.dim() - 1:
            flat_targets = targets.reshape(-1)
            target_distr = functional.one_hot(flat_targets, num_classes=3).float()
        else:
            target_distr = targets.reshape(-1, 3).float()
            target_distr = target_distr / target_distr.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if mask is not None:
            keep = mask.reshape(-1).bool()
            flat_logits, target_distr = flat_logits[keep], target_distr[keep]
        log_probs = functional.log_softmax(flat_logits, dim=-1)
        probs = log_probs.exp()
        ce = -(target_distr * log_probs).sum(dim=-1)
        penalty = ((target_distr @ self.pair_weights) * probs).sum(dim=-1)
        return ((1.0 + penalty) * ce).mean()


def to_label_distribution(targets: Tensor) -> Tensor:
    """Hard ``[..., T]`` long or soft ``[..., T, 3]`` float -> ``[..., 3]`` float rows summing to 1."""
    if targets.dtype == torch.long:
        distr = functional.one_hot(targets.long(), num_classes=3).float()
    else:
        distr = targets.float()
    return distr / distr.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class PVADMultitaskLoss(nn.Module):
    """Main PVAD loss plus lightweight auxiliary supervision.

    - ``cosine``: frame speaker embedding is pulled toward +1 on TSS, -1 on
      NTSS (``target = p_tss - p_ntss``), scored with MSE on speech frames only.
      NS frames carry no speaker identity and are ignored.
    - ``overlap_logits``: BCE for TSS+NTSS overlap frames. With soft labels an
      overlap frame is exactly a fractional row (e.g. ``[0.9, 0.1, 0]``); with
      hard labels the target is all zeros.
    - ``vad_logits``: BCE for speech (TSS or NTSS) vs NS, read from the
      pre-FiLM acoustic features.
    """

    def __init__(self, tss_ntss_weight: float = 1.0, ns_ntss_weight: float = 0.1, cosine_weight: float = 0.3, overlap_weight: float = 0.5, vad_weight: float = 0.3, overlap_pos_weight: float = 5.0) -> None:
        super().__init__()
        self.pvad = WeightedPairwiseLoss(tss_ntss_weight, ns_ntss_weight)
        self.cosine_weight, self.overlap_weight, self.vad_weight = float(cosine_weight), float(overlap_weight), float(vad_weight)
        self.register_buffer("overlap_pos_weight", torch.tensor(float(overlap_pos_weight)))
        self.last_parts: dict[str, float] = {}

    def forward_with_parts(self, outputs: dict[str, Tensor], targets: Tensor, mask: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        logits, cosine = outputs["logits"], outputs["cosine"]
        overlap_logits, vad_logits = outputs["overlap_logits"], outputs["vad_logits"]
        main = self.pvad(logits, targets, mask)
        distr = to_label_distribution(targets)
        if mask is not None:
            keep = mask.reshape(-1).bool()
            flat_cosine, flat_overlap, flat_vad = cosine.reshape(-1)[keep], overlap_logits.reshape(-1)[keep], vad_logits.reshape(-1)[keep]
            flat_distr = distr.reshape(-1, 3)[keep]
        else:
            flat_cosine, flat_overlap, flat_vad = cosine.reshape(-1), overlap_logits.reshape(-1), vad_logits.reshape(-1)
            flat_distr = distr.reshape(-1, 3)
        overlap_target = ((flat_distr[:, 0] > 1e-3) & (flat_distr[:, 0] < 1.0 - 1e-3) & (flat_distr[:, 1] > 1e-3)).float()
        speech_target = 1.0 - flat_distr[:, 2]
        cosine_target = flat_distr[:, 0] - flat_distr[:, 1]
        pos_weight = self.overlap_pos_weight.to(flat_overlap.device)
        loss_overlap = functional.binary_cross_entropy_with_logits(flat_overlap, overlap_target, pos_weight=pos_weight)
        loss_vad = functional.binary_cross_entropy_with_logits(flat_vad, speech_target)
        speech_mask = speech_target > 0.5
        if bool(speech_mask.any()):
            loss_cosine = functional.mse_loss(flat_cosine[speech_mask], cosine_target[speech_mask])
        else:
            loss_cosine = flat_cosine.sum() * 0.0
        total = main + self.overlap_weight * loss_overlap + self.vad_weight * loss_vad + self.cosine_weight * loss_cosine
        parts = {"pvad": main.detach(), "overlap": loss_overlap.detach(), "vad": loss_vad.detach(), "cosine": loss_cosine.detach()}
        self.last_parts = {key: value.item() for key, value in parts.items()}
        return total, parts

    def forward(self, outputs: dict[str, Tensor], targets: Tensor, mask: Tensor | None = None) -> Tensor:
        total, _ = self.forward_with_parts(outputs, targets, mask)
        return total
