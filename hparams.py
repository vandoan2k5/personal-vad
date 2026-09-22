"""Central configuration for Personal-VAD training.

Edit this file for normal experiments. Command-line arguments in ``train.py``
remain available as one-off overrides.
"""

from dataclasses import dataclass, field
import torch


@dataclass
class DataConfig:
    audio_root: str = "data/audio_with_id"
    noise_root: str = "data/noise"
    # FSD50K clips carrying these human-vocal labels are not used as noise.
    excluded_noise_labels: tuple[str, ...] = ("Human voice", "Speech", "Singing", "Respiratory_sounds")
    sample_rate: int = 16_000
    duration_seconds: float = 30.0
    workers: int = 3
    gender_metadata: str = "data/audio_with_id_gender.json"
    # Frame quotas are (tss, ntss, ns). L1 has no overlap so quotas are exact.
    # L2/L3 overlap: TSS keeps its quota (wins overlap), NS is the remainder, so the
    # NTSS quota is raised by the expected overlap to keep dominant labels ~= 1/3 each
    # (measured L2 [0.333, 0.325, 0.341], L3 [0.333, 0.327, 0.340]).
    level_ratios: tuple[tuple[float, float, float], ...] = ((1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 2, 1 / 6), (1 / 3, 4 / 7, 2 / 21))
    # Hierarchical soft labels for TSS+NTSS overlap frames: (tss, ntss, ns).
    # Single-class frames stay one-hot; overlap frames use (soft_dominant, soft_secondary, 0).
    soft_dominant: float = 0.9
    soft_secondary: float = 0.1
    # Default fraction of samples generated at levels 1, 2, and 3.
    level_sampling_ratios: tuple[float, float, float] = (0.20, 0.40, 0.40)
    # Curriculum entries are (first training step, (L1, L2, L3) ratios).
    # Easy-to-hard: L1 first, then shift to overlap levels. Together with balanced
    # labels and rising overlap (0 -> 0.25 -> 0.90) this orders difficulty L1 < L2 < L3,
    # i.e. expected macro-F1 L1 > L2 > L3.
    curriculum: dict[int, tuple[float, float, float]] = field(default_factory=lambda: {
        0: (0.60, 0.30, 0.10),
        8_000: (0.40, 0.40, 0.20),
        16_000: (0.25, 0.45, 0.30),
        26_000: (0.20, 0.40, 0.40),
    })
    steps_per_epoch: int = 2_000


@dataclass
class ModelConfig:
    n_mels: int = 80
    hidden_dim: int = 192
    speaker_prenet_hidden_dim: int = 96
    gru_layers: int = 2
    # Chon encoder enrollment: "campplus" hay "ecapa" (doi speaker_dir tuong ung).
    speaker_encoder: str = "campplus"
    speaker_dir: str = "model/pretrained/campplus"


@dataclass
class LossConfig:
    tss_ntss_weight: float = 1.0
    ns_ntss_weight: float = 0.1
    # Auxiliary multi-task weights (0 disables the corresponding head loss).
    cosine_weight: float = 0.5
    overlap_weight: float = 0.5
    vad_weight: float = 0.3
    # NTSS frames push cosine below this margin (hinge); TSS frames pull to +1.
    cosine_margin: float = 0.2
    # Overlap frames are ~15% of L2/L3; upweight positives in BCE.
    overlap_pos_weight: float = 5.0


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_gradient_norm: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output: str = "checkpoints/pvad.pt"
    seed: int = 42
    log_every_steps: int = 1
    eval_every_steps: int = 5_000
    checkpoint_every_steps: int = 5_000


@dataclass
class EvalConfig:
    batches_per_level: int = 20
    batch_size: int = 8


@dataclass
class HParams:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


HPARAMS = HParams()
