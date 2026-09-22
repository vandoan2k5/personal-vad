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
    # Frame ratios are (tss, ntss, ns). Easy samples contain all three classes.
    level_ratios: tuple[tuple[float, float, float], ...] = ((1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3))
    # Default fraction of samples generated at levels 1, 2, and 3.
    level_sampling_ratios: tuple[float, float, float] = (0.20, 0.40, 0.40)
    # Curriculum entries are (first training step, (L1, L2, L3) ratios).
    curriculum: dict[int, tuple[float, float, float]] = field(default_factory=lambda: {
        0: (0.3, 0.4, 0.3),
        5_000: (0.3, 0.4, 0.3),
        10_000: (0.3, 0.4, 0.3),
        20_000: (0.3, 0.4, 0.3),
    })
    steps_per_epoch: int = 2_000


@dataclass
class ModelConfig:
    n_mels: int = 80
    hidden_dim: int = 192
    speaker_prenet_hidden_dim: int = 96
    gru_layers: int = 2
    speaker_dir: str = "model/pretrained/campplus"


@dataclass
class LossConfig:
    tss_ntss_weight: float = 1.0
    ns_ntss_weight: float = 0.1


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
