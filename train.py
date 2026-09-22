from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PVADMixtureDataset, collate_pvad
from hparams import HPARAMS
from model import PersonalVAD, PVADConfig
from model.components import PVADMultitaskLoss


def make_dataset(samples: int, level: int | None, seed: int, batch_size: int = 1, curriculum: dict[int, tuple[float, float, float]] | None = None) -> PVADMixtureDataset:
    return PVADMixtureDataset(
        HPARAMS.data.audio_root,
        HPARAMS.data.noise_root,
        samples,
        HPARAMS.data.duration_seconds,
        HPARAMS.data.sample_rate,
        seed,
        level,
        HPARAMS.data.gender_metadata,
        HPARAMS.data.level_ratios,
        HPARAMS.data.level_sampling_ratios,
        batch_size,
        curriculum,
        HPARAMS.data.excluded_noise_labels,
        HPARAMS.data.soft_dominant,
        HPARAMS.data.soft_secondary,
    )


def crop_outputs(outputs: dict[str, Tensor], frame_count: int) -> dict[str, Tensor]:
    """Crop time-major outputs to ``frame_count``; scalars/states pass through."""
    cropped: dict[str, Tensor] = {}
    for key, value in outputs.items():
        if isinstance(value, Tensor) and value.dim() >= 2 and value.size(1) == outputs["logits"].size(1):
            cropped[key] = value[:, :frame_count]
        else:
            cropped[key] = value
    return cropped


def dominant_labels(labels: Tensor) -> Tensor:
    """Dominant (hard) labels for accuracy/confusion; soft [B, T, 3] -> [B, T]."""
    return labels.argmax(dim=-1) if labels.dim() == 3 and labels.size(-1) == 3 else labels


def accuracy(logits: Tensor, labels: Tensor) -> float:
    return (logits.argmax(dim=-1) == dominant_labels(labels)).float().mean().item()


@torch.inference_mode()
def evaluate(model: PersonalVAD, device: torch.device, criterion: PVADMultitaskLoss, global_step: int) -> dict[int, dict[str, float]]:
    """Evaluate independent, forced samples for every curriculum level."""
    was_training = model.training
    model.eval()
    metrics: dict[int, dict[str, float]] = {}
    for level in (1, 2, 3):
        dataset = make_dataset(HPARAMS.evaluation.batches_per_level * HPARAMS.evaluation.batch_size, level, HPARAMS.train.seed + global_step + level, HPARAMS.evaluation.batch_size)
        loader = DataLoader(dataset, batch_size=HPARAMS.evaluation.batch_size, collate_fn=collate_pvad, num_workers=HPARAMS.data.workers, pin_memory=device.type == "cuda")
        correct = total = 0
        loss_total = 0.0
        for batch in loader:
            stream, enrollment, labels = (batch[key].to(device) for key in ("stream", "enrollment", "labels"))
            target_embedding = model.enroll(enrollment)
            outputs = model(stream, target_embedding)
            frame_count = min(outputs["logits"].size(1), labels.size(1))
            outputs, labels = crop_outputs(outputs, frame_count), labels[:, :frame_count]
            loss_total += criterion(outputs, labels).item()
            correct += (outputs["logits"].argmax(dim=-1) == dominant_labels(labels)).sum().item()
            total += dominant_labels(labels).numel()
        metrics[level] = {"loss": loss_total / len(loader), "accuracy": correct / total}
    if was_training:
        model.train()
    return metrics


def save_checkpoint(model: PersonalVAD, optimizer: torch.optim.Optimizer, global_step: int, output: str) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_name(f"{output_path.stem}_iter_{global_step:07d}{output_path.suffix}")
    checkpoint = model.checkpoint()
    checkpoint.update({"optimizer": optimizer.state_dict(), "global_step": global_step})
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, output_path)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=HPARAMS.train.epochs)
    parser.add_argument("--steps-per-epoch", type=int, default=HPARAMS.data.steps_per_epoch)
    parser.add_argument("--batch-size", type=int, default=HPARAMS.train.batch_size)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=None, help="Force one mixer difficulty level; default uses level_sampling_ratios curriculum.")
    parser.add_argument("--device", default=HPARAMS.train.device)
    parser.add_argument("--output", default=HPARAMS.train.output)
    parser.add_argument(
        "--resume",
        default="",
        help="Checkpoint to resume, or an empty string to start from scratch.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(HPARAMS.train.seed)
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    logger = logging.getLogger("pvad")

    dataset = make_dataset(args.steps_per_epoch * args.epochs * args.batch_size, args.level, HPARAMS.train.seed, args.batch_size, HPARAMS.data.curriculum)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_pvad, num_workers=HPARAMS.data.workers, pin_memory=device.type == "cuda")
    config = PVADConfig(
        sample_rate=HPARAMS.data.sample_rate,
        n_mels=HPARAMS.model.n_mels,
        hidden_dim=HPARAMS.model.hidden_dim,
        speaker_prenet_hidden_dim=HPARAMS.model.speaker_prenet_hidden_dim,
        gru_layers=HPARAMS.model.gru_layers,
        speaker_encoder=HPARAMS.model.speaker_encoder,
        speaker_dir=HPARAMS.model.speaker_dir,
    )
    model = PersonalVAD(config, device=device).to(device)
    criterion = PVADMultitaskLoss(HPARAMS.loss.tss_ntss_weight, HPARAMS.loss.ns_ntss_weight, HPARAMS.loss.cosine_weight, HPARAMS.loss.overlap_weight, HPARAMS.loss.vad_weight, HPARAMS.loss.overlap_pos_weight, HPARAMS.loss.cosine_margin).to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=HPARAMS.train.learning_rate, weight_decay=HPARAMS.train.weight_decay)
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        logger.info("resumed checkpoint=%s at iter=%d", resume_path, global_step)
    model.train()
    progress = tqdm(loader, total=args.epochs * args.steps_per_epoch, unit="iter")
    for local_step, batch in enumerate(progress, start=1):
        global_step = global_step + 1
        stream, enrollment, labels = (batch[key].to(device) for key in ("stream", "enrollment", "labels"))
        with torch.no_grad():
            target_embedding = model.enroll(enrollment)
        outputs = model(stream, target_embedding)
        frame_count = min(outputs["logits"].size(1), labels.size(1))
        outputs, labels = crop_outputs(outputs, frame_count), labels[:, :frame_count]
        loss, parts = criterion.forward_with_parts(outputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS.train.max_gradient_norm)
        optimizer.step()
        step_accuracy = accuracy(outputs["logits"].detach(), labels)
        progress.set_postfix(iter=global_step, loss=f"{loss.item():.4f}", accuracy=f"{step_accuracy:.4f}")
        if global_step % HPARAMS.train.log_every_steps == 0:
            logger.info("iter=%d loss=%.5f accuracy=%.4f pvad=%.5f overlap=%.5f vad=%.5f cosine=%.5f", global_step, loss.item(), step_accuracy, parts["pvad"].item(), parts["overlap"].item(), parts["vad"].item(), parts["cosine"].item())
        if global_step % HPARAMS.train.eval_every_steps == 0:
            metrics = evaluate(model, device, criterion, global_step)
            logger.info("eval iter=%d %s", global_step, " ".join(f"L{level}(loss={item['loss']:.5f},acc={item['accuracy']:.4f})" for level, item in metrics.items()))
        if global_step % HPARAMS.train.checkpoint_every_steps == 0:
            logger.info("saved checkpoint=%s", save_checkpoint(model, optimizer, global_step, args.output))
    if global_step and global_step % HPARAMS.train.checkpoint_every_steps:
        logger.info("saved final checkpoint=%s", save_checkpoint(model, optimizer, global_step, args.output))


if __name__ == "__main__":
    main()
