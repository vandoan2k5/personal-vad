from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import collate_pvad
from model import PersonalVAD
from model.pvad import CLASS_NAMES
from train import make_dataset


def evaluate_level(model: PersonalVAD, level: int, batches: int, batch_size: int, seed: int, device: torch.device) -> torch.Tensor:
    dataset = make_dataset(batches * batch_size, level, seed + level, batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_pvad, num_workers=0)
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for batch in loader:
        stream = batch["stream"].to(device)
        enrollment = batch["enrollment"].to(device)
        labels = batch["labels"].to(device)
        with torch.inference_mode():
            logits = model(stream, model.enroll(enrollment))["logits"]
        frame_count = min(logits.size(1), labels.size(1))
        predicted = logits[:, :frame_count].argmax(dim=-1)
        labels = labels[:, :frame_count]
        for actual in range(3):
            for predicted_label in range(3):
                confusion[actual, predicted_label] += ((labels == actual) & (predicted == predicted_label)).sum().cpu()
    return confusion


def metrics(confusion: torch.Tensor) -> dict[str, object]:
    correct = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    recall = correct / support.clamp_min(1)
    precision = correct / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "confusion": confusion.tolist(),
        "accuracy": float(correct.sum() / support.sum().clamp_min(1)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def plot(results: dict[int, dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for level, axis in zip((1, 2, 3), axes):
        confusion = np.asarray(results[level]["confusion"], dtype=float)
        normalized = confusion / confusion.sum(axis=1, keepdims=True).clip(min=1)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axis.set_title(f"L{level} — accuracy={results[level]['accuracy']:.3f}\nmacro-F1={results[level]['macro_f1']:.3f}")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(3), CLASS_NAMES)
        axis.set_yticks(range(3), CLASS_NAMES)
        for row in range(3):
            for column in range(3):
                value = normalized[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "black")
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.03, label="Row-normalized fraction")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Personal-VAD separately on curriculum levels.")
    parser.add_argument("--checkpoint", default="checkpoints/pvad_iter_0005000.pt")
    parser.add_argument("--batches-per-level", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="docs/evaluation")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PersonalVAD.from_checkpoint(args.checkpoint, device=device).to(device).eval()
    results: dict[int, dict[str, object]] = {}
    for level in (1, 2, 3):
        print(f"Evaluating L{level}...", flush=True)
        results[level] = metrics(evaluate_level(model, level, args.batches_per_level, args.batch_size, 42, device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem
    json_path = output_dir / f"{stem}_metrics.json"
    image_path = output_dir / f"{stem}_confusion_by_level.png"
    json_path.write_text(json.dumps({f"L{level}": value for level, value in results.items()}, indent=2))
    plot(results, image_path)
    print(f"Metrics: {json_path}")
    print(f"Confusion matrices: {image_path}")


if __name__ == "__main__":
    main()
