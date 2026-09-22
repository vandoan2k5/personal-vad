from __future__ import annotations

import json
import csv
import os
import random
from pathlib import Path
import torch
from torch import Tensor
import torch.nn.functional as functional
import torchaudio
from torch.utils.data import Dataset
from model.pvad import NS, NTSS, TSS


class PVADMixtureDataset(Dataset):
    """Online mixer with balanced frame quotas for each curriculum level."""
    DEFAULT_LEVEL_RATIOS = ((1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3), (1 / 3, 1 / 3, 1 / 3))

    def __init__(self, audio_root: str | Path, noise_root: str | Path, samples: int, duration_seconds: float = 30, sample_rate: int = 16_000, seed: int = 42, level: int | None = None, gender_metadata: str | Path | None = None, level_ratios: tuple[tuple[float, float, float], ...] = DEFAULT_LEVEL_RATIOS, level_sampling_ratios: tuple[float, float, float] = (0.20, 0.40, 0.40), batch_size: int = 1, curriculum: dict[int, tuple[float, float, float]] | None = None, excluded_noise_labels: tuple[str, ...] = (), soft_dominant: float = 0.9, soft_secondary: float = 0.1) -> None:
        self.sample_rate, self.total_samples, self.samples, self.seed = sample_rate, int(duration_seconds * sample_rate), samples, seed
        self.soft_dominant, self.soft_secondary = float(soft_dominant), float(soft_secondary)
        if not 0.0 <= self.soft_dominant <= 1.0 or not 0.0 <= self.soft_secondary <= 1.0 or abs(self.soft_dominant + self.soft_secondary - 1.0) > 1e-6:
            raise ValueError("soft_dominant and soft_secondary must be in [0, 1] and sum to 1")
        self.rng = random.Random(seed)
        self.speakers = {item.name: sorted(item.glob("*.wav")) for item in Path(audio_root).iterdir() if item.is_dir()}
        self.speakers = {speaker: files for speaker, files in self.speakers.items() if len(files) >= 2}
        self.level_ratios = level_ratios
        if len(level_ratios) != 3 or any(len(ratio) != 3 or abs(sum(ratio) - 1.0) > 1e-6 for ratio in level_ratios):
            raise ValueError("level_ratios must contain three 3-class distributions summing to 1")
        if len(level_sampling_ratios) != 3 or any(ratio < 0 for ratio in level_sampling_ratios) or abs(sum(level_sampling_ratios) - 1.0) > 1e-6:
            raise ValueError("level_sampling_ratios must contain three non-negative values summing to 1")
        self.level_sampling_ratios = level_sampling_ratios
        self.batch_size = batch_size
        self.curriculum = dict(sorted((curriculum or {}).items()))
        for start_step, ratios in self.curriculum.items():
            if start_step < 0 or len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
                raise ValueError("curriculum must map non-negative steps to three non-negative ratios summing to 1")
        audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}
        excluded_noise_stems = self._find_excluded_noise_stems(noise_root, excluded_noise_labels)
        self.noises = sorted(
            Path(root) / filename
            for root, _, filenames in os.walk(noise_root, followlinks=True)
            for filename in filenames
            if Path(filename).suffix.lower() in audio_extensions
            and Path(filename).stem not in excluded_noise_stems
        )
        self.level = level
        if level is not None and level not in (1, 2, 3): raise ValueError("level must be 1, 2, 3, or None for the curriculum")
        metadata = Path(gender_metadata) if gender_metadata else Path(audio_root).parent / "audio_with_id_gender.json"
        self.gender = {}
        if metadata.exists():
            with metadata.open() as handle: self.gender = {speaker: gender.lower() for speaker, gender in json.load(handle).items()}
        if len(self.speakers) < 3: raise ValueError("Need >=3 speakers, each with >=2 wav files.")
    def __len__(self) -> int: return self.samples

    @staticmethod
    def _find_excluded_noise_stems(noise_root: str | Path, excluded_labels: tuple[str, ...]) -> set[str]:
        """Read FSD50K label CSVs and return clip IDs containing vocal labels."""
        excluded = set(excluded_labels)
        stems: set[str] = set()
        if not excluded:
            return stems
        label_files = (
            Path(root) / filename
            for root, _, filenames in os.walk(noise_root, followlinks=True)
            for filename in filenames
            if Path(root).name == "labels" and filename.endswith(".csv")
        )
        for labels_path in label_files:
            try:
                with labels_path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        labels = {label.strip() for label in row.get("labels", "").split(",")}
                        if labels & excluded and row.get("fname"):
                            stems.add(Path(row["fname"]).stem)
            except (OSError, UnicodeError, csv.Error):
                continue
        return stems
    def _load(self, path: Path) -> Tensor:
        audio, rate = torchaudio.load(path); audio = audio.mean(0)
        return torchaudio.functional.resample(audio, rate, self.sample_rate) if rate != self.sample_rate else audio
    def _segment(self, audio: Tensor, length: int) -> Tensor:
        if audio.numel() <= length: return functional.pad(audio, (0, length - audio.numel()))
        start = self.rng.randrange(audio.numel() - length + 1); return audio[start:start + length]
    def _add(self, mix: Tensor, labels: Tensor | None, audio: Tensor, label: int | None, start: int, gain: float) -> None:
        end = min(self.total_samples, start + audio.numel()); mix[start:end] += audio[:end-start] * 10 ** (gain / 20)
        if label is not None and labels is not None:
            labels[start:end] = label

    def _turn_lengths(self, total: int) -> list[int]:
        frame_total = total // 160
        minimum, maximum = int(0.8 * self.sample_rate / 160), int(3.0 * self.sample_rate / 160)
        lengths = []
        remaining = frame_total
        while remaining:
            if remaining <= maximum:
                if lengths and remaining < minimum:
                    lengths[-1] += remaining
                else:
                    lengths.append(remaining)
                break
            length = self.rng.randint(minimum, min(maximum, remaining - minimum))
            lengths.append(length)
            remaining -= length
        return [length * 160 for length in lengths]

    def _random_gaps(self, total: int, count: int) -> list[int]:
        frame_total = total // 160
        minimum = int(0.2 * self.sample_rate / 160)
        if frame_total < count * minimum:
            return [frame_total * 160 // count] * count
        remainder = frame_total - count * minimum
        weights = [self.rng.random() + 0.05 for _ in range(count)]
        gaps = [minimum + int(remainder * weight / sum(weights)) for weight in weights]
        gaps[-1] += frame_total - sum(gaps)
        return [gap * 160 for gap in gaps]

    def _schedule(self, lengths: list[int], occupied: list[tuple[int, int]] | None = None, overlap_rate: float = 0.0) -> list[tuple[int, int]]:
        occupied = occupied or []
        intervals = []
        for length in lengths:
            for _ in range(40):
                if occupied and self.rng.random() < overlap_rate:
                    begin, end = self.rng.choice(occupied)
                    low = max(0, begin - length + int(0.25 * self.sample_rate))
                    high = min(self.total_samples - length, end - int(0.25 * self.sample_rate))
                    start = self.rng.randint(low, high) if low <= high else self.rng.randint(0, self.total_samples - length)
                else:
                    start = self.rng.randint(0, self.total_samples - length)
                if all(start >= end or start + length <= begin for begin, end in intervals):
                    intervals.append((start, start + length))
                    break
            else:
                intervals.append((max(0, self.total_samples - length), self.total_samples))
        return intervals

    def _soft_from_masks(self, frame_target: Tensor, frame_ntss: Tensor) -> Tensor:
        """Hierarchical soft labels with tss > ntss > ns.

        Background noise covers the whole mix and is NOT an NS label.
        NS means "neither speaker active". Hence reachable states are:
        TSS-only -> [1, 0, 0], NTSS-only -> [0, 1, 0],
        NS-only -> [0, 0, 1], TSS+NTSS overlap -> [soft_dominant, soft_secondary, 0].
        """
        soft = torch.zeros(frame_target.numel(), 3)
        target = frame_target.bool()
        ntss = frame_ntss.bool()
        t_only = target & ~ntss
        n_only = ntss & ~target
        neither = ~(target | ntss)
        both = target & ntss
        soft[t_only, TSS] = 1.0
        soft[n_only, NTSS] = 1.0
        soft[neither, NS] = 1.0
        if bool(both.any()):
            soft[both, TSS] = self.soft_dominant
            soft[both, NTSS] = self.soft_secondary
            soft[both, NS] = 0.0
        return soft

    def _add_overlap_tracks(self, mix: Tensor, target_audio: Tensor, ntss_audio: Tensor, target_quota: int, ntss_quota: int, level: int) -> Tensor:
        """Mix overlapping speech and return per-frame soft labels ``[frames, 3]``."""
        target_intervals = self._schedule(self._turn_lengths(target_quota))
        overlap_rate = 0.25 if level == 2 else 0.90
        ntss_intervals = self._schedule(self._turn_lengths(ntss_quota), target_intervals, overlap_rate)
        target_mask = torch.zeros(self.total_samples, dtype=torch.bool)
        ntss_mask = torch.zeros(self.total_samples, dtype=torch.bool)
        target_offset = ntss_offset = 0
        for start, end in target_intervals:
            length = end - start
            # SIR doi xung: target va ntss cung uniform(-5, 5) dB -> SIR trong [-10, 10] dB.
            # (Truoc day target -10..-5 dB con ntss +5..+12 dB, giang target bi chon 15-22 dB
            # duoi interferer, day model bo qua giong nho va loan TSS/NTSS.)
            self._add(mix, None, target_audio[target_offset:target_offset + length], None, start, self.rng.uniform(-5, 5))
            target_mask[start:end] = True
            target_offset += length
        for start, end in ntss_intervals:
            length = end - start
            self._add(mix, None, ntss_audio[ntss_offset:ntss_offset + length], None, start, self.rng.uniform(-5, 5))
            ntss_mask[start:end] = True
            ntss_offset += length
        return self._soft_from_masks(target_mask[::160], ntss_mask[::160])

    def _quotas(self, level: int) -> tuple[int, int, int]:
        frame_count = self.total_samples // 160
        raw = [frame_count * ratio for ratio in self.level_ratios[level - 1]]
        quotas = [int(value) for value in raw]
        for index in sorted(range(3), key=lambda item: raw[item] - quotas[item], reverse=True)[:frame_count - sum(quotas)]:
            quotas[index] += 1
        return tuple(quota * 160 for quota in quotas)

    def _curriculum_level(self, index: int) -> int:
        step = index // self.batch_size + 1
        ratios = self.level_sampling_ratios
        for start_step, candidate in self.curriculum.items():
            if step >= start_step:
                ratios = candidate
            else:
                break
        # ``self.rng`` is reseeded per sample in __getitem__, so this remains
        # deterministic while honoring the configured ratios at every stage.
        position = self.rng.random()
        cumulative = 0.0
        for level, ratio in enumerate(ratios, start=1):
            cumulative += ratio
            if position < cumulative:
                return level
        return 3
    def _other(self, target: str, same_gender: bool | None = None) -> str:
        candidates = [speaker for speaker in self.speakers if speaker != target]
        if same_gender is not None and target in self.gender:
            filtered = [speaker for speaker in candidates if (self.gender.get(speaker) == self.gender[target]) == same_gender]
            if filtered: candidates = filtered
        return self.rng.choice(candidates)
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        # Index-derived RNG prevents identical augmentation streams across workers.
        self.rng = random.Random(self.seed + index)
        level = self.level or self._curriculum_level(index)
        target = self.rng.choice(list(self.speakers)); enroll_path, target_path = self.rng.sample(self.speakers[target], 2)
        enrollment_frames = self.rng.randint(50, 300)
        enrollment = self._segment(self._load(enroll_path), enrollment_frames * 160)
        mix = torch.zeros(self.total_samples)
        if self.noises: mix += self._segment(self._load(self.rng.choice(self.noises)), self.total_samples) * 10 ** (self.rng.uniform(-12, -2) / 20)
        target_quota, ntss_quota, ns_quota = self._quotas(level)
        target_audio = self._segment(self._load(target_path), target_quota)
        others = [self._other(target, True), self._other(target, False)]
        ntss_audio = self._segment(self._load(self.rng.choice(self.speakers[self.rng.choice(others)])), ntss_quota)
        if level == 1:
            labels_sample = torch.full((self.total_samples,), NS, dtype=torch.long)
            turns = [(target_audio, TSS, length) for length in self._turn_lengths(target_quota)]
            turns += [(ntss_audio, NTSS, length) for length in self._turn_lengths(ntss_quota)]
            self.rng.shuffle(turns)
            gaps = self._random_gaps(ns_quota, len(turns) + 1)
            cursor, target_offset, ntss_offset = gaps[0], 0, 0
            for index, (audio, label, length) in enumerate(turns):
                offset = target_offset if label == TSS else ntss_offset
                self._add(mix, labels_sample, audio[offset:offset + length], label, cursor, self.rng.uniform(-8, 4))
                if label == TSS:
                    target_offset += length
                else:
                    ntss_offset += length
                cursor += length + gaps[index + 1]
            # Level 1 has no overlap by construction, so one-hot == hierarchical soft.
            labels = functional.one_hot(labels_sample[::160], num_classes=3).float()
        else:
            labels = self._add_overlap_tracks(mix, target_audio, ntss_audio, target_quota, ntss_quota, level)
        ramp = torch.linspace(self.rng.uniform(.1, 1), self.rng.uniform(1, 2), self.total_samples)
        return {"stream": (mix*ramp).clamp(-1,1), "enrollment": enrollment, "labels": labels}


def collate_pvad(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    maximum = max(item["enrollment"].numel() for item in batch)
    return {"stream": torch.stack([item["stream"] for item in batch]), "enrollment": torch.stack([functional.pad(item["enrollment"], (0, maximum-item["enrollment"].numel())) for item in batch]), "labels": torch.stack([item["labels"] for item in batch])}
