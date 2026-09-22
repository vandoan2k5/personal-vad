# Personal VAD

Streaming three-class personal VAD: `tss` (target speech), `ntss` (non-target speech), and `ns` (non-speech). Target identity is represented by a frozen local CAM++ profile extracted once from enrollment audio. A causal acoustic GRU and causal speaker pre-net process each newly available Mel frame; the target cosine score conditions the acoustic path through FiLM.

## Data

Place speech as `data/audio_with_id/<speaker_id>/*.wav` and noise under `data/noise/`. Noise can be stored in nested directories or symlinked directories and supports common `wav`, `flac`, `mp3`, `ogg`, `m4a`, `aac`, and `opus` files. The online mixer uses an easy-to-hard curriculum over levels: L1 has non-overlapping target/non-target turns, L2 adds 25% overlap, L3 adds 90% overlap with symmetric SIR in [-10, 10] dB. Overlap frames carry hierarchical soft labels (`tss` 0.9 / `ntss` 0.1, `tss > ntss > ns`); background noise covers the whole mix and is not an `ns` label. Per-level quotas are tuned so dominant labels stay ~= 1/3 each of `tss`, `ntss`, and `ns`, keeping macro-F1 comparable with expected order L1 > L2 > L3. The noise is present in every example.

`data/audio_with_id_gender.json` provides `{ "speaker_id": "male|female" }` metadata for strict same-/different-gender negative sampling. All current `audio_with_id` speaker directories are covered by this file.

## Run

```bash
uv sync
uv run python train.py --epochs 20 --batch-size 8 --device cuda
uv run python train.py --epochs 5 --batch-size 8 --device cuda --level 1
uv run python inference.py --checkpoint checkpoints/pvad.pt --enrollment target.wav --stream stream.wav --chunk-ms 40
```

Training starts from scratch by default. Enrollment duration is randomized between 0.5 and 3 seconds, while speech turns are randomized with a minimum duration of 0.8 seconds. Use `--resume /path/to/checkpoint.pt` only for a checkpoint trained with this CAM++/speaker-pre-net architecture.

All default paths, model dimensions, loss weights, 30-second augmentation duration, label ratios, curriculum schedule, and optimizer settings are centralized in `hparams.py`. CLI arguments can force one mixer level with `--level 1|2|3`; without it, training follows the curriculum from L1-heavy to overlap-heavy (ending at 20% level 1, 40% level 2, 40% level 3).

Training logs total loss with its `pvad/overlap/vad/cosine` parts plus accuracy, balanced accuracy, and macro-F1 at every iterator, evaluates forced level 1/2/3 datasets every 1,000 iterators, and writes `checkpoints/pvad_iter_0005000.pt` (plus the latest checkpoint) every 5,000 iterators. These cadences and evaluation size are configurable in `hparams.py`.

The CAM++ checkpoint is loaded from `model/pretrained/campplus`. Enrollment can be 0.5–3 seconds or longer. The frozen encoder runs only during enrollment; streaming chunks should be below 150 ms and are made chunk-invariant by the stateful frontend. Output has 10 ms posteriors and cosine scores for thresholding/diagnostics. Train a new checkpoint after this architecture change; checkpoints created before the CAM++/speaker-pre-net architecture are not compatible with this model.
