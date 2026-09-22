from __future__ import annotations

import argparse
import json
import torch
import torchaudio
from model import PersonalVAD
from model.pvad import CLASS_NAMES


def load(path: str, rate: int = 16_000) -> torch.Tensor:
    audio, source_rate = torchaudio.load(path); audio = audio.mean(dim=0)
    return torchaudio.functional.resample(audio, source_rate, rate) if source_rate != rate else audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--enrollment", required=True); parser.add_argument("--stream", required=True)
    parser.add_argument("--chunk-ms", type=float, default=40.0, help="Input chunk size in milliseconds (must be below 150).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 < args.chunk_ms < 150:
        parser.error("--chunk-ms must be between 0 and 150 ms")
    device = torch.device(args.device)
    model = PersonalVAD.from_checkpoint(args.checkpoint, device).to(device).eval()
    with torch.inference_mode():
        embedding = model.enroll(load(args.enrollment).unsqueeze(0).to(device))
        stream = load(args.stream).unsqueeze(0).to(device)
        chunk_samples = max(1, round(args.chunk_ms * 16))
        state = model.init_stream(1, device, stream.dtype)
        logits, cosine = [], []
        for start in range(0, stream.size(1), chunk_samples):
            output, state = model.stream_step(stream[:, start:start + chunk_samples], embedding, state)
            logits.append(output["logits"])
            cosine.append(output["cosine"])
        output_logits = torch.cat(logits, dim=1)
        output_cosine = torch.cat(cosine, dim=1)
        posterior = output_logits.softmax(-1)[0].cpu()
    print(json.dumps({"frame_hop_seconds": .01, "chunk_ms": args.chunk_ms, "classes": CLASS_NAMES, "posterior": posterior.tolist(), "cosine": output_cosine[0].cpu().tolist()}))


if __name__ == "__main__": main()
