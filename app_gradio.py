"""Gradio demo cho Personal-VAD streaming 3-class (tss / ntss / ns)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch
import torchaudio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model import PersonalVAD
from model.pvad import CLASS_NAMES

BASE_DIR = Path(__file__).parent
CKPT_DIR = BASE_DIR / "checkpoints"
SR = 16_000
HOP = 0.01  # 10 ms posteriors
STREAM_CHUNK_MS = 40

_MODEL_CACHE: dict[str, PersonalVAD] = {}


def list_checkpoints() -> list[str]:
    files = sorted(CKPT_DIR.glob("*.pt"))
    if not files:
        return []
    # Ưu tiên pvad.pt lên đầu
    names = [f.name for f in files]
    if "pvad.pt" in names:
        names.remove("pvad.pt")
        names = ["pvad.pt"] + names
    return names


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(ckpt_name: str, device: torch.device) -> PersonalVAD:
    key = f"{ckpt_name}@{device.type}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    path = CKPT_DIR / ckpt_name
    if not path.exists():
        raise gr.Error(f"Không tìm thấy checkpoint: {path}")
    model = PersonalVAD.from_checkpoint(str(path), device=str(device))
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def load_mono_16k(path: str | Path) -> torch.Tensor:
    """Load audio -> mono 16k Tensor [T]."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)


def smooth_posterior(post: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or post.shape[0] < win:
        return post
    kernel = np.ones(win, dtype=np.float32) / win
    out = np.stack(
        [np.convolve(post[:, c], kernel, mode="same") for c in range(post.shape[1])],
        axis=1,
    )
    out = np.clip(out, 1e-6, 1.0)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def run_inference(
    ckpt_name: str,
    enroll_path: str | None,
    stream_path: str | None,
    device_choice: str,
    tss_thresh: float,
    smooth_win: int,
):
    if not ckpt_name:
        raise gr.Error("Chưa chọn checkpoint.")
    if not enroll_path:
        raise gr.Error("Vui lòng upload / ghi âm file ENROLLMENT (giọng target).")
    if not stream_path:
        raise gr.Error("Vui lòng upload / ghi âm file STREAM cần test.")

    device = resolve_device(device_choice)
    model = get_model(ckpt_name, device)

    enroll = load_mono_16k(enroll_path)
    stream = load_mono_16k(stream_path)
    if enroll.numel() < SR // 2:
        raise gr.Error("File enrollment quá ngắn (< 0.5s).")
    if stream.numel() < 1600:
        raise gr.Error("File stream quá ngắn.")

    with torch.inference_mode():
        emb = model.enroll(enroll.unsqueeze(0).to(device))
        stream_batch = stream.unsqueeze(0).to(device)
        state = model.init_stream(1, device, stream_batch.dtype)
        logits_parts, cosine_parts = [], []
        chunk_samples = SR * STREAM_CHUNK_MS // 1000
        for start in range(0, stream_batch.size(1), chunk_samples):
            out, state = model.stream_step(stream_batch[:, start:start + chunk_samples], emb, state)
            logits_parts.append(out["logits"])
            cosine_parts.append(out["cosine"])
        logits = torch.cat(logits_parts, dim=1)[0].cpu()
        cosine = torch.cat(cosine_parts, dim=1)[0].cpu().numpy()
    post = torch.softmax(logits, dim=-1).numpy()  # [T, 3]
    post = smooth_posterior(post, int(smooth_win))

    # Thresholding: nếu p(tss) >= thresh -> tss, else argmax(ntss, ns)
    pred = post.argmax(axis=1)
    if tss_thresh > 0:
        tss_prob = post[:, 0]
        fallback = 1 + post[:, 1:].argmax(axis=1)  # 1=ntss, 2=ns
        pred = np.where(tss_prob >= tss_thresh, 0, fallback)

    n_frames = post.shape[0]
    times = np.arange(n_frames) * HOP
    dur = n_frames * HOP

    # ---- Stats ----
    counts = [(pred == c).sum() for c in range(3)]
    pct = [100.0 * c / max(n_frames, 1) for c in counts]
    summary = (
        f"### Kết quả ({ckpt_name} @ {device.type})\n"
        f"- Stream: {stream.numel()/SR:.2f}s ({n_frames} frames x 10ms) | "
        f"Enrollment: {enroll.numel()/SR:.2f}s\n"
        f"- Cosine mean: {cosine.mean():+.3f} (min {cosine.min():+.3f} / max {cosine.max():+.3f})\n"
        f"- **tss** (target): {counts[0]} frames ({pct[0]:.1f}% = {counts[0]*HOP:.2f}s)\n"
        f"- **ntss** (non-target): {counts[1]} frames ({pct[1]:.1f}% = {counts[1]*HOP:.2f}s)\n"
        f"- **ns** (non-speech): {counts[2]} frames ({pct[2]:.1f}% = {counts[2]*HOP:.2f}s)\n"
    )

    # ---- Segments (gộp frame liên tiếp cùng label) ----
    segs: list[str] = []
    if n_frames > 0:
        start = 0
        for i in range(1, n_frames + 1):
            if i == n_frames or pred[i] != pred[start]:
                segs.append(
                    f"{start*HOP:6.2f}s - {i*HOP:6.2f}s : {CLASS_NAMES[int(pred[start])]} "
                    f"(p_tss={post[start:i,0].mean():.2f}, cos={cosine[start:i].mean():+.2f})"
                )
                start = i
    seg_text = "\n".join(segs[:200])
    if len(segs) > 200:
        seg_text += f"\n... (+{len(segs)-200} segments)"

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2, 1.2]})
    # waveform
    wav_np = stream.numpy()
    t_wav = np.arange(wav_np.shape[0]) / SR
    axes[0].plot(t_wav, wav_np, linewidth=0.5, color="black")
    axes[0].set_ylabel("waveform")
    axes[0].set_title(f"Personal-VAD | {ckpt_name} | enroll={enroll.numel()/SR:.1f}s stream={dur:.1f}s")
    axes[0].grid(alpha=0.3)

    colors = {"tss": "tab:green", "ntss": "tab:orange", "ns": "tab:gray"}
    for c, name in enumerate(CLASS_NAMES):
        axes[1].plot(times, post[:, c], label=f"{name}", color=colors[name], linewidth=1.2)
    axes[1].axhline(tss_thresh, color="green", linestyle="--", linewidth=1, alpha=0.7,
                    label=f"thr_tss={tss_thresh:.2f}")
    axes[1].set_ylabel("posterior")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(ncol=4, fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(times, cosine, color="tab:blue", linewidth=1.0, label="cosine")
    axes[2].set_ylabel("cosine")
    axes[2].set_xlabel("time (s)")
    axes[2].set_xlim(0, max(dur, 0.1))
    axes[2].grid(alpha=0.3)
    # decision bar
    bar_colors = [colors[CLASS_NAMES[c]] for c in pred]
    axes[2].scatter(times, np.full_like(times, cosine.min() - 0.05),
                    c=bar_colors, s=4, marker="s", label="pred (xanh=tss)")
    axes[2].legend(fontsize=9)
    fig.tight_layout()

    # ---- TSS-only audio (giữ lại đoạn pred==tss) ----
    hop_samples = 160
    mask_frames = (pred == 0).astype(np.float32)
    mask_samples = np.repeat(mask_frames, hop_samples)[: wav_np.shape[0]]
    if mask_samples.shape[0] < wav_np.shape[0]:  # pad nếu lệch
        mask_samples = np.pad(mask_samples, (0, wav_np.shape[0] - mask_samples.shape[0]))
    tss_audio = (wav_np * mask_samples).astype(np.float32)

    # ---- CSV posteriors ----
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="pvad_")
    header = "time_s,p_tss,p_ntss,p_ns,cosine,pred"
    rows = np.column_stack([
        times,
        post[:, 0], post[:, 1], post[:, 2],
        cosine, pred,
    ])
    np.savetxt(tmp.name, rows, delimiter=",", header=header, comments="", fmt="%.4f")
    tmp.close()

    return fig, summary, seg_text, (SR, tss_audio), tmp.name


ckpt_list = list_checkpoints()
default_ckpt = "pvad.pt" if "pvad.pt" in ckpt_list else (ckpt_list[0] if ckpt_list else None)

with gr.Blocks(title="Personal-VAD Demo") as demo:
    gr.Markdown(
        "# Personal-VAD Demo\n"
        "Streaming 3-class: `tss` (target) / `ntss` (non-target) / `ns` (non-speech). "
        "Upload file **enrollment** (giọng người cần detect) + file **stream** cần chấm, chọn checkpoint trong `checkpoints/` rồi bấm **Chạy**."
    )
    with gr.Row():
        ckpt = gr.Dropdown(choices=ckpt_list, value=default_ckpt, label="Checkpoint (checkpoints/)")
        device_in = gr.Dropdown(choices=["auto", "cuda", "cpu"], value="auto", label="Device")
    with gr.Row():
        enroll_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="1) Enrollment — giọng target (nên >2s, sạch)")
        stream_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="2) Stream — file cần test VAD")
    with gr.Row():
        thr = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Ngưỡng tss (p_tss >= ngưỡng -> tss, thấp hơn -> chọn ntss/ns)")
        smooth = gr.Slider(1, 51, value=5, step=2, label="Làm mượt posterior (số frames, lẻ)")
    run_btn = gr.Button("Chạy inference", variant="primary")

    plot_out = gr.Plot(label="Biểu đồ posterior + cosine")
    summary_out = gr.Markdown()
    with gr.Row():
        seg_out = gr.Textbox(label="Segments (start - end : label)", lines=12, max_lines=20)
        with gr.Column():
            tss_audio_out = gr.Audio(label="Audio chỉ giữ lại tss (nghe kiểm tra)", type="numpy")
            csv_out = gr.File(label="Tải CSV posteriors (time, p_tss, p_ntss, p_ns, cosine, pred)")

    examples = None
    ex1 = str(BASE_DIR / "model" / "pretrained" / "example1.wav")
    if Path(ex1).exists():
        gr.Markdown("Ví dụ nhanh: dùng cùng 1 file làm enrollment + stream để kiểm tra pipeline.")
        examples = gr.Examples(examples=[[default_ckpt, ex1, ex1, "auto", 0.5, 5]], inputs=[ckpt, enroll_in, stream_in, device_in, thr, smooth])

    run_btn.click(
        fn=run_inference,
        inputs=[ckpt, enroll_in, stream_in, device_in, thr, smooth],
        outputs=[plot_out, summary_out, seg_out, tss_audio_out, csv_out],
    )

if __name__ == "__main__":
    if not ckpt_list:
        print(f"CẢNH BÁO: không thấy file *.pt trong {CKPT_DIR}")
    demo.launch(server_name="0.0.0.0", server_port=7860)
