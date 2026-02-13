# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run inference with the fine-tuned Qwen-backed Moshi model using real audio input.

Unlike benchmark_qwen_moshi.py (which feeds silence), this script feeds actual
audio from a WAV file as "user input" and generates the model's audio response.

For stereo DailyTalkContiguous files:
  - Right channel (index 1) = user input  -> fed to the model
  - Left channel  (index 0) = main speaker -> what the model should produce

Usage::

    # Use a sample from the training set to verify the model works:
    python scripts/inference_qwen_moshi.py \\
        --qwen-weights runs/qwen_moshi_ft/checkpoint_epoch_35.safetensors \\
        --input-wav ./daily-talk-contiguous/data_stereo/0.wav \\
        --out-wav generated_response.wav

    # Use a custom mono WAV (treated as user input):
    python scripts/inference_qwen_moshi.py \\
        --qwen-weights runs/qwen_moshi_ft/checkpoint_final.safetensors \\
        --input-wav my_question.wav \\
        --out-wav model_response.wav
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moshi.models import get_qwen_moshi_lm, LMGen
from moshi.models.loaders import CheckpointInfo


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load audio file and return as numpy array [channels, samples] at target_sr."""
    try:
        import sphn
        wav, sr = sphn.read(path)
    except Exception:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32")
        wav = wav.T if wav.ndim > 1 else wav[np.newaxis, :]

    if wav.ndim == 1:
        wav = wav[np.newaxis, :]

    # Resample if needed
    if sr != target_sr:
        import torchaudio
        wav_t = torch.tensor(wav, dtype=torch.float32)
        wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=target_sr)
        wav = wav_t.numpy()

    return wav


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen-Moshi")
    parser.add_argument("--qwen-weights", type=str, required=True,
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-wav", type=str, required=True,
                        help="Input WAV file. Stereo: right channel = user. Mono: treated as user input.")
    parser.add_argument("--out-wav", type=str, default="generated_response.wav",
                        help="Output WAV path for the model's audio response")
    parser.add_argument("--out-text", type=str, default=None,
                        help="If set, save decoded text tokens to this file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-duration", type=float, default=60.0,
                        help="Max duration in seconds to process")
    # Sampling parameters
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-k-text", type=int, default=25)
    args = parser.parse_args()

    config_path = args.config or str(REPO_ROOT / "configs" / "moshi_qwen_3b.json")

    # --- Load models ---
    print("Loading Mimi codec...")
    ckpt = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    mimi = ckpt.get_mimi(device=args.device)
    sample_rate = mimi.sample_rate
    frame_rate = mimi.frame_rate
    frame_size = int(sample_rate / frame_rate)
    print(f"  Mimi: sr={sample_rate}, fr={frame_rate}, frame_size={frame_size}")

    print("Loading Qwen-backed Moshi LM...")
    lm = get_qwen_moshi_lm(
        qwen_weights=args.qwen_weights,
        config_path=config_path,
        device=args.device,
        dtype=torch.bfloat16,
    )
    total_params = sum(p.numel() for p in lm.parameters())
    print(f"  LM: {total_params / 1e9:.2f}B params")

    lm_gen = LMGen(
        lm,
        use_sampling=True,
        temp=args.temp,
        temp_text=args.temp_text,
        top_k=args.top_k,
        top_k_text=args.top_k_text,
    )

    # --- Load input audio ---
    print(f"\nLoading input: {args.input_wav}")
    wav = load_audio(args.input_wav, sample_rate)
    n_channels = wav.shape[0]

    if n_channels >= 2:
        # Stereo: use right channel as user input
        user_audio = wav[1:2]  # [1, samples]
        main_audio_ref = wav[0:1]  # [1, samples] - reference
        print(f"  Stereo input: right channel = user ({user_audio.shape[1] / sample_rate:.1f}s)")
    else:
        # Mono: treat as user input
        user_audio = wav[0:1]
        main_audio_ref = None
        print(f"  Mono input: treated as user ({user_audio.shape[1] / sample_rate:.1f}s)")

    # Truncate to max duration
    max_samples = int(args.max_duration * sample_rate)
    if user_audio.shape[1] > max_samples:
        user_audio = user_audio[:, :max_samples]
        if main_audio_ref is not None:
            main_audio_ref = main_audio_ref[:, :max_samples]
        print(f"  Truncated to {args.max_duration:.0f}s")

    total_samples = user_audio.shape[1]
    total_frames = total_samples // frame_size
    print(f"  Processing {total_frames} frames ({total_samples / sample_rate:.1f}s)")

    # --- Run streaming inference ---
    print(f"\nRunning inference...")
    user_tensor = torch.tensor(user_audio, dtype=torch.float32, device=args.device)

    generated_audio = []
    generated_text_tokens = []
    bs = 1

    t_start = time.time()
    with torch.no_grad():
        with mimi.streaming(bs), lm_gen.streaming(bs):
            for frame_idx in range(total_frames):
                # Get one frame of user audio
                start_sample = frame_idx * frame_size
                end_sample = start_sample + frame_size
                chunk = user_tensor[:, start_sample:end_sample].unsqueeze(0)  # [1, 1, frame_size]

                # Encode user audio to codes
                user_codes = mimi.encode(chunk)  # [1, n_codebooks, 1]

                # LM step
                tokens = lm_gen.step(user_codes[:, :, :1])

                if tokens is None:
                    # Delay buffer warmup
                    continue

                # Extract generated audio tokens and decode
                audio_tokens = tokens[:, 1:, :]  # [1, n_codebooks, 1]
                pcm = mimi.decode(audio_tokens)   # [1, 1, samples]
                generated_audio.append(pcm[0].cpu())

                # Collect text token
                text_tok = tokens[0, 0, 0].item()
                generated_text_tokens.append(text_tok)

                if (frame_idx + 1) % 50 == 0:
                    elapsed = time.time() - t_start
                    print(f"  Frame {frame_idx + 1}/{total_frames}  "
                          f"elapsed={elapsed:.1f}s  "
                          f"last_text_tok={text_tok}")

    total_time = time.time() - t_start
    print(f"\nInference complete: {total_frames} frames in {total_time:.1f}s "
          f"({total_frames * (1000 / frame_rate) / 1000 / total_time:.2f}x real-time)")

    # --- Save output ---
    if generated_audio:
        import sphn
        audio_cat = torch.cat(generated_audio, dim=-1)
        sphn.write_wav(args.out_wav, audio_cat[0].numpy().astype(np.float32), sample_rate)
        duration = audio_cat.shape[-1] / sample_rate
        print(f"\nGenerated audio saved: {args.out_wav} ({duration:.1f}s)")

        # Check if it's actually silence
        rms = torch.sqrt(torch.mean(audio_cat ** 2)).item()
        peak = torch.max(torch.abs(audio_cat)).item()
        print(f"  Audio stats: RMS={rms:.6f}, Peak={peak:.6f}")
        if peak < 0.001:
            print("  WARNING: Output appears to be silence!")
        else:
            print(f"  Output has audio content (peak={peak:.4f})")
    else:
        print("\nNo audio generated (all frames were in warmup)")

    # --- Text tokens summary ---
    if generated_text_tokens:
        non_padding = [t for t in generated_text_tokens if t not in (0, 3, -1)]
        print(f"\nText tokens: {len(generated_text_tokens)} total, "
              f"{len(non_padding)} non-padding")
        if non_padding and args.out_text:
            # Try to decode with Qwen tokenizer
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
                decoded = tok.decode(non_padding, skip_special_tokens=True)
                print(f"  Decoded text: {decoded[:200]}")
                with open(args.out_text, "w") as f:
                    f.write(decoded)
                print(f"  Saved to {args.out_text}")
            except Exception as e:
                print(f"  Could not decode text: {e}")

    # --- Compare with reference (if stereo input) ---
    if main_audio_ref is not None and generated_audio:
        ref_tensor = torch.tensor(main_audio_ref, dtype=torch.float32)
        gen_tensor = audio_cat[0:1]
        min_len = min(ref_tensor.shape[-1], gen_tensor.shape[-1])
        ref_rms = torch.sqrt(torch.mean(ref_tensor[:, :min_len] ** 2)).item()
        gen_rms = torch.sqrt(torch.mean(gen_tensor[:, :min_len] ** 2)).item()
        print(f"\n  Reference RMS: {ref_rms:.6f}")
        print(f"  Generated RMS: {gen_rms:.6f}")


if __name__ == "__main__":
    main()
