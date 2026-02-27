#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run inference with a trained Qwen-Moshi model.

Given a stereo WAV (or mono user audio), generates Moshi's response audio.

Usage::

    python scripts/inference_qwen_moshi.py \\
        --qwen-weights runs/phase2/checkpoint_final.safetensors \\
        --input-wav test.wav \\
        --out-wav response.wav
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moshi.models import get_qwen_moshi_lm, LMGen
from moshi.models.loaders import CheckpointInfo

SAMPLE_RATE = 24_000


def main():
    parser = argparse.ArgumentParser(description="Qwen-Moshi inference")
    parser.add_argument("--qwen-weights", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-wav", type=str, required=True,
                        help="Input WAV: stereo (right=user) or mono (=user)")
    parser.add_argument("--out-wav", type=str, default="response.wav")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--temp-text", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-k-text", type=int, default=25)
    args = parser.parse_args()

    config_path = args.config or str(REPO_ROOT / "configs" / "moshi_qwen_3b.json")

    # --- Load models ---
    print("Loading Mimi codec...")
    ckpt = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    mimi = ckpt.get_mimi(device=args.device)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    print("Loading Qwen-Moshi LM...")
    lm = get_qwen_moshi_lm(
        qwen_weights=args.qwen_weights,
        config_path=config_path,
        device=args.device,
        dtype=torch.bfloat16,
    )

    # --- Load audio ---
    import sphn
    wav, wav_sr = sphn.read(args.input_wav)

    if wav_sr != SAMPLE_RATE:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.tensor(wav, dtype=torch.float32), wav_sr, SAMPLE_RATE
        ).numpy()

    if wav.shape[0] >= 2:
        user_wav = wav[1:2]
    else:
        user_wav = wav[0:1]

    # Trim to frame boundary
    n_samples = (user_wav.shape[1] // frame_size) * frame_size
    user_wav = user_wav[:, :n_samples]
    user_tensor = torch.tensor(user_wav, dtype=torch.float32, device=args.device)
    n_frames = n_samples // frame_size

    print(f"Input: {n_frames} frames ({n_frames / mimi.frame_rate:.1f}s)")

    # --- Generate ---
    print("Generating...")
    lm_gen = LMGen(
        lm,
        use_sampling=True,
        temp=args.temp,
        temp_text=args.temp_text,
        top_k=args.top_k,
        top_k_text=args.top_k_text,
    )

    gen_audio = []
    gen_text_tokens = []

    with torch.no_grad():
        with mimi.streaming(1), lm_gen.streaming(1):
            for i in range(n_frames):
                chunk = user_tensor[:, i * frame_size:(i + 1) * frame_size].unsqueeze(0)
                user_codes = mimi.encode(chunk)
                tokens = lm_gen.step(user_codes[:, :, :1])
                if tokens is None:
                    continue
                gen_audio.append(mimi.decode(tokens[:, 1:, :]))
                gen_text_tokens.append(tokens[0, 0, 0].item())

    if not gen_audio:
        print("No audio generated!")
        return

    gen_pcm = torch.cat(gen_audio, dim=-1)
    rms = torch.sqrt(torch.mean(gen_pcm ** 2)).item()
    peak = torch.max(torch.abs(gen_pcm)).item()
    print(f"  Audio: RMS={rms:.6f}, Peak={peak:.6f}")

    sphn.write_wav(args.out_wav,
                   gen_pcm[0, 0].cpu().numpy().astype(np.float32), SAMPLE_RATE)
    print(f"  Saved: {args.out_wav}")

    # Decode generated text
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
        padding_id = lm.existing_text_padding_id
        non_padding = [t for t in gen_text_tokens if t != padding_id]
        if non_padding:
            text = tok.decode(non_padding, skip_special_tokens=True)
            print(f"  Generated text: {text[:300]}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
