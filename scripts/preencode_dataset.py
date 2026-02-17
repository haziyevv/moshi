# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-encode a DailyTalkContiguous-style dataset into codes for training.

Reads stereo WAV + per-file JSON alignments, encodes audio with Mimi and
text with the Qwen tokenizer, and writes one codes tensor per sample.

Usage::

    python scripts/preencode_dataset.py \
        --data-dir ./daily-talk-contiguous \
        --jsonl ./daily-talk-contiguous/dailytalk.jsonl \
        --out-dir ./encoded_codes \
        --duration-sec 30

The output directory will contain one .pt file per sample:
    encoded_codes/0000.pt  ->  {"codes": tensor of shape [1, K, T]}
    encoded_codes/0001.pt  ->  ...

These can then be loaded by the training script.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_text_stream(
    alignments: list,
    segment_duration: float,
    frame_rate: float,
    tokenizer,
    text_padding_id: int,
    end_of_text_padding_id: int,
    zero_token_id: int,
    main_speaker: str = "SPEAKER_MAIN",
    keep_main_only: bool = True,
) -> torch.Tensor:
    """Build a text token stream from word-level alignments.

    Returns a tensor of shape [1, 1, T] with text token ids at the correct
    time positions (at frame_rate Hz), padded elsewhere.

    This mirrors moshi-finetune's Interleaver.build_token_stream but uses the
    Qwen tokenizer instead of SentencePiece.
    """
    T = math.ceil(segment_duration * frame_rate)
    text_tokens = [text_padding_id] * T

    # Filter to main speaker only
    if keep_main_only:
        alignments = [a for a in alignments if a[2] == main_speaker]

    # Filter out zero/negative duration
    alignments = [a for a in alignments if a[1][0] < a[1][1]]

    # Sort by start time
    alignments = sorted(alignments, key=lambda a: a[1][0])

    # Tokenize each word
    from collections import deque
    to_append: deque = deque()
    last_word_end = -1
    alignment_idx = 0
    is_new_word = False

    for t in range(T):
        # Consume alignments whose start falls before frame t+1
        while (
            alignment_idx < len(alignments)
            and alignments[alignment_idx][1][0] * frame_rate < t + 1
        ):
            word = alignments[alignment_idx][0].strip()
            word_start, word_end = alignments[alignment_idx][1]
            last_word_end = int(word_end * frame_rate)

            # Tokenize with Qwen tokenizer — extend (not replace) to preserve
            # tokens from earlier words that haven't been placed yet.
            tokens = tokenizer.encode(word, add_special_tokens=False)
            to_append.extend(tokens)
            if tokens:
                is_new_word = True
            alignment_idx += 1

        if to_append:
            # Mark end-of-padding before the first token of a new word
            if is_new_word and t > 0 and text_tokens[t - 1] == text_padding_id:
                text_tokens[t - 1] = end_of_text_padding_id
            text_tokens[t] = to_append.popleft()
            is_new_word = False
        elif t < last_word_end:
            # Within a word boundary but no tokens left: use padding
            text_tokens[t] = text_padding_id

    return torch.tensor(text_tokens, dtype=torch.long).view(1, 1, T)


def main():
    parser = argparse.ArgumentParser(description="Pre-encode dataset to codes for Qwen-Moshi training.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root of the dataset (e.g. ./daily-talk-contiguous)")
    parser.add_argument("--jsonl", type=str, required=True,
                        help="Path to the .jsonl manifest")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for .pt files")
    parser.add_argument("--hf-repo", type=str, default="Qwen/Qwen2.5-3B",
                        help="HuggingFace repo for the Qwen tokenizer")
    parser.add_argument("--duration-sec", type=float, default=30.0,
                        help="Max duration in seconds per sample (truncate/pad)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="If >0, only encode this many samples (for testing)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from moshi.models.loaders import CheckpointInfo

    # Load Mimi
    print("Loading Mimi codec...")
    ckpt = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    mimi = ckpt.get_mimi(device=args.device)
    mimi.eval()
    frame_rate = mimi.frame_rate
    sample_rate = mimi.sample_rate
    print(f"  Mimi: sample_rate={sample_rate}, frame_rate={frame_rate}")

    # Load Qwen tokenizer
    print(f"Loading Qwen tokenizer from {args.hf_repo}...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Model token IDs (must match moshi_qwen_3b.json)
    # Use Qwen special tokens to avoid collision with regular vocabulary
    text_padding_id = 151643       # <|endoftext|> - Qwen's pad/eos token
    end_of_text_padding_id = 151645  # <|im_end|> - marks end of padding before text
    zero_token_id = -1

    num_audio_frames = math.ceil(args.duration_sec * frame_rate)

    # Read manifest
    data_dir = Path(args.data_dir)
    with open(args.jsonl) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    print(f"Manifest: {len(manifest)} samples")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_encoded = 0
    n_skipped = 0

    for idx, entry in enumerate(manifest):
        if args.max_samples > 0 and n_encoded >= args.max_samples:
            break

        wav_path = data_dir / entry["path"]
        json_path = wav_path.with_suffix(".json")

        if not wav_path.exists():
            n_skipped += 1
            continue
        if not json_path.exists():
            n_skipped += 1
            continue

        # Load audio
        try:
            import sphn
            wav_data, wav_sr = sphn.read(str(wav_path))
        except Exception:
            import soundfile as sf
            wav_data, wav_sr = sf.read(str(wav_path), dtype="float32")
            wav_data = wav_data.T  # [channels, samples]

        if wav_sr != sample_rate:
            # Resample to Mimi's expected sample rate
            wav_tensor = torch.tensor(wav_data, dtype=torch.float32)
            wav_tensor = torchaudio.functional.resample(wav_tensor, orig_freq=wav_sr, new_freq=sample_rate)
            wav_data = wav_tensor.numpy()

        # Stereo: channel 0 = Moshi (main), channel 1 = user
        if wav_data.ndim == 1:
            wav_data = np.stack([wav_data, wav_data])
        elif wav_data.shape[0] > 2:
            wav_data = wav_data[:2]

        # Truncate/pad to duration_sec
        max_samples = int(args.duration_sec * sample_rate)
        if wav_data.shape[1] > max_samples:
            wav_data = wav_data[:, :max_samples]

        # Encode both channels with Mimi
        with torch.no_grad():
            # Mimi expects [B, 1, T_audio]
            main_audio = torch.tensor(wav_data[0:1], dtype=torch.float32, device=args.device).unsqueeze(0)
            user_audio = torch.tensor(wav_data[1:2], dtype=torch.float32, device=args.device).unsqueeze(0)

            main_codes = mimi.encode(main_audio)  # [1, num_codebooks, T_frames]
            user_codes = mimi.encode(user_audio)   # [1, num_codebooks, T_frames]

        actual_frames = main_codes.shape[-1]

        # Pad/truncate to num_audio_frames
        if actual_frames < num_audio_frames:
            main_codes = torch.nn.functional.pad(main_codes, (0, num_audio_frames - actual_frames), value=zero_token_id)
            user_codes = torch.nn.functional.pad(user_codes, (0, num_audio_frames - actual_frames), value=zero_token_id)
        else:
            main_codes = main_codes[:, :, :num_audio_frames]
            user_codes = user_codes[:, :, :num_audio_frames]

        # Load alignments and build text stream
        with open(json_path) as f:
            transcript = json.load(f)
        alignments = transcript.get("alignments", [])

        segment_duration = min(entry["duration"], args.duration_sec)
        text_stream = build_text_stream(
            alignments,
            segment_duration,
            frame_rate,
            tokenizer,
            text_padding_id,
            end_of_text_padding_id,
            zero_token_id,
        )
        # Pad text to num_audio_frames
        if text_stream.shape[-1] < num_audio_frames:
            text_stream = torch.nn.functional.pad(
                text_stream, (0, num_audio_frames - text_stream.shape[-1]), value=zero_token_id
            )
        else:
            text_stream = text_stream[:, :, :num_audio_frames]

        # Assemble codes: [1, K, T] where K = 1 (text) + n_q (audio)
        # Moshi layout (n_q=16, dep_q=8):
        #   codebook 0:     text tokens
        #   codebooks 1-8:  main speaker audio (8 Mimi codebooks)
        #   codebooks 9-16: user speaker audio (8 Mimi codebooks)
        # Both main_codes and user_codes have shape [1, 8, T] from Mimi.
        codes = torch.cat([
            text_stream.to(main_codes.device),   # [1, 1, T]
            main_codes,                           # [1, 8, T] - main speaker
            user_codes,                           # [1, 8, T] - user speaker
        ], dim=1)  # [1, 17, T]

        # Save
        out_path = out_dir / f"{n_encoded:06d}.pt"
        torch.save({"codes": codes.cpu()}, str(out_path))
        n_encoded += 1

        if n_encoded % 50 == 0:
            print(f"  Encoded {n_encoded}/{len(manifest)} (skipped {n_skipped}), codes shape: {codes.shape}")

    print(f"\nDone. Encoded {n_encoded} samples to {out_dir} (skipped {n_skipped}).")
    print(f"Each file contains codes of shape [1, {codes.shape[1] if n_encoded else '?'}, {num_audio_frames}].")


if __name__ == "__main__":
    main()
