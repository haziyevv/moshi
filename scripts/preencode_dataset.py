#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Pre-encode a stereo dialogue dataset into training tensors for Qwen-Moshi.

Each stereo WAV (left = Moshi / SPEAKER_MAIN, right = user) is encoded with
Mimi (8 codebooks per side) and combined with time-aligned Qwen text tokens
into a ``[1, 17, T]`` tensor::

    channel  0       : Qwen text tokens (inner monologue)
    channels 1  – 8  : Moshi audio codebooks (left channel)
    channels 9  – 16 : user audio codebooks  (right channel)

Word-level timestamps from the companion ``.json`` alignment files are used to
place text tokens at the correct frame positions.  Gaps are filled with the
text padding token (``existing_text_padding_id`` from the model config).

Supports multiple data directories in a single run::

    python scripts/preencode_dataset.py \\
        --data-dirs ./daily-talk-contiguous/data_stereo ./multidialog-stereo \\
        --out-dir ./encoded_codes_all \\
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 24_000
FRAME_RATE = 12.5


def load_audio(path: str | Path):
    """Load audio, return (numpy array [C, T], sample_rate)."""
    import sphn
    wav, sr = sphn.read(str(path))
    return wav, sr


def resample_if_needed(wav_tensor: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav_tensor
    import torchaudio.functional as F
    return F.resample(wav_tensor, orig_sr, target_sr)


def align_text_to_frames(
    alignment_path: str | Path,
    num_frames: int,
    tokenizer,
    padding_id: int,
    end_padding_id: int,
) -> torch.Tensor:
    """Build the text token channel from a word-alignment JSON.

    Uses a deque to buffer subword tokens and places one per frame, handling
    cases where a word produces more tokens than available frames before the
    next word.  Places ``end_padding_id`` at the frame just before each new
    word to signal the padding-to-text transition (used by the Moshi loss).

    Returns a LongTensor of shape ``[num_frames]`` with Qwen token ids at
    the correct frame positions and ``padding_id`` / ``end_padding_id``
    elsewhere.
    """
    from collections import deque

    text_codes = [padding_id] * num_frames

    with open(alignment_path) as f:
        data = json.load(f)

    alignments = data.get("alignments", [])
    if not alignments:
        return torch.tensor(text_codes, dtype=torch.long)

    main_alignments = [
        (w, (s, e)) for w, (s, e), spk in alignments
        if spk == "SPEAKER_MAIN" and s < e
    ]
    main_alignments.sort(key=lambda a: a[1][0])

    to_append: deque = deque()
    alignment_idx = 0
    is_new_word = False
    is_first_word = True

    for t in range(num_frames):
        while (
            alignment_idx < len(main_alignments)
            and main_alignments[alignment_idx][1][0] * FRAME_RATE < t + 1
        ):
            word = main_alignments[alignment_idx][0].strip()
            text = word if is_first_word else " " + word
            tokens = tokenizer.encode(text, add_special_tokens=False)
            to_append.extend(tokens)
            if tokens:
                is_new_word = True
            is_first_word = False
            alignment_idx += 1

        if to_append:
            if is_new_word and t > 0 and text_codes[t - 1] == padding_id:
                text_codes[t - 1] = end_padding_id
            text_codes[t] = to_append.popleft()
            is_new_word = False

    return torch.tensor(text_codes, dtype=torch.long)


def encode_file(
    wav_path: Path,
    json_path: Path,
    mimi,
    tokenizer,
    padding_id: int,
    end_padding_id: int,
    device: str,
    min_left_rms: float = 0.0,
) -> torch.Tensor | None:
    """Encode one stereo WAV + alignment into a [1, 17, T] tensor."""
    wav_np, sr = load_audio(wav_path)
    if wav_np.shape[0] < 2:
        return None

    if min_left_rms > 0:
        import numpy as np
        left_rms = np.sqrt(np.mean(wav_np[0] ** 2))
        if left_rms < min_left_rms:
            return None

    wav_tensor = torch.from_numpy(wav_np).float()
    wav_tensor = resample_if_needed(wav_tensor, sr, SAMPLE_RATE)

    moshi_wav = wav_tensor[0:1]   # [1, samples]
    user_wav = wav_tensor[1:2]    # [1, samples]

    frame_size = int(SAMPLE_RATE / FRAME_RATE)
    # Trim to exact frame boundary
    n_samples = (moshi_wav.shape[1] // frame_size) * frame_size
    if n_samples == 0:
        return None
    moshi_wav = moshi_wav[:, :n_samples]
    user_wav = user_wav[:, :n_samples]
    num_frames = n_samples // frame_size

    with torch.no_grad():
        moshi_codes = mimi.encode(moshi_wav.unsqueeze(0).to(device))  # [1, 8, T]
        user_codes = mimi.encode(user_wav.unsqueeze(0).to(device))    # [1, 8, T]

    # Ensure frame counts match (they should, but guard against rounding)
    T = min(moshi_codes.shape[2], user_codes.shape[2], num_frames)
    moshi_codes = moshi_codes[:, :, :T].cpu()
    user_codes = user_codes[:, :, :T].cpu()

    text_codes = align_text_to_frames(json_path, T, tokenizer, padding_id, end_padding_id)
    text_codes = text_codes.unsqueeze(0).unsqueeze(0)  # [1, 1, T]

    # [1, 17, T] = [text, moshi_audio(8), user_audio(8)]
    codes = torch.cat([text_codes, moshi_codes, user_codes], dim=1)
    return codes


def main():
    parser = argparse.ArgumentParser(description="Pre-encode dialogue dataset for Qwen-Moshi")
    parser.add_argument("--data-dirs", nargs="+", required=True,
                        help="One or more directories containing .wav + .json pairs")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config JSON (default: moshi_qwen_7b.json); "
                             "must match the model you train with (padding ids, vocab).")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="HuggingFace tokenizer name (e.g. Qwen/Qwen2.5-3B or Qwen/Qwen2.5-7B). "
                             "If unset, inferred from --config (7b -> Qwen2.5-7B, else Qwen2.5-3B).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit number of files (0 = no limit)")
    parser.add_argument("--min-left-rms", type=float, default=0.0,
                        help="Skip files where left channel RMS is below this threshold")
    args = parser.parse_args()

    # Load config for padding id (must match the model you train with)
    config_path = args.config or str(REPO_ROOT / "configs" / "moshi_qwen_7b.json")
    with open(config_path) as f:
        config = json.load(f)
    padding_id = config["existing_text_padding_id"]
    end_padding_id = config["existing_text_end_padding_id"]
    print(f"Text padding token id: {padding_id}, end_padding_id: {end_padding_id}")

    # Load Mimi
    print("Loading Mimi codec...")
    from moshi.models.loaders import CheckpointInfo
    ckpt = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    mimi = ckpt.get_mimi(device=args.device)
    mimi.eval()
    print(f"  Mimi loaded: frame_rate={mimi.frame_rate}, sample_rate={mimi.sample_rate}")

    # Load Qwen tokenizer
    print("Loading Qwen tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")

    # Collect all WAV/JSON pairs from all data directories
    pairs: list[tuple[Path, Path]] = []
    for data_dir_str in args.data_dirs:
        data_dir = Path(data_dir_str)
        if not data_dir.exists():
            print(f"  WARNING: {data_dir} does not exist, skipping")
            continue
        wav_files = sorted(data_dir.glob("*.wav"))
        for wav_path in wav_files:
            json_path = wav_path.with_suffix(".json")
            if json_path.exists():
                pairs.append((wav_path, json_path))
        print(f"  {data_dir.name}: {len(wav_files)} WAVs, {sum(1 for w, _ in pairs if w.parent == data_dir)} with JSON")

    if args.max_files > 0:
        pairs = pairs[:args.max_files]
    print(f"\nTotal files to process: {len(pairs)}")

    # Encode
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoded = 0
    skipped = 0
    for idx, (wav_path, json_path) in enumerate(pairs):
        try:
            codes = encode_file(wav_path, json_path, mimi, tokenizer, padding_id, end_padding_id, args.device, min_left_rms=args.min_left_rms)
        except Exception as e:
            print(f"  ERROR on {wav_path.name}: {e}")
            skipped += 1
            continue

        if codes is None:
            skipped += 1
            continue

        out_path = out_dir / f"{encoded:06d}.pt"
        torch.save({"codes": codes, "source": str(wav_path)}, str(out_path))
        encoded += 1

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(pairs)}, encoded {encoded}, skipped {skipped}")

    print(f"\nDone. Encoded {encoded} files, skipped {skipped}.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
