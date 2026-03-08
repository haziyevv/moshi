#!/usr/bin/env python3
"""Synthesize dialogue turns into mono WAV files using CosyVoice.

Reads dialogue.json files produced by generate_dialogue_scripts.py and
synthesizes each turn as a separate mono WAV file using CosyVoice-3.

Pipeline overview:
  1. [generate_dialogue_scripts.py]  Generate dialogue text (long, 16-30+ turns)
  2. [split_dialogues.py]            Split into short chunks (6-10 turns each)
  3. [This script]                   Synthesize each turn → mono WAVs
  4. [build_stereo_dataset.py]       Combine into stereo WAVs + manifest
  5. [preencode_dataset.py]          Encode to training tensors
  6. [train_qwen_moshi.py]           Train

Usage::

    python scripts/synthesize_dialogues.py \\
        --dialogues-dir ./dialogues \\
        --cosyvoice-dir /mnt/data/farid_projects/CosyVoice \\
        --skip-existing

    # Process only first 50 dialogues
    python scripts/synthesize_dialogues.py \\
        --dialogues-dir ./dialogues \\
        --cosyvoice-dir /mnt/data/farid_projects/CosyVoice \\
        --max-dialogues 50 --skip-existing

Output structure (added to each dialogue directory)::

    dialogues/
      0000/
        dialogue.json       # from step 1
        turn_00.wav         # mono WAV from CosyVoice
        turn_01.wav
        ...
      0001/
        dialogue.json
        turn_00.wav
        ...
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def init_cosyvoice(cosyvoice_dir: str, hf_token: str | None = None):
    """Initialize CosyVoice-3 model.

    Args:
        cosyvoice_dir: Path to the CosyVoice repo (contains third_party/Matcha-TTS).
        hf_token: HuggingFace token for downloading the model. Falls back to
                  HF_TOKEN env var.
    """
    cosyvoice_dir = str(Path(cosyvoice_dir).resolve())

    # Add CosyVoice paths
    if cosyvoice_dir not in sys.path:
        sys.path.insert(0, cosyvoice_dir)
    matcha_path = os.path.join(cosyvoice_dir, "third_party", "Matcha-TTS")
    if matcha_path not in sys.path:
        sys.path.insert(0, matcha_path)

    from huggingface_hub import snapshot_download
    from cosyvoice.cli.cosyvoice import AutoModel

    token = hf_token or os.environ.get("HF_TOKEN")
    model_path = snapshot_download(
        "identityailabs-com/CosyVoice-3-2026.01.23",
        token=token,
    )
    model = AutoModel(model_dir=model_path, load_vllm=True)
    print(f"  CosyVoice loaded. Sample rate: {model.sample_rate}")
    return model


def set_seeds(seed: int = 1986):
    """Set seeds once at startup. No deterministic mode — max GPU throughput."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # auto-tune kernel selection
    os.environ["PYTHONHASHSEED"] = str(seed)


def synthesize_turn(model, text: str, speaker_id: str) -> tuple[np.ndarray, int]:
    """Synthesize a single text turn with CosyVoice-3.

    Args:
        model: CosyVoice AutoModel instance.
        text: Text to synthesize.
        speaker_id: CosyVoice speaker ID (e.g. "Speaker_Samantha").

    Returns:
        (audio, sample_rate): audio is a 1D numpy float32 array.
    """
    # Text preprocessing (matches gradio_ui.py)
    processed_text = text.replace("\u2019", "'")
    processed_text = re.sub(r"\s{2,}", " ", processed_text).strip()
    processed_text = f"You are a helpfull assistant.<|endofprompt|>{processed_text}"

    with torch.inference_mode():
        outputs = list(model.inference_sft(processed_text, speaker_id, stream=False))

    audio = outputs[0]["tts_speech"].numpy().flatten().astype(np.float32)
    return audio, model.sample_rate


def write_wav(path: str, audio: np.ndarray, sr: int):
    """Write a WAV file (used as async I/O target)."""
    sf.write(path, audio, sr)


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize dialogue turns with CosyVoice-3."
    )
    parser.add_argument("--dialogues-dir", type=str, required=True,
                        help="Directory with dialogue folders (from generate_dialogue_scripts.py)")
    parser.add_argument("--cosyvoice-dir", type=str,
                        default="/mnt/data/farid_projects/CosyVoice",
                        help="Path to the CosyVoice repo directory")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (falls back to HF_TOKEN env var)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip turns that already have a .wav file")
    parser.add_argument("--max-dialogues", type=int, default=0,
                        help="Max dialogues to process (0 = all)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index for sharding (inclusive, default 0)")
    parser.add_argument("--end-idx", type=int, default=0,
                        help="End index for sharding (exclusive, 0 = all)")
    parser.add_argument("--seed", type=int, default=1986,
                        help="Random seed (set once at startup)")
    args = parser.parse_args()

    dialogues_dir = Path(args.dialogues_dir)

    # Find all dialogue directories
    dialogue_dirs = sorted(
        [d for d in dialogues_dir.iterdir()
         if d.is_dir() and (d / "dialogue.json").exists()],
        key=lambda d: d.name,
    )

    if args.end_idx > 0:
        dialogue_dirs = dialogue_dirs[args.start_idx:args.end_idx]
    elif args.start_idx > 0:
        dialogue_dirs = dialogue_dirs[args.start_idx:]

    if args.max_dialogues > 0:
        dialogue_dirs = dialogue_dirs[:args.max_dialogues]

    print(f"Found {len(dialogue_dirs)} dialogues in {dialogues_dir}")

    # Set seeds once — no per-turn reseeding, no deterministic mode
    set_seeds(args.seed)

    # Initialize CosyVoice
    print("Initializing CosyVoice-3...")
    model = init_cosyvoice(args.cosyvoice_dir, args.hf_token)

    n_synthesized = 0
    n_skipped = 0
    n_failed = 0
    t0 = time.time()

    # Background thread pool for async WAV writing
    io_executor = ThreadPoolExecutor(max_workers=2)
    io_futures = []

    for di, dialogue_dir in enumerate(dialogue_dirs):
        with open(dialogue_dir / "dialogue.json") as f:
            dialogue = json.load(f)

        turns = dialogue["turns"]
        speakers = dialogue["speakers"]

        for turn in turns:
            turn_idx = turn["turn_idx"]
            wav_path = dialogue_dir / f"turn_{turn_idx:02d}.wav"

            if args.skip_existing and wav_path.exists():
                n_skipped += 1
                continue

            role = turn["role"]
            speaker_id = speakers[role]["id"]
            text = turn["text"]

            try:
                audio, sr = synthesize_turn(model, text, speaker_id)
                fut = io_executor.submit(write_wav, str(wav_path), audio, sr)
                io_futures.append(fut)
                n_synthesized += 1
            except Exception as e:
                print(f"  FAILED {dialogue_dir.name}/turn_{turn_idx:02d}: {e}")
                n_failed += 1

        if (di + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = n_synthesized / elapsed if elapsed > 0 else 0
            print(f"  {di+1}/{len(dialogue_dirs)} dialogues | "
                  f"{n_synthesized} turns synthesized ({rate:.1f}/sec) | "
                  f"{n_failed} failed | {n_skipped} skipped")

    # Wait for all background writes to finish
    for fut in io_futures:
        try:
            fut.result()
        except Exception as e:
            print(f"  WAV write error: {e}")
            n_failed += 1

    io_executor.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s!")
    print(f"  Synthesized: {n_synthesized} turns")
    print(f"  Failed:      {n_failed}")
    print(f"  Skipped:     {n_skipped}")
    if n_synthesized > 0:
        print(f"  Speed:       {n_synthesized / elapsed:.2f} turns/sec")
    print(f"\nNext step: build stereo dataset:")
    print(f"  python scripts/build_stereo_dataset.py \\")
    print(f"      --dialogues-dir {dialogues_dir} \\")
    print(f"      --out-dir ./stereo-dataset \\")
    print(f"      --generate-alignments")


if __name__ == "__main__":
    main()