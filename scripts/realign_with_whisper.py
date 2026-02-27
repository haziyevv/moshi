#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Re-align stereo dialogue datasets using Whisper word-level timestamps.

The existing .json alignment files may have inaccurate word-level timing
(e.g. evenly-spread within sentences).  This script runs Whisper on the
left channel (SPEAKER_MAIN / Moshi) of each stereo WAV to obtain precise
word-level timestamps, then overwrites the .json files in-place (with
backup).

Usage::

    python scripts/realign_with_whisper.py \\
        --data-dirs ./daily-talk-contiguous/data_stereo ./multidialog-stereo \\
        --model large-v3 \\
        --device cuda

To verify results afterwards::

    python scripts/realign_with_whisper.py \\
        --data-dirs ./multidialog-stereo \\
        --verify-only --max-files 10
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_left_channel(wav_path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Load the left channel (Moshi / SPEAKER_MAIN) and resample to 16kHz for Whisper."""
    import sphn
    wav, sr = sphn.read(str(wav_path))
    if wav.shape[0] >= 2:
        mono = wav[0]
    else:
        mono = wav[0]
    if sr != target_sr:
        import torch
        import torchaudio.functional as F
        mono = F.resample(
            torch.tensor(mono, dtype=torch.float32), sr, target_sr
        ).numpy()
    return mono


def transcribe_with_whisper(model, audio: np.ndarray, language: str = "en"):
    """Run faster-whisper with word timestamps, return list of word dicts."""
    segments, info = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
    )

    words = []
    for segment in segments:
        if segment.words is None:
            continue
        for w in segment.words:
            words.append({
                "word": w.word.strip(),
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "probability": round(w.probability, 3),
            })
    return words


def words_to_alignment_json(words: list[dict]) -> dict:
    """Convert Whisper word list to the dataset's JSON format."""
    alignments = []
    for w in words:
        if not w["word"]:
            continue
        alignments.append([
            w["word"],
            [w["start"], w["end"]],
            "SPEAKER_MAIN",
        ])
    return {"alignments": alignments}


def verify_alignment(wav_path: Path, json_path: Path, label: str = ""):
    """Print quality metrics for a single alignment file."""
    import sphn
    wav, sr = sphn.read(str(wav_path))
    left = wav[0]
    duration = len(left) / sr

    with open(json_path) as f:
        data = json.load(f)
    aligns = data.get("alignments", [])
    if not aligns:
        print(f"  {label}: NO ALIGNMENTS")
        return

    durs = [a[1][1] - a[1][0] for a in aligns]

    # Energy check: RMS during words vs during gaps
    word_rms_list = []
    for _, (start, end), _ in aligns:
        s, e = int(start * sr), int(end * sr)
        if 0 <= s < len(left) and s < e <= len(left):
            seg = left[s:e]
            word_rms_list.append(np.sqrt(np.mean(seg ** 2)))

    gap_rms_list = []
    for i in range(len(aligns) - 1):
        gap_s = aligns[i][1][1]
        gap_e = aligns[i + 1][1][0]
        if gap_e - gap_s > 0.1:
            s, e = int(gap_s * sr), int(gap_e * sr)
            if 0 <= s < len(left) and s < e <= len(left):
                seg = left[s:e]
                gap_rms_list.append(np.sqrt(np.mean(seg ** 2)))

    mean_word_rms = np.mean(word_rms_list) if word_rms_list else 0
    mean_gap_rms = np.mean(gap_rms_list) if gap_rms_list else 0
    ratio = mean_word_rms / (mean_gap_rms + 1e-8)

    # Duration variance (high = real alignment, low = evenly spread)
    dur_std = np.std(durs)
    dur_cv = dur_std / (np.mean(durs) + 1e-8)  # coefficient of variation

    coverage = sum(durs) / duration * 100

    print(f"  {label}: {len(aligns)} words, "
          f"dur_std={dur_std:.3f} (cv={dur_cv:.2f}), "
          f"word/gap_energy={ratio:.1f}x, "
          f"coverage={coverage:.1f}%")

    if dur_cv < 0.15:
        print(f"    WARNING: low duration variance — may be evenly spread")
    if ratio < 3.0:
        print(f"    WARNING: low word/gap energy contrast")


def main():
    parser = argparse.ArgumentParser(description="Re-align datasets with Whisper")
    parser.add_argument("--data-dirs", nargs="+", required=True,
                        help="Directories containing .wav + .json pairs")
    parser.add_argument("--model", type=str, default="large-v3",
                        help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute-type", type=str, default="float16",
                        help="Compute type for faster-whisper (float16, int8, int8_float16)")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit files to process (0 = all)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating .json.bak backup files")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing alignments, don't re-align")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have a .json.bak (already re-aligned)")
    args = parser.parse_args()

    # Collect all WAV/JSON pairs
    pairs: list[tuple[Path, Path]] = []
    for data_dir_str in args.data_dirs:
        data_dir = Path(data_dir_str)
        if not data_dir.exists():
            print(f"WARNING: {data_dir} does not exist, skipping")
            continue
        wav_files = sorted(data_dir.glob("*.wav"))
        count = 0
        for wav_path in wav_files:
            json_path = wav_path.with_suffix(".json")
            if json_path.exists():
                pairs.append((wav_path, json_path))
                count += 1
        print(f"{data_dir.name}: {count} files with JSON")

    if args.max_files > 0:
        pairs = pairs[:args.max_files]
    print(f"Total files: {len(pairs)}")

    # --- Verify-only mode ---
    if args.verify_only:
        print("\nVerification mode:")
        for wav_path, json_path in pairs:
            verify_alignment(wav_path, json_path, label=wav_path.stem)
        return

    # --- Re-alignment mode ---
    print(f"\nLoading Whisper model '{args.model}' on {args.device}...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(
        args.model, device=args.device, compute_type=args.compute_type,
    )
    print("  Model loaded.")

    processed = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for idx, (wav_path, json_path) in enumerate(pairs):
        # Use a .lock file to coordinate parallel instances safely.
        # The first process to create the lock "owns" this file.
        lock_path = json_path.with_suffix(".json.lock")
        backup_path = json_path.with_suffix(".json.bak")

        if args.skip_existing and backup_path.exists():
            skipped += 1
            continue

        # Atomic claim: try to create lock file exclusively
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            skipped += 1
            continue

        try:
            audio = load_left_channel(wav_path)
            words = transcribe_with_whisper(whisper_model, audio, language=args.language)

            if not words:
                print(f"  WARNING: no words detected in {wav_path.name}, keeping original")
                lock_path.unlink(missing_ok=True)
                skipped += 1
                continue

            new_alignment = words_to_alignment_json(words)

            if not args.no_backup and not backup_path.exists():
                shutil.copy2(json_path, backup_path)

            with open(json_path, "w") as f:
                json.dump(new_alignment, f)

            processed += 1

        except Exception as e:
            print(f"  ERROR on {wav_path.name}: {e}")
            lock_path.unlink(missing_ok=True)
            errors += 1

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (processed + skipped + errors) / elapsed if elapsed > 0 else 0
            remaining = sum(1 for _, jp in pairs[idx+1:]
                           if not jp.with_suffix(".json.bak").exists()
                           and not jp.with_suffix(".json.lock").exists())
            eta = remaining / max(rate, 0.1)
            print(f"  Progress: {idx + 1}/{len(pairs)} "
                  f"({processed} aligned, {skipped} skipped, {errors} errors) "
                  f"[{rate:.1f} files/s, ~{eta / 60:.0f}min remaining]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"  Aligned: {processed}, Skipped: {skipped}, Errors: {errors}")

    # Verify a sample
    print("\nSample verification (first 5 re-aligned files):")
    verified = 0
    for wav_path, json_path in pairs:
        if verified >= 5:
            break
        if json_path.with_suffix(".json.bak").exists():
            verify_alignment(wav_path, json_path, label=wav_path.stem)
            verified += 1


if __name__ == "__main__":
    main()
