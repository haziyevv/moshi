#!/usr/bin/env python3
"""Generate word-level alignment JSON files for stereo WAV files using faster-whisper.

For each stereo WAV, transcribes the LEFT channel (main/moshi speaker) and produces
a JSON file with word-level timestamps in the format expected by preencode_dataset.py:

    {"alignments": [["word", [start_sec, end_sec], "SPEAKER_MAIN"], ...]}

Usage::

    python scripts/generate_alignments.py \
        --wav-dir ./multidialog-stereo \
        --model large-v3 \
        --device cuda

    # Use specific GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_alignments.py \
        --wav-dir ./multidialog-stereo --model large-v3
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

WHISPER_SR = 16_000


def main():
    parser = argparse.ArgumentParser(description="Generate word-level alignments with faster-whisper")
    parser.add_argument("--wav-dir", type=str, required=True,
                        help="Directory containing stereo WAV files (e.g. 0.wav, 1.wav, ...)")
    parser.add_argument("--model", type=str, default="large-v3",
                        help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--compute-type", type=str, default="float16",
                        help="Compute type (float16, int8_float16, int8)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for batched inference")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max files to process (0 = all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have a .json alignment file")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code for transcription")
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)

    # Find all WAV files (numbered files from prepare_multidialog.py)
    wav_files = sorted(wav_dir.glob("*.wav"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    if args.max_files > 0:
        wav_files = wav_files[:args.max_files]

    print(f"Found {len(wav_files)} WAV files in {wav_dir}")

    # Check how many already have alignments
    if args.skip_existing:
        wav_files = [f for f in wav_files if not f.with_suffix(".json").exists()]
        print(f"  {len(wav_files)} remaining after skipping existing")

    if not wav_files:
        print("Nothing to do!")
        return

    # Load model
    print(f"Loading faster-whisper model '{args.model}' on {args.device}...")
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("  Model loaded.")

    n_done = 0
    n_failed = 0

    for i, wav_path in enumerate(wav_files):
        json_path = wav_path.with_suffix(".json")

        try:
            # Load audio and extract left channel (main/moshi speaker)
            try:
                import sphn
                wav_data, sr = sphn.read(str(wav_path))
            except Exception:
                import soundfile as sf
                wav_data, sr = sf.read(str(wav_path), dtype="float32")
                wav_data = wav_data.T if wav_data.ndim > 1 else wav_data[np.newaxis, :]

            if wav_data.ndim == 1:
                wav_data = wav_data[np.newaxis, :]

            # Left channel = main speaker
            left_channel = wav_data[0].astype(np.float32)

            # Resample to 16kHz if needed — faster-whisper assumes numpy arrays are 16kHz
            if sr != WHISPER_SR:
                import torch
                import torchaudio
                audio_16k = torchaudio.functional.resample(
                    torch.from_numpy(left_channel).unsqueeze(0),
                    orig_freq=sr, new_freq=WHISPER_SR,
                ).squeeze(0).numpy()
            else:
                audio_16k = left_channel

            # Run whisper with word timestamps
            segments, info = model.transcribe(
                audio_16k,
                language=args.language,
                word_timestamps=True,
                beam_size=5,
                vad_filter=True,
            )

            # Collect word-level alignments (keep all words, no filtering)
            alignments = []
            for segment in segments:
                if segment.words:
                    for word_info in segment.words:
                        alignments.append([
                            word_info.word.strip(),
                            [round(word_info.start, 3), round(word_info.end, 3)],
                            "SPEAKER_MAIN"
                        ])

            # Save alignment JSON
            with open(json_path, "w") as f:
                json.dump({"alignments": alignments}, f)

            n_done += 1

        except Exception as e:
            print(f"  FAILED {wav_path.name}: {e}")
            n_failed += 1
            continue

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(wav_files)}, done {n_done}, failed {n_failed}")

    print(f"\nDone! Generated {n_done} alignment files, {n_failed} failures.")


if __name__ == "__main__":
    main()
