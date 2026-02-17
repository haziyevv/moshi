#!/usr/bin/env python3
"""Split long stereo WAVs into fixed-length chunks and generate word-level alignments.

Takes the output of prepare_multidialog.py (long stereo WAVs) and:
1. Splits each into chunks of --chunk-sec seconds
2. Runs faster-whisper on the left channel (main speaker) of each chunk
3. Saves chunk WAVs + alignment JSONs ready for preencode_dataset.py

Usage::

    CUDA_VISIBLE_DEVICES=6 python scripts/split_and_align.py \
        --input-dir ./multidialog-stereo \
        --out-dir ./multidialog-chunks \
        --chunk-sec 30 \
        --model large-v3

"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Split stereo WAVs and generate alignments")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with long stereo WAV files")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for chunks + alignments")
    parser.add_argument("--chunk-sec", type=float, default=30.0,
                        help="Chunk duration in seconds")
    parser.add_argument("--min-chunk-sec", type=float, default=5.0,
                        help="Discard chunks shorter than this")
    parser.add_argument("--model", type=str, default="large-v3",
                        help="Whisper model size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute-type", type=str, default="float16")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max input files to process (0 = all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output chunk already exists")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find input WAV files
    wav_files = sorted(
        [f for f in input_dir.glob("*.wav") if f.stem.isdigit()],
        key=lambda p: int(p.stem)
    )
    if args.max_files > 0:
        wav_files = wav_files[:args.max_files]

    print(f"Found {len(wav_files)} input WAV files")

    # Load Whisper model
    print(f"Loading faster-whisper model '{args.model}' on {args.device}...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("  Model loaded.")

    import sphn

    chunk_idx = 0
    n_skipped = 0
    total_duration = 0.0
    manifest_entries = []

    for file_i, wav_path in enumerate(wav_files):
        # Load stereo audio
        wav_data, sr = sphn.read(str(wav_path))  # [channels, samples]
        if wav_data.ndim == 1:
            wav_data = wav_data[np.newaxis, :]

        total_samples = wav_data.shape[1]
        chunk_samples = int(args.chunk_sec * sr)
        min_chunk_samples = int(args.min_chunk_sec * sr)
        n_chunks = math.ceil(total_samples / chunk_samples)

        for c in range(n_chunks):
            start = c * chunk_samples
            end = min(start + chunk_samples, total_samples)
            chunk_audio = wav_data[:, start:end]

            # Skip short tail chunks
            if chunk_audio.shape[1] < min_chunk_samples:
                n_skipped += 1
                continue

            chunk_name = f"{chunk_idx:06d}"
            chunk_wav_path = out_dir / f"{chunk_name}.wav"
            chunk_json_path = out_dir / f"{chunk_name}.json"

            # Skip if already processed
            if args.skip_existing and chunk_wav_path.exists() and chunk_json_path.exists():
                chunk_idx += 1
                continue

            # Save chunk WAV
            sphn.write_wav(str(chunk_wav_path), np.ascontiguousarray(chunk_audio), sr)

            # Run Whisper on left channel (main speaker)
            left_channel = chunk_audio[0].astype(np.float32)

            try:
                segments, _ = whisper_model.transcribe(
                    left_channel,
                    language=args.language,
                    word_timestamps=True,
                    beam_size=5,
                    vad_filter=True,
                )

                alignments = []
                for segment in segments:
                    if segment.words:
                        for word_info in segment.words:
                            alignments.append([
                                word_info.word.strip(),
                                [round(word_info.start, 3), round(word_info.end, 3)],
                                "SPEAKER_MAIN"
                            ])
            except Exception as e:
                print(f"  Whisper failed on chunk {chunk_name}: {e}")
                alignments = []

            # Save alignment JSON
            with open(chunk_json_path, "w") as f:
                json.dump({"alignments": alignments}, f)

            chunk_duration = chunk_audio.shape[1] / sr
            total_duration += chunk_duration
            manifest_entries.append({
                "path": f"{chunk_name}.wav",
                "duration": round(chunk_duration, 2)
            })

            chunk_idx += 1

        if (file_i + 1) % 50 == 0:
            print(f"  Processed {file_i+1}/{len(wav_files)} input files -> "
                  f"{chunk_idx} chunks, {total_duration/3600:.1f}h, skipped {n_skipped}")

    # Write manifest
    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone! Created {chunk_idx} chunks in {out_dir}")
    print(f"Total duration: {total_duration/3600:.1f}h")
    print(f"Skipped {n_skipped} short tail chunks")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
