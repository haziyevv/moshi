#!/usr/bin/env python3
"""Build a stereo dataset from synthesized dialogue turns for Moshi training.

Reads dialogue directories (dialogue.json + per-turn mono WAVs) and produces:
- Stereo WAV files (left=main speaker, right=user speaker) at 24kHz
- Word-level alignment JSONs (via faster-whisper on the left channel)
- A JSONL manifest

Each dialogue becomes one stereo WAV file (dialogues should already be
chunked to ≤30s by split_dialogues.py).

The output is directly compatible with preencode_dataset.py.

Pipeline overview:
  1. [generate_dialogue_scripts.py]  Generate dialogue text (long, 16-30+ turns)
  2. [split_dialogues.py]            Split into short chunks (6-10 turns each)
  3. [synthesize_dialogues.py]       Synthesize each turn with CosyVoice
  4. [This script]                   Combine into stereo WAVs + alignments + manifest
  5. [preencode_dataset.py]          Encode to training tensors
  6. [train_qwen_moshi.py]           Train

Usage::

    # Basic: build stereo WAVs + manifest (run generate_alignments.py later)
    python scripts/build_stereo_dataset.py \\
        --dialogues-dir ./dialogues \\
        --out-dir ./stereo-dataset

    # Full: also generate word-level alignments with faster-whisper
    python scripts/build_stereo_dataset.py \\
        --dialogues-dir ./dialogues \\
        --out-dir ./stereo-dataset \\
        --generate-alignments \\
        --whisper-model large-v3

    # Quick: approximate text-based alignments (no Whisper needed)
    python scripts/build_stereo_dataset.py \\
        --dialogues-dir ./dialogues \\
        --out-dir ./stereo-dataset \\
        --text-alignments

Output structure::

    stereo-dataset/
      000000.wav        # stereo WAV, 24kHz, ≤30s
      000000.json       # {"alignments": [["word", [start, end], "SPEAKER_MAIN"], ...]}
      000001.wav
      000001.json
      ...
      manifest.jsonl    # {"path": "000000.wav", "duration": 28.5} per line

Then run::

    python scripts/preencode_dataset.py \\
        --data-dir ./stereo-dataset \\
        --jsonl ./stereo-dataset/manifest.jsonl \\
        --out-dir ./encoded_codes
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

TARGET_SR = 24000


def load_mono_wav(path: str, target_sr: int) -> np.ndarray:
    """Load a mono WAV file and resample to target_sr.

    Returns a 1D numpy float32 array.
    """
    try:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32")
    except Exception:
        import sphn
        wav, sr = sphn.read(path)
        if wav.ndim > 1:
            wav = wav[0]

    if wav.ndim > 1:
        wav = wav[:, 0] if wav.shape[1] < wav.shape[0] else wav[0]

    if sr != target_sr:
        wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=target_sr)
        wav = wav_t.squeeze(0).numpy()

    return wav.astype(np.float32)


def build_stereo_from_turns(
    dialogue: dict,
    dialogue_dir: Path,
    target_sr: int,
    silence_between_turns: float = 0.3,
) -> tuple[np.ndarray, list[dict]] | None:
    """Build a stereo WAV from dialogue turns.

    Returns:
        (stereo_audio, turn_info) where stereo_audio is [2, samples] and
        turn_info has timing metadata per turn, or None on failure.
    """
    turns = dialogue["turns"]

    segments: list[dict] = []
    for turn in turns:
        turn_idx = turn["turn_idx"]
        wav_path = dialogue_dir / f"turn_{turn_idx:02d}.wav"

        if not wav_path.exists():
            print(f"    Missing {wav_path.name}, skipping dialogue")
            return None

        audio = load_mono_wav(str(wav_path), target_sr)
        channel = 0 if turn["role"] == "main" else 1

        segments.append({
            "audio": audio,
            "channel": channel,
            "role": turn["role"],
            "text": turn["text"],
            "turn_idx": turn_idx,
        })

    if not segments:
        return None

    silence_samples = int(silence_between_turns * target_sr)

    # Calculate total length
    total_samples = 0
    for seg in segments:
        total_samples += len(seg["audio"]) + silence_samples
    total_samples -= silence_samples  # no trailing silence

    stereo = np.zeros((2, total_samples), dtype=np.float32)
    pos = 0
    turn_info = []

    for seg in segments:
        n = len(seg["audio"])
        start_sec = pos / target_sr
        end_sec = (pos + n) / target_sr

        stereo[seg["channel"], pos:pos + n] = seg["audio"]

        turn_info.append({
            "role": seg["role"],
            "text": seg["text"],
            "turn_idx": seg["turn_idx"],
            "channel": seg["channel"],
            "start_sec": round(start_sec, 4),
            "end_sec": round(end_sec, 4),
        })

        pos += n + silence_samples

    return stereo, turn_info


WHISPER_SR = 16_000


def generate_alignment_whisper(
    stereo: np.ndarray,
    whisper_model,
    language: str = "en",
    sample_rate: int = 24000,
) -> list:
    """Generate word-level alignments for the main speaker using faster-whisper.

    Transcribes the left channel (main speaker) and returns alignments in
    the format expected by preencode_dataset.py.
    """
    left_channel = stereo[0].astype(np.float32)

    # Resample to 16kHz — faster-whisper assumes numpy arrays are 16kHz
    if sample_rate != WHISPER_SR:
        audio_16k = torchaudio.functional.resample(
            torch.from_numpy(left_channel).unsqueeze(0),
            orig_freq=sample_rate, new_freq=WHISPER_SR,
        ).squeeze(0).numpy()
    else:
        audio_16k = left_channel

    segments, _ = whisper_model.transcribe(
        audio_16k,
        language=language,
        word_timestamps=True,
        beam_size=5,
        condition_on_previous_text=False,
    )

    alignments = []
    for segment in segments:
        if segment.words:
            for word_info in segment.words:
                alignments.append([
                    word_info.word.strip(),
                    [round(word_info.start, 3), round(word_info.end, 3)],
                    "SPEAKER_MAIN",
                ])

    return alignments


def generate_alignment_from_text(turn_info: list[dict]) -> list:
    """Generate approximate word-level alignments from known text and turn timing.

    Since we know the exact text and the start/end time of each main speaker
    turn, we distribute words proportional to character length within each
    turn's time window.  Less accurate than Whisper but requires no extra model.
    """
    alignments = []

    for turn in turn_info:
        if turn["role"] != "main":
            continue

        words = turn["text"].split()
        if not words:
            continue

        start = turn["start_sec"]
        end = turn["end_sec"]
        duration = end - start

        if duration <= 0:
            continue

        char_counts = [max(len(w), 1) for w in words]
        total_chars = sum(char_counts)

        pos = start
        for j, word in enumerate(words):
            word_dur = duration * char_counts[j] / total_chars
            w_start = pos
            w_end = pos + word_dur
            alignments.append([
                word,
                [round(w_start, 3), round(w_end, 3)],
                "SPEAKER_MAIN",
            ])
            pos = w_end

    return alignments


def save_wav(path: str, stereo: np.ndarray, sample_rate: int):
    """Save a stereo numpy array [2, samples] to WAV."""
    import soundfile as sf
    sf.write(path, stereo.T, sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Build stereo dataset from synthesized dialogue turns."
    )
    parser.add_argument("--dialogues-dir", type=str, required=True,
                        help="Directory with dialogue folders")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for stereo WAVs + alignments + manifest")

    # Audio
    parser.add_argument("--target-sr", type=int, default=24000,
                        help="Target sample rate (24000 for Mimi)")
    parser.add_argument("--silence-sec", type=float, default=0.3,
                        help="Silence between turns in seconds")
    parser.add_argument("--min-duration", type=float, default=3.0,
                        help="Skip dialogues shorter than this (seconds)")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Truncate dialogues longer than this (seconds)")

    # Alignments
    parser.add_argument("--generate-alignments", action="store_true",
                        help="Generate word-level alignments with faster-whisper")
    parser.add_argument("--whisper-model", type=str, default="large-v3",
                        help="Whisper model size for alignment generation")
    parser.add_argument("--whisper-device", type=str, default="cuda",
                        help="Device for Whisper (cuda or cpu)")
    parser.add_argument("--whisper-compute-type", type=str, default="float16",
                        help="Compute type for Whisper")
    parser.add_argument("--text-alignments", action="store_true",
                        help="Generate approximate alignments from text timing "
                             "(no Whisper needed, less accurate)")
    parser.add_argument("--language", type=str, default="en",
                        help="Language for Whisper transcription")

    # Other
    parser.add_argument("--max-dialogues", type=int, default=0,
                        help="Max dialogues to process (0 = all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip output files that already exist")
    args = parser.parse_args()

    dialogues_dir = Path(args.dialogues_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all dialogue directories
    dialogue_dirs = sorted(
        [d for d in dialogues_dir.iterdir()
         if d.is_dir() and (d / "dialogue.json").exists()],
        key=lambda d: d.name,
    )

    if args.max_dialogues > 0:
        dialogue_dirs = dialogue_dirs[:args.max_dialogues]

    print(f"Found {len(dialogue_dirs)} dialogues in {dialogues_dir}")
    print(f"Output: {out_dir}")
    print(f"Target sample rate: {args.target_sr} Hz")
    print(f"Max duration: {args.max_duration}s")

    # Load Whisper if needed
    whisper_model = None
    if args.generate_alignments:
        print(f"Loading faster-whisper model '{args.whisper_model}' on {args.whisper_device}...")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(
            args.whisper_model,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
        )
        print("  Whisper loaded.")

    output_idx = 0
    n_skipped = 0
    n_failed = 0
    n_truncated = 0
    total_duration = 0.0
    manifest_entries = []

    for di, dialogue_dir in enumerate(dialogue_dirs):
        with open(dialogue_dir / "dialogue.json") as f:
            dialogue = json.load(f)

        result = build_stereo_from_turns(
            dialogue, dialogue_dir, args.target_sr,
            silence_between_turns=args.silence_sec,
        )

        if result is None:
            n_failed += 1
            continue

        stereo, turn_info = result
        duration = stereo.shape[1] / args.target_sr

        if duration < args.min_duration:
            n_skipped += 1
            continue

        if duration > args.max_duration:
            max_samples = int(args.max_duration * args.target_sr)
            stereo = stereo[:, :max_samples]
            duration = args.max_duration
            n_truncated += 1

        # Output file paths
        file_name = f"{output_idx:06d}"
        wav_path = out_dir / f"{file_name}.wav"
        json_path = out_dir / f"{file_name}.json"

        if args.skip_existing and wav_path.exists() and json_path.exists():
            output_idx += 1
            continue

        # Generate alignments
        if args.generate_alignments and whisper_model is not None:
            alignments = generate_alignment_whisper(
                stereo, whisper_model, args.language,
                sample_rate=args.target_sr,
            )
        elif args.text_alignments:
            alignments = generate_alignment_from_text(turn_info)
        else:
            alignments = []

        # Save stereo WAV
        save_wav(str(wav_path), stereo, args.target_sr)

        # Save alignment JSON
        with open(json_path, "w") as f:
            json.dump({"alignments": alignments}, f)

        total_duration += duration
        manifest_entries.append({
            "path": f"{file_name}.wav",
            "duration": round(duration, 2),
        })

        output_idx += 1

        if (di + 1) % 50 == 0:
            print(f"  Processed {di+1}/{len(dialogue_dirs)} dialogues -> "
                  f"{output_idx} output files, {total_duration/60:.1f} min, "
                  f"skipped {n_skipped}, failed {n_failed}")

    # Write manifest
    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone!")
    print(f"  Output files:   {output_idx}")
    print(f"  Total duration: {total_duration/3600:.2f} hours ({total_duration/60:.0f} min)")
    print(f"  Truncated:      {n_truncated}")
    print(f"  Failed:         {n_failed}")
    print(f"  Skipped (short):{n_skipped}")
    print(f"  Manifest:       {manifest_path}")
    print(f"\nNext step: pre-encode for training:")
    print(f"  python scripts/preencode_dataset.py \\")
    print(f"      --data-dir {out_dir} \\")
    print(f"      --jsonl {manifest_path} \\")
    print(f"      --out-dir ./encoded_codes")


if __name__ == "__main__":
    main()