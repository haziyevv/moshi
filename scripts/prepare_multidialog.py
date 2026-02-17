#!/usr/bin/env python3
"""Convert MultiDialog dataset (HuggingFace) to stereo WAV chunks for Moshi training.

MultiDialog has individual mono WAV files per utterance + transcripts in
metadata.jsonl. This script:
1. Downloads audio chunks from HuggingFace (or uses already-downloaded ones)
2. Downloads metadata.jsonl for ground-truth transcripts
3. Groups utterances by conversation
4. Concatenates turns into ~30s stereo WAV chunks:
   - Left channel (0) = first speaker (main/moshi speaker)
   - Right channel (1) = second speaker (user speaker)
5. Generates word-level alignment JSONs from the known transcripts
   (no Whisper needed!)
6. Resamples from 16kHz to 24kHz (Mimi's sample rate)
7. Saves stereo WAVs + alignment JSONs + manifest

Output is directly compatible with preencode_dataset.py.

Usage::

    # Download and convert all training data
    python scripts/prepare_multidialog.py --out-dir ./multidialog-stereo

    # Use already-downloaded/extracted chunks
    python scripts/prepare_multidialog.py --out-dir ./multidialog-stereo \\
        --chunks-dir ./multidialog-stereo/extracted

    # Limit to N conversations (for testing)
    python scripts/prepare_multidialog.py --out-dir ./multidialog-stereo --max-conversations 100
"""

import argparse
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


TARGET_SR = 24000


def parse_utterance_index(filename: str) -> int:
    """Extract the utterance index from a filename like 'convid_3c.wav' -> 3."""
    match = re.search(r'_(\d+)[a-z]\.wav$', filename)
    if match:
        return int(match.group(1))
    return -1


def get_speaker_from_filename(filename: str) -> str:
    """Extract speaker code from filename: last letter before .wav."""
    return filename[-5]  # e.g., '...3c.wav' -> 'c'


def load_and_resample(path: str, target_sr: int) -> np.ndarray:
    """Load a mono WAV file and resample to target_sr. Returns 1D float32 array."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]

    if sr != target_sr:
        wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=target_sr)
        wav = wav_t.squeeze(0).numpy()

    return wav.astype(np.float32)


def build_alignments_from_text(segments_info: list[dict], target_sr: int) -> list:
    """Build word-level alignments from known text and audio timing.

    Only includes the main speaker (channel 0) since that's what
    preencode_dataset.py uses for the text stream.

    Each segment_info has: channel, text, start_sample, end_sample.
    """
    alignments = []

    for seg in segments_info:
        if seg["channel"] != 0:
            continue

        text = seg.get("text", "")
        if not text:
            continue

        words = text.split()
        if not words:
            continue

        start_sec = seg["start_sample"] / target_sr
        end_sec = seg["end_sample"] / target_sr
        duration = end_sec - start_sec

        if duration <= 0:
            continue

        word_duration = duration / len(words)
        for j, word in enumerate(words):
            w_start = start_sec + j * word_duration
            w_end = start_sec + (j + 1) * word_duration
            alignments.append([
                word,
                [round(w_start, 3), round(w_end, 3)],
                "SPEAKER_MAIN",
            ])

    return alignments


def flush_chunk(segments, segments_info, out_dir, output_idx, target_sr,
                min_samples, manifest_entries):
    """Build a stereo WAV + alignment JSON from accumulated segments and save.

    Returns new output_idx and duration added.
    """
    if not segments:
        return output_idx, 0.0

    total_samples = sum(a.shape[0] for _, a in segments)
    if total_samples < min_samples:
        return output_idx, 0.0

    # Build stereo audio
    stereo = np.zeros((2, total_samples), dtype=np.float32)
    pos = 0
    for channel, audio in segments:
        n = len(audio)
        stereo[channel, pos:pos + n] = audio
        pos += n

    chunk_name = f"{output_idx:06d}"
    wav_path = out_dir / f"{chunk_name}.wav"
    json_path = out_dir / f"{chunk_name}.json"

    # Save stereo WAV
    sf.write(str(wav_path), stereo.T, target_sr)

    # Build and save alignments from ground-truth text
    alignments = build_alignments_from_text(segments_info, target_sr)
    with open(json_path, "w") as f:
        json.dump({"alignments": alignments}, f)

    duration = total_samples / target_sr
    manifest_entries.append({
        "path": f"{chunk_name}.wav",
        "duration": round(duration, 2),
    })

    return output_idx + 1, duration


def load_metadata(cache_dir: str | None = None) -> dict[str, dict[int, str]]:
    """Download metadata.jsonl and return {conv_id: {utterance_id: text}}."""
    from huggingface_hub import hf_hub_download

    print("Downloading metadata.jsonl for transcripts...")
    meta_path = hf_hub_download(
        "IVLLab/MultiDialog", "metadata.jsonl",
        repo_type="dataset", cache_dir=cache_dir,
    )

    transcripts: dict[str, dict[int, str]] = defaultdict(dict)
    with open(meta_path) as f:
        for line in f:
            row = json.loads(line)
            conv_id = row["conv_id"]
            utt_id = int(row["utterance_id"])
            text = row.get("value", "")
            transcripts[conv_id][utt_id] = text

    print(f"  Loaded transcripts for {len(transcripts)} conversations")
    return dict(transcripts)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MultiDialog to stereo WAV chunks + alignments for Moshi"
    )
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for stereo WAV chunks + alignments + manifest")
    parser.add_argument("--chunks-dir", type=str, default=None,
                        help="Directory with already-extracted conversation folders. "
                             "If not provided, downloads from HuggingFace.")
    parser.add_argument("--hf-cache-dir", type=str, default=None,
                        help="Custom HuggingFace cache directory")
    parser.add_argument("--max-conversations", type=int, default=0,
                        help="Limit number of conversations (0 = all)")
    parser.add_argument("--chunk-sec", type=float, default=30.0,
                        help="Target chunk duration in seconds (default 30)")
    parser.add_argument("--min-chunk-sec", type=float, default=5.0,
                        help="Discard chunks shorter than this (seconds)")
    parser.add_argument("--split", type=str, default="train",
                        help="Which split to process (train, valid_freq, etc.)")
    parser.add_argument("--num-chunks", type=int, default=0,
                        help="Number of HF tar.gz chunks to download (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_samples = int(args.chunk_sec * TARGET_SR)
    min_chunk_samples = int(args.min_chunk_sec * TARGET_SR)

    # --- Load transcripts ---
    transcripts = load_metadata(args.hf_cache_dir)

    # --- Get audio data ---
    if args.chunks_dir:
        data_root = Path(args.chunks_dir)
    else:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        files = list(api.list_repo_tree(
            "IVLLab/MultiDialog", repo_type="dataset",
            path_in_repo=f"data/{args.split}"
        ))
        chunk_files = [f for f in files if f.path.endswith('.tar.gz')]
        chunk_files.sort(key=lambda x: x.path)

        if args.num_chunks > 0:
            chunk_files = chunk_files[:args.num_chunks]

        print(f"Downloading {len(chunk_files)} audio chunks from IVLLab/MultiDialog ({args.split})...")

        data_root = out_dir / "extracted"
        data_root.mkdir(exist_ok=True)

        for i, cf in enumerate(chunk_files):
            print(f"  [{i+1}/{len(chunk_files)}] Downloading {cf.path} ({cf.size/1e6:.0f} MB)...")
            local_path = hf_hub_download(
                "IVLLab/MultiDialog", cf.path, repo_type="dataset",
                cache_dir=args.hf_cache_dir
            )
            print(f"    Extracting...")
            with tarfile.open(local_path, 'r:gz') as tar:
                tar.extractall(data_root)

    # --- Group files by conversation ---
    print(f"\nScanning {data_root} for conversations...")
    conversations: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

    for conv_dir in sorted(data_root.iterdir()):
        if not conv_dir.is_dir():
            continue
        conv_id = conv_dir.name
        for wav_file in sorted(conv_dir.glob("*.wav")):
            idx = parse_utterance_index(wav_file.name)
            speaker = get_speaker_from_filename(wav_file.name)
            conversations[conv_id].append((idx, speaker, str(wav_file)))

    print(f"Found {len(conversations)} conversations")

    if args.max_conversations > 0:
        conv_ids = sorted(conversations.keys())[:args.max_conversations]
        conversations = {k: conversations[k] for k in conv_ids}
        print(f"Limited to {len(conversations)} conversations")

    # --- Convert to ~30s stereo chunks with alignments ---
    print(f"\nConverting to stereo WAV chunks (~{args.chunk_sec}s each, sr={TARGET_SR})...")
    output_idx = 0
    n_convs_used = 0
    n_convs_skipped = 0
    total_duration = 0.0
    manifest_entries: list[dict] = []

    for i, (conv_id, utterances) in enumerate(sorted(conversations.items())):
        utterances.sort(key=lambda x: x[0])

        # Determine speaker-to-channel mapping
        speakers = set(s for _, s, _ in utterances)
        speaker_list = sorted(speakers)
        if len(speaker_list) < 2:
            n_convs_skipped += 1
            continue

        speaker_to_channel = {speaker_list[0]: 0, speaker_list[1]: 1}

        # Get transcripts for this conversation
        conv_texts = transcripts.get(conv_id, {})

        # Accumulate turns into ~30s chunks
        current_segments: list[tuple[int, np.ndarray]] = []
        current_info: list[dict] = []
        current_samples = 0

        for idx, speaker, fpath in utterances:
            audio = load_and_resample(fpath, TARGET_SR)
            channel = speaker_to_channel[speaker]
            text = conv_texts.get(idx, "")

            start_sample = current_samples
            current_segments.append((channel, audio))
            current_samples += len(audio)

            current_info.append({
                "channel": channel,
                "text": text,
                "start_sample": start_sample,
                "end_sample": current_samples,
            })

            # Flush when we've accumulated enough
            if current_samples >= chunk_samples:
                output_idx, dur = flush_chunk(
                    current_segments, current_info, out_dir, output_idx,
                    TARGET_SR, min_chunk_samples, manifest_entries,
                )
                total_duration += dur
                current_segments = []
                current_info = []
                current_samples = 0

        # Flush remaining turns from this conversation
        if current_segments:
            output_idx, dur = flush_chunk(
                current_segments, current_info, out_dir, output_idx,
                TARGET_SR, min_chunk_samples, manifest_entries,
            )
            total_duration += dur

        n_convs_used += 1

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(conversations)} conversations -> "
                  f"{output_idx} chunks, {total_duration/3600:.1f}h, "
                  f"skipped {n_convs_skipped}")

    # Write manifest
    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone!")
    print(f"  Conversations: {n_convs_used} used, {n_convs_skipped} skipped")
    print(f"  Output chunks: {output_idx}")
    print(f"  Total duration: {total_duration/3600:.1f} hours")
    print(f"  Manifest: {manifest_path}")
    print(f"\nNext step — pre-encode for training (no Whisper needed!):")
    print(f"  python scripts/preencode_dataset.py \\")
    print(f"      --data-dir {out_dir} \\")
    print(f"      --jsonl {manifest_path} \\")
    print(f"      --out-dir ./encoded_codes")


if __name__ == "__main__":
    main()
