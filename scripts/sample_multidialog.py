#!/usr/bin/env python3
"""Download a few sample conversations from MultiDialog for quick listening.

Just grabs 1 chunk, converts a handful of conversations to stereo WAVs,
and saves them so you can listen before committing to the full download.

Usage::

    python scripts/sample_multidialog.py --out-dir ./multidialog-samples --num-samples 10
"""

import argparse
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
    match = re.search(r'_(\d+)[a-z]\.wav$', filename)
    if match:
        return int(match.group(1))
    return -1


def get_speaker_from_filename(filename: str) -> str:
    return filename[-5]


def load_and_resample(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != target_sr:
        wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=target_sr)
        wav = wav_t.squeeze(0).numpy()
    return wav


def main():
    parser = argparse.ArgumentParser(description="Download a few MultiDialog samples")
    parser.add_argument("--out-dir", type=str, default="./multidialog-samples")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of conversations to save")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download just 1 chunk
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = list(api.list_repo_tree(
        "IVLLab/MultiDialog", repo_type="dataset",
        path_in_repo="data/train"
    ))
    chunk_files = [f for f in files if f.path.endswith('.tar.gz')]
    chunk_files.sort(key=lambda x: x.path)

    if not chunk_files:
        print("No chunks found!")
        return

    cf = chunk_files[0]
    print(f"Downloading 1 chunk: {cf.path} ({cf.size / 1e6:.0f} MB)...")
    local_path = hf_hub_download(
        "IVLLab/MultiDialog", cf.path, repo_type="dataset"
    )

    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    print("Extracting...")
    with tarfile.open(local_path, 'r:gz') as tar:
        tar.extractall(extract_dir)

    # Group by conversation
    conversations: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for conv_dir in sorted(extract_dir.iterdir()):
        if not conv_dir.is_dir():
            continue
        for wav_file in sorted(conv_dir.glob("*.wav")):
            idx = parse_utterance_index(wav_file.name)
            speaker = get_speaker_from_filename(wav_file.name)
            conversations[conv_dir.name].append((idx, speaker, str(wav_file)))

    print(f"Found {len(conversations)} conversations in this chunk")

    # Convert a few to stereo
    saved = 0
    for conv_id, utterances in sorted(conversations.items()):
        if saved >= args.num_samples:
            break

        utterances.sort(key=lambda x: x[0])
        speakers = set(s for _, s, _ in utterances)
        speaker_list = sorted(speakers)
        if len(speaker_list) < 2:
            continue

        speaker_to_channel = {speaker_list[0]: 0, speaker_list[1]: 1}

        total_samples = 0
        segments = []
        for idx, speaker, fpath in utterances:
            audio = load_and_resample(fpath, TARGET_SR)
            channel = speaker_to_channel[speaker]
            segments.append((channel, audio))
            total_samples += len(audio)

        stereo = np.zeros((2, total_samples), dtype=np.float32)
        pos = 0
        for channel, audio in segments:
            n = len(audio)
            stereo[channel, pos:pos + n] = audio
            pos += n

        duration = total_samples / TARGET_SR
        out_path = out_dir / f"sample_{saved:02d}_{conv_id}_{duration:.0f}s.wav"
        sf.write(str(out_path), stereo.T, TARGET_SR)
        print(f"  [{saved+1}] {out_path.name}  ({duration:.1f}s, {len(utterances)} turns)")
        saved += 1

    print(f"\nSaved {saved} sample conversations to {out_dir}")
    print("Listen to them and decide if the quality is good enough!")


if __name__ == "__main__":
    main()
