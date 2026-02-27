#!/usr/bin/env python3
"""Download a small AMI Meeting Corpus sample and convert to Moshi training format.

Downloads headset audio for one meeting + word-level annotations, then creates
stereo WAV files (speaker A left, speaker B right) with companion JSON
alignment files in the same format expected by preencode_dataset.py.

Usage::

    python scripts/download_ami_sample.py --meeting ES2002a --out-dir ./ami-sample
"""

import argparse
import io
import json
import os
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf


AMI_AUDIO_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
AMI_ANNO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"

SEGMENT_SECONDS = 30
TARGET_SR = 24_000


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    if dest.exists():
        print(f"  [skip] {desc or dest.name} already exists")
        return True
    print(f"  Downloading {desc or url} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  -> {dest.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return False


def download_headset_audio(meeting_id: str, out_dir: Path, n_speakers: int = 4):
    """Download individual headset WAV files for a meeting."""
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for i in range(n_speakers):
        fname = f"{meeting_id}.Headset-{i}.wav"
        url = f"{AMI_AUDIO_BASE}/{meeting_id}/audio/{fname}"
        dest = audio_dir / fname
        if download_file(url, dest, fname):
            paths[i] = dest
    return paths


def download_annotations(out_dir: Path):
    """Download and extract the manual annotations ZIP."""
    anno_dir = out_dir / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "ami_annotations.zip"

    if not zip_path.exists():
        print(f"  Downloading annotations ({AMI_ANNO_URL}) ...")
        urllib.request.urlretrieve(AMI_ANNO_URL, str(zip_path))
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"  -> ami_annotations.zip ({size_mb:.1f} MB)")

    import zipfile
    if not (anno_dir / "words").exists():
        print("  Extracting annotations ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(anno_dir)
        print("  -> extracted")
    return anno_dir


def parse_words_xml(xml_path: Path) -> list[dict]:
    """Parse an NXT words XML file, returning list of {word, start, end}.

    Skips punctuation-only elements (``punc="true"``) and vocalsounds.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    words = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag != "w":
            continue
        if elem.get("punc") == "true":
            continue

        start_str = elem.get("starttime")
        end_str = elem.get("endtime")
        text = elem.text
        if start_str and end_str and text:
            try:
                start = float(start_str)
                end = float(end_str)
                if start < end and text.strip():
                    words.append({"word": text.strip(), "start": start, "end": end})
            except ValueError:
                continue
    return words


def find_words_files(anno_dir: Path, meeting_id: str) -> dict[str, Path]:
    """Find word-level annotation XML files for each speaker in a meeting.

    AMI naming convention: ``{meeting_id}.{A,B,C,D}.words.xml``
    Speaker letter maps directly to headset channel (A=0, B=1, C=2, D=3).

    Returns dict mapping speaker label (e.g. 'A', 'B') to file path.
    """
    words_dir = anno_dir / "words"
    if not words_dir.exists():
        print(f"  [WARN] words directory not found at {words_dir}")
        return {}

    speaker_files = {}
    for letter in ["A", "B", "C", "D"]:
        xml_file = words_dir / f"{meeting_id}.{letter}.words.xml"
        if xml_file.exists():
            speaker_files[letter] = xml_file
    return speaker_files


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample using torchaudio."""
    if orig_sr == target_sr:
        return audio
    import torch
    import torchaudio.functional as F
    t = torch.from_numpy(audio).float().unsqueeze(0)  # [1, T]
    t = F.resample(t, orig_sr, target_sr)
    return t.squeeze(0).numpy()


def create_stereo_segments(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sr: int,
    words_a: list[dict],
    words_b: list[dict],
    segment_secs: int,
    speaker_a_label: str,
    speaker_b_label: str,
) -> list[dict]:
    """Create fixed-length stereo segments from two speaker audio streams.

    Returns list of dicts with keys: stereo_audio, words, start_time, end_time.
    """
    min_len = min(len(audio_a), len(audio_b))
    audio_a = audio_a[:min_len]
    audio_b = audio_b[:min_len]
    total_duration = min_len / sr

    segments = []
    seg_samples = segment_secs * sr

    for seg_start_sample in range(0, min_len - sr, seg_samples):
        seg_end_sample = min(seg_start_sample + seg_samples, min_len)
        t_start = seg_start_sample / sr
        t_end = seg_end_sample / sr

        chunk_a = audio_a[seg_start_sample:seg_end_sample]
        chunk_b = audio_b[seg_start_sample:seg_end_sample]
        stereo = np.stack([chunk_a, chunk_b], axis=0)

        seg_words = []
        for w in words_a:
            if w["start"] >= t_start and w["end"] <= t_end:
                seg_words.append({
                    "word": w["word"],
                    "start": w["start"] - t_start,
                    "end": w["end"] - t_start,
                    "speaker": "SPEAKER_MAIN",
                })
        for w in words_b:
            if w["start"] >= t_start and w["end"] <= t_end:
                seg_words.append({
                    "word": w["word"],
                    "start": w["start"] - t_start,
                    "end": w["end"] - t_start,
                    "speaker": "SPEAKER_OTHER",
                })
        seg_words.sort(key=lambda x: x["start"])

        rms_a = np.sqrt(np.mean(chunk_a**2))
        rms_b = np.sqrt(np.mean(chunk_b**2))
        has_speech = rms_a > 0.005 or rms_b > 0.005

        if has_speech and len(seg_words) > 3:
            segments.append({
                "stereo_audio": stereo,
                "words": seg_words,
                "start_time": t_start,
                "end_time": t_end,
                "speaker_a": speaker_a_label,
                "speaker_b": speaker_b_label,
            })

    return segments


def save_segment(segment: dict, idx: int, out_dir: Path, sr: int, meeting_id: str):
    """Save one segment as stereo WAV + alignment JSON."""
    pair_label = f"{segment['speaker_a']}-{segment['speaker_b']}"
    base_name = f"{meeting_id}_{pair_label}_{idx:04d}"

    wav_path = out_dir / f"{base_name}.wav"
    json_path = out_dir / f"{base_name}.json"

    sf.write(str(wav_path), segment["stereo_audio"].T, sr)

    alignments = [
        [w["word"], [w["start"], w["end"]], w["speaker"]]
        for w in segment["words"]
    ]
    alignment_data = {
        "source": f"AMI/{meeting_id}",
        "speakers": {
            "SPEAKER_MAIN": segment["speaker_a"],
            "SPEAKER_OTHER": segment["speaker_b"],
        },
        "duration": segment["end_time"] - segment["start_time"],
        "alignments": alignments,
    }
    with open(json_path, "w") as f:
        json.dump(alignment_data, f, indent=2)

    return wav_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Download AMI sample and convert to Moshi format")
    parser.add_argument("--meeting", default="ES2002a", help="Meeting ID to download")
    parser.add_argument("--out-dir", default="./ami-sample", help="Output directory")
    parser.add_argument("--segment-secs", type=int, default=SEGMENT_SECONDS)
    parser.add_argument("--max-pairs", type=int, default=2,
                        help="Max speaker pairs to process (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== AMI Sample Extraction: {args.meeting} ===\n")

    # 1. Download headset audio
    print("[1/4] Downloading headset audio ...")
    audio_paths = download_headset_audio(args.meeting, out_dir)
    if len(audio_paths) < 2:
        print("ERROR: Need at least 2 headset files")
        sys.exit(1)
    print(f"  Got {len(audio_paths)} headset files\n")

    # 2. Download annotations
    print("[2/4] Downloading word annotations ...")
    anno_dir = download_annotations(out_dir)

    speaker_files = find_words_files(anno_dir, args.meeting)
    print(f"  Found word files for speakers: {list(speaker_files.keys())}\n")
    if len(speaker_files) < 2:
        print("ERROR: Need word annotations for at least 2 speakers")
        sys.exit(1)

    # 3. Parse annotations and load audio
    print("[3/4] Parsing annotations and loading audio ...")
    speaker_data = {}

    headset_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    for speaker_label, xml_path in sorted(speaker_files.items()):
        headset_idx = headset_map.get(speaker_label)
        if headset_idx is None or headset_idx not in audio_paths:
            print(f"  Speaker {speaker_label}: no matching headset audio, skipping")
            continue

        words = parse_words_xml(xml_path)
        print(f"  Speaker {speaker_label} (Headset-{headset_idx}): {len(words)} words")
        if not words:
            continue

        audio, sr = sf.read(str(audio_paths[headset_idx]))
        print(f"    Audio: {len(audio)/sr:.1f}s @ {sr}Hz")

        if sr != TARGET_SR:
            print(f"    Resampling {sr} -> {TARGET_SR} Hz ...")
            audio = resample_audio(audio, sr, TARGET_SR)

        speaker_data[speaker_label] = {"audio": audio, "words": words}

    print()

    # 4. Create speaker pairs and segment
    print(f"[4/4] Creating stereo segments ({args.segment_secs}s each) ...")
    data_dir = out_dir / "data_stereo"
    data_dir.mkdir(parents=True, exist_ok=True)

    speaker_labels = sorted(speaker_data.keys())
    all_pairs = list(combinations(speaker_labels, 2))
    all_pairs.sort(
        key=lambda p: len(speaker_data[p[0]]["words"]) + len(speaker_data[p[1]]["words"]),
        reverse=True,
    )
    pairs = all_pairs[:args.max_pairs] if args.max_pairs > 0 else all_pairs
    counts = ", ".join(f"{s}={len(speaker_data[s]['words'])}" for s in speaker_labels)
    print(f"  Speaker word counts: {counts}")
    print(f"  Selected {len(pairs)} pairs (ranked by total words): {pairs}")

    total_segments = 0
    for pair_idx, (sa, sb) in enumerate(pairs):
        print(f"\n  Pair {pair_idx + 1}: {sa} (left/Moshi) + {sb} (right/user)")
        segments = create_stereo_segments(
            speaker_data[sa]["audio"],
            speaker_data[sb]["audio"],
            TARGET_SR,
            speaker_data[sa]["words"],
            speaker_data[sb]["words"],
            args.segment_secs,
            sa, sb,
        )
        print(f"    -> {len(segments)} segments with speech")

        for seg_idx, seg in enumerate(segments):
            wav_p, json_p = save_segment(seg, seg_idx, data_dir, TARGET_SR, args.meeting)
            total_segments += 1

        if segments:
            s = segments[0]
            main_words = [w for w in s["words"] if w["speaker"] == "SPEAKER_MAIN"]
            other_words = [w for w in s["words"] if w["speaker"] == "SPEAKER_OTHER"]
            print(f"    Sample segment 0: {len(main_words)} main words, {len(other_words)} other words")
            if main_words:
                print(f"      Main text: \"{' '.join(w['word'] for w in main_words[:15])}...\"")
            if other_words:
                print(f"      Other text: \"{' '.join(w['word'] for w in other_words[:15])}...\"")

    print(f"\n{'='*60}")
    print(f"Done! Created {total_segments} segments in {data_dir}")
    print(f"Files: {total_segments} x (.wav + .json)")
    print(f"\nTo pre-encode for training:")
    print(f"  python scripts/preencode_dataset.py \\")
    print(f"    --data-dirs {data_dir} \\")
    print(f"    --out-dir ./encoded_ami_sample \\")
    print(f"    --device cuda")


if __name__ == "__main__":
    main()
