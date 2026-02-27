#!/usr/bin/env python3
"""Convert the full AMI Meeting Corpus to Moshi training format.

Downloads headset audio for each meeting, pairs speakers into stereo WAVs
with companion JSON alignments, then deletes the raw audio to save disk.

Annotations must already be downloaded (run download_ami_sample.py first,
or pass --anno-dir to an existing extraction).

Usage::

    python -u scripts/convert_ami_full.py \
        --anno-dir ./ami-sample/annotations \
        --out-dir ./ami-stereo \
        --segment-secs 30 \
        --max-pairs 3 \
        --workers 4
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

AMI_AUDIO_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
TARGET_SR = 24_000


def get_all_meeting_ids(anno_dir: Path) -> list[str]:
    words_dir = anno_dir / "words"
    ids = set()
    for xml_file in words_dir.glob("*.words.xml"):
        meeting_id = xml_file.name.split(".")[0]
        ids.add(meeting_id)
    return sorted(ids)


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception:
        return False


def download_headset_audio(meeting_id: str, audio_dir: Path) -> dict[int, Path]:
    paths = {}
    for i in range(4):
        fname = f"{meeting_id}.Headset-{i}.wav"
        url = f"{AMI_AUDIO_BASE}/{meeting_id}/audio/{fname}"
        dest = audio_dir / fname
        if download_file(url, dest):
            paths[i] = dest
    return paths


def parse_words_xml(xml_path: Path) -> list[dict]:
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


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    t = torch.from_numpy(audio).float().unsqueeze(0)
    t = F.resample(t, orig_sr, target_sr)
    return t.squeeze(0).numpy()


def create_stereo_segments(
    audio_a: np.ndarray, audio_b: np.ndarray, sr: int,
    words_a: list[dict], words_b: list[dict],
    segment_secs: int, speaker_a: str, speaker_b: str,
) -> list[dict]:
    min_len = min(len(audio_a), len(audio_b))
    audio_a = audio_a[:min_len]
    audio_b = audio_b[:min_len]
    seg_samples = segment_secs * sr
    segments = []

    for seg_start in range(0, min_len - sr, seg_samples):
        seg_end = min(seg_start + seg_samples, min_len)
        t_start = seg_start / sr
        t_end = seg_end / sr

        chunk_a = audio_a[seg_start:seg_end]
        chunk_b = audio_b[seg_start:seg_end]
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

        rms_a = np.sqrt(np.mean(chunk_a ** 2))
        rms_b = np.sqrt(np.mean(chunk_b ** 2))
        if (rms_a > 0.005 or rms_b > 0.005) and len(seg_words) > 3:
            segments.append({
                "stereo_audio": stereo,
                "words": seg_words,
                "start_time": t_start,
                "end_time": t_end,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            })
    return segments


def process_meeting(
    meeting_id: str, anno_dir: Path, audio_cache_dir: Path,
    out_dir: Path, segment_secs: int, max_pairs: int,
    cleanup_audio: bool = True,
) -> int:
    """Process one meeting. Returns number of segments created."""
    words_dir = anno_dir / "words"
    headset_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    speaker_xmls = {}
    for letter in ["A", "B", "C", "D"]:
        xml_path = words_dir / f"{meeting_id}.{letter}.words.xml"
        if xml_path.exists():
            speaker_xmls[letter] = xml_path

    if len(speaker_xmls) < 2:
        return 0

    audio_paths = download_headset_audio(meeting_id, audio_cache_dir)
    if len(audio_paths) < 2:
        return 0

    speaker_data = {}
    for letter, xml_path in speaker_xmls.items():
        headset_idx = headset_map.get(letter)
        if headset_idx is None or headset_idx not in audio_paths:
            continue
        words = parse_words_xml(xml_path)
        if not words:
            continue
        try:
            audio, sr = sf.read(str(audio_paths[headset_idx]))
        except Exception:
            continue
        if audio.ndim == 2:
            audio = audio[:, 0]
        if sr != TARGET_SR:
            audio = resample_audio(audio, sr, TARGET_SR)
        speaker_data[letter] = {"audio": audio, "words": words}

    if len(speaker_data) < 2:
        if cleanup_audio:
            for p in audio_paths.values():
                p.unlink(missing_ok=True)
        return 0

    all_pairs = list(combinations(sorted(speaker_data.keys()), 2))
    all_pairs.sort(
        key=lambda p: len(speaker_data[p[0]]["words"]) + len(speaker_data[p[1]]["words"]),
        reverse=True,
    )
    pairs = all_pairs[:max_pairs] if max_pairs > 0 else all_pairs

    total_segments = 0
    for sa, sb in pairs:
        segments = create_stereo_segments(
            speaker_data[sa]["audio"], speaker_data[sb]["audio"],
            TARGET_SR, speaker_data[sa]["words"], speaker_data[sb]["words"],
            segment_secs, sa, sb,
        )
        for seg_idx, seg in enumerate(segments):
            pair_label = f"{seg['speaker_a']}-{seg['speaker_b']}"
            base = f"{meeting_id}_{pair_label}_{seg_idx:04d}"
            wav_path = out_dir / f"{base}.wav"
            json_path = out_dir / f"{base}.json"

            sf.write(str(wav_path), seg["stereo_audio"].T, TARGET_SR)
            alignments = [
                [w["word"], [w["start"], w["end"]], w["speaker"]]
                for w in seg["words"]
            ]
            with open(json_path, "w") as f:
                json.dump({
                    "source": f"AMI/{meeting_id}",
                    "speakers": {"SPEAKER_MAIN": sa, "SPEAKER_OTHER": sb},
                    "duration": seg["end_time"] - seg["start_time"],
                    "alignments": alignments,
                }, f)
            total_segments += 1

    if cleanup_audio:
        for p in audio_paths.values():
            p.unlink(missing_ok=True)

    return total_segments


def main():
    parser = argparse.ArgumentParser(description="Convert full AMI corpus to Moshi format")
    parser.add_argument("--anno-dir", default="./ami-sample/annotations",
                        help="Path to extracted AMI annotations")
    parser.add_argument("--out-dir", default="./ami-stereo", help="Output directory for stereo segments")
    parser.add_argument("--audio-cache", default="./ami-audio-cache",
                        help="Temp directory for downloaded headset WAVs")
    parser.add_argument("--segment-secs", type=int, default=30)
    parser.add_argument("--max-pairs", type=int, default=3,
                        help="Max speaker pairs per meeting (ranked by word count)")
    parser.add_argument("--keep-audio", action="store_true",
                        help="Don't delete raw headset WAVs after processing")
    parser.add_argument("--meetings", nargs="*", default=None,
                        help="Specific meeting IDs to process (default: all)")
    args = parser.parse_args()

    anno_dir = Path(args.anno_dir)
    out_dir = Path(args.out_dir)
    audio_cache = Path(args.audio_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_cache.mkdir(parents=True, exist_ok=True)

    if args.meetings:
        meeting_ids = args.meetings
    else:
        meeting_ids = get_all_meeting_ids(anno_dir)

    already_done = set()
    for f in out_dir.glob("*.wav"):
        mid = f.stem.split("_")[0]
        already_done.add(mid)

    remaining = [m for m in meeting_ids if m not in already_done]
    print(f"AMI Full Conversion")
    print(f"  Total meetings: {len(meeting_ids)}")
    print(f"  Already done: {len(already_done)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Output: {out_dir}")
    print(f"  Segment length: {args.segment_secs}s, max pairs: {args.max_pairs}")
    print()

    total_segments = 0
    t_start = time.time()

    for i, meeting_id in enumerate(remaining):
        t0 = time.time()
        n_segs = process_meeting(
            meeting_id, anno_dir, audio_cache, out_dir,
            args.segment_secs, args.max_pairs,
            cleanup_audio=not args.keep_audio,
        )
        dt = time.time() - t0
        total_segments += n_segs

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed * 60
        eta = (len(remaining) - i - 1) / max(rate, 0.01)
        print(
            f"  [{i+1:3d}/{len(remaining)}] {meeting_id}: "
            f"{n_segs:3d} segments ({dt:.0f}s) | "
            f"total: {total_segments} | {rate:.1f} meetings/min | ETA: {eta:.0f} min"
        )

    existing_count = len(list(out_dir.glob("*.wav")))
    print(f"\nDone! {total_segments} new segments from {len(remaining)} meetings")
    print(f"Total files in {out_dir}: {existing_count} WAV+JSON pairs")


if __name__ == "__main__":
    main()
