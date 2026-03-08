"""
Convert MultiDialog to moshi-finetune format.
Each output file = exactly 2 turns: gpt (moshi) + human (user).

Output:
  stereo-data/
    0.wav   (ch0=gpt, ch1=human)
    0.json  (alignments for gpt/SPEAKER_MAIN only)
    1.wav
    1.json
    ...
    multidialog.jsonl
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# === CONFIG ===
DATA_ROOT = Path("/workspace/moshi/MultiDialog/data/train")
OUTPUT_DIR = Path("/workspace/moshi/MultiDialog/stereo-data")
TARGET_SR = 24000
METADATA_ROOT = Path("/workspace/moshi/MultiDialog/metadata/train")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 1. Load metadata, group by conversation ===
print("Loading metadata...")
conversations = defaultdict(list)
metadata_files = sorted(METADATA_ROOT.glob("train_metadata_*.jsonl"))

for meta_file in metadata_files:
    with open(meta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            conversations[d["conv_id"]].append(d)

for conv_id in conversations:
    conversations[conv_id].sort(key=lambda x: x["utterance_id"])

print(f"Found {len(conversations)} conversations")

# === 2. Extract consecutive (gpt, human) pairs ===
pairs = []
for conv_id, utts in conversations.items():
    for i in range(len(utts) - 1):
        if utts[i]["from"] == "gpt" and utts[i + 1]["from"] == "human":
            pairs.append((utts[i], utts[i + 1]))

print(f"Found {len(pairs)} (gpt, human) turn pairs")

# === 3. Build stereo WAVs ===
manifest = []
skipped = 0

for idx, (gpt_utt, human_utt) in enumerate(tqdm(pairs, desc="Building stereo")):
    gpt_path = DATA_ROOT / gpt_utt["file_name"]
    human_path = DATA_ROOT / human_utt["file_name"]

    if not gpt_path.exists() or not human_path.exists():
        skipped += 1
        continue

    gpt_audio, gpt_sr = sf.read(str(gpt_path), dtype="float32")
    human_audio, human_sr = sf.read(str(human_path), dtype="float32")

    # Mono
    if gpt_audio.ndim > 1:
        gpt_audio = gpt_audio.mean(axis=1)
    if human_audio.ndim > 1:
        human_audio = human_audio.mean(axis=1)

    # Resample
    if gpt_sr != TARGET_SR:
        import librosa
        gpt_audio = librosa.resample(gpt_audio, orig_sr=gpt_sr, target_sr=TARGET_SR).astype(np.float32)
    if human_sr != TARGET_SR:
        import librosa
        human_audio = librosa.resample(human_audio, orig_sr=human_sr, target_sr=TARGET_SR).astype(np.float32)

    # Total length = gpt + human sequentially
    total_samples = len(gpt_audio) + len(human_audio)
    stereo = np.zeros((total_samples, 2), dtype=np.float32)

    # ch0 (left) = gpt/moshi, ch1 (right) = human/user
    stereo[:len(gpt_audio), 0] = gpt_audio
    stereo[len(gpt_audio):, 1] = human_audio

    # Save WAV
    sf.write(str(OUTPUT_DIR / f"{idx}.wav"), stereo, TARGET_SR)

    # Build alignments for gpt channel only
    gpt_start = 0.0
    gpt_end = len(gpt_audio) / TARGET_SR
    gpt_duration = gpt_end - gpt_start

    words = gpt_utt["value"].split()
    alignments = []
    if words:
        word_dur = gpt_duration / len(words)
        for i, word in enumerate(words):
            w_start = round(gpt_start + i * word_dur, 2)
            w_end = round(gpt_start + (i + 1) * word_dur, 2)
            alignments.append([word, [w_start, w_end], "SPEAKER_MAIN"])

    # Save JSON
    with open(OUTPUT_DIR / f"{idx}.json", "w") as f:
        json.dump({"alignments": alignments}, f)

    manifest.append({
        "path": f"{idx}.wav",
        "duration": round(total_samples / TARGET_SR, 4),
    })

# === 4. Write manifest ===
jsonl_path = OUTPUT_DIR / "multidialog.jsonl"
with open(jsonl_path, "w") as f:
    for entry in manifest:
        json.dump(entry, f)
        f.write("\n")

print(f"\nDone!")
print(f"  Pairs converted: {len(manifest)}")
print(f"  Skipped: {skipped}")
print(f"  Output: {OUTPUT_DIR}")
print(f"\nFor precise alignments run:")
print(f"  python annotate.py {jsonl_path}")