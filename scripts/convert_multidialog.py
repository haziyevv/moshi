"""
Convert MultiDialog to moshi-finetune format.

Input:
  /workspace/moshi/MultiDialog/data/train/
    - train_metadata_*.jsonl  (14 files)
    - t_<conv_id>/<utterance>.wav  (per-utterance mono WAVs)

Output:
  /workspace/moshi/MultiDialog/stereo-data/
    - 0.wav, 0.json, 1.wav, 1.json, ...
    - multidialog.jsonl  (manifest)

Stereo format: ch0 (left) = moshi/gpt, ch1 (right) = user/human
JSON format: {"alignments": [["word", [start, end], "SPEAKER_MAIN"], ...]}
  - Only moshi/gpt channel words are included (SPEAKER_MAIN)
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
TARGET_SR = 24000  # Moshi/Mimi expects 24kHz
METADATA_ROOT = Path("/workspace/moshi/MultiDialog/metadata/train")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 1. Load all metadata and group by conversation ===
print("Loading metadata...")
conversations = defaultdict(list)
metadata_files = sorted(METADATA_ROOT.glob("train_metadata_*.jsonl"))
print(f"Found {len(metadata_files)} metadata files")

for meta_file in metadata_files:
    with open(meta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            conversations[d["conv_id"]].append(d)

# Sort each conversation by utterance_id
for conv_id in conversations:
    conversations[conv_id].sort(key=lambda x: x["utterance_id"])

print(f"Found {len(conversations)} conversations, {sum(len(v) for v in conversations.values())} total utterances")

# === 2. Build stereo WAVs + alignment JSONs ===
manifest = []
skipped = 0
idx = 0

for conv_id, utts in tqdm(sorted(conversations.items()), desc="Building stereo"):
    # Load all utterance audio
    segments = []
    valid = True

    for utt in utts:
        import pdb; pdb.set_trace()
        wav_path = DATA_ROOT / utt["file_name"]
        if not wav_path.exists():
            print(f"  Missing: {wav_path}")
            valid = False
            break

        audio, sr = sf.read(str(wav_path), dtype="float32")

        # Handle stereo source -> mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)

        segments.append({
            "audio": audio,
            "from": utt["from"],       # "gpt" or "human"
            "text": utt["value"],
        })

    if not valid or not segments:
        skipped += 1
        continue

    # Build stereo: concatenate utterances sequentially
    # ch0 (left) = gpt/moshi, ch1 (right) = human/user
    total_samples = sum(len(s["audio"]) for s in segments)
    stereo = np.zeros((total_samples, 2), dtype=np.float32)

    alignments = []
    offset = 0

    for seg in segments:
        n = len(seg["audio"])
        ch = 0 if seg["from"] == "gpt" else 1
        stereo[offset:offset + n, ch] = seg["audio"]

        # Build word-level alignments for moshi/gpt channel only
        if seg["from"] == "gpt":
            start_sec = offset / TARGET_SR
            end_sec = (offset + n) / TARGET_SR
            duration = end_sec - start_sec

            words = seg["text"].split()
            if words:
                word_dur = duration / len(words)
                for i, word in enumerate(words):
                    w_start = round(start_sec + i * word_dur, 2)
                    w_end = round(start_sec + (i + 1) * word_dur, 2)
                    alignments.append([word, [w_start, w_end], "SPEAKER_MAIN"])

        offset += n

    # Save stereo WAV
    out_wav = OUTPUT_DIR / f"{idx}.wav"
    sf.write(str(out_wav), stereo, TARGET_SR)

    # Save alignment JSON
    out_json = OUTPUT_DIR / f"{idx}.json"
    with open(out_json, "w") as f:
        json.dump({"alignments": alignments}, f)

    duration = total_samples / TARGET_SR
    manifest.append({
        "path": f"{idx}.wav",
        "duration": round(duration, 4),
    })

    idx += 1

# === 3. Write manifest JSONL ===
jsonl_path = OUTPUT_DIR / "multidialog.jsonl"
with open(jsonl_path, "w") as f:
    for entry in manifest:
        json.dump(entry, f)
        f.write("\n")

print(f"\nDone!")
print(f"  Converted: {len(manifest)} conversations")
print(f"  Skipped:   {skipped}")
print(f"  Output:    {OUTPUT_DIR}")
print(f"  Manifest:  {jsonl_path}")
print(f"\nNote: Alignments are approximate (evenly spaced).")
print(f"For precise alignments, run:  python annotate.py {jsonl_path}")