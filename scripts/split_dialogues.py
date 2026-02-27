#!/usr/bin/env python3
"""Split long dialogues into shorter chunks suitable for training.

Reads dialogue.json files produced by generate_dialogue_scripts.py and
splits each one into several sub-dialogues of --chunk-size turns (default 8).
Each chunk becomes its own dialogue directory with a new dialogue.json.

Pipeline overview:
  1. [generate_dialogue_scripts.py]  Generate dialogue text (long, 16-30+ turns)
  2. [This script]                   Split into short chunks (6-10 turns each)
  3. [synthesize_dialogues.py]       Synthesize each turn with CosyVoice
  4. [build_stereo_dataset.py]       Combine into stereo WAVs + manifest
  5. [preencode_dataset.py]          Encode to training tensors
  6. [train_qwen_moshi.py]           Train

Usage::

    python scripts/split_dialogues.py \\
        --dialogues-dir ./dialogues \\
        --out-dir ./dialogues-split \\
        --chunk-size 8

    # Larger chunks (~30-40s audio)
    python scripts/split_dialogues.py \\
        --dialogues-dir ./dialogues \\
        --out-dir ./dialogues-split \\
        --chunk-size 10 --min-chunk-size 6

Output structure::

    dialogues-split/
      0000/dialogue.json   # chunk 0 of original dialogue 0000
      0001/dialogue.json   # chunk 1 of original dialogue 0000
      0002/dialogue.json   # chunk 0 of original dialogue 0001
      ...
"""

import argparse
import json
from pathlib import Path


def split_dialogue(dialogue: dict, chunk_size: int, min_chunk_size: int) -> list[dict]:
    """Split a single dialogue into chunks of chunk_size turns.

    Always splits on even boundaries so each chunk starts with the "main"
    speaker. Drops the last chunk if it has fewer than min_chunk_size turns.
    """
    turns = dialogue["turns"]
    speakers = dialogue["speakers"]
    topic = dialogue["topic"]
    original_id = dialogue["dialogue_id"]

    chunks = []
    for start in range(0, len(turns), chunk_size):
        chunk_turns = turns[start : start + chunk_size]

        if len(chunk_turns) < min_chunk_size:
            break

        renumbered = []
        for i, turn in enumerate(chunk_turns):
            renumbered.append({
                "role": turn["role"],
                "speaker": turn["speaker"],
                "text": turn["text"],
                "turn_idx": i,
            })

        chunks.append({
            "dialogue_id": None,  # assigned by caller
            "topic": topic,
            "speakers": speakers,
            "turns": renumbered,
            "source_dialogue": original_id,
            "source_turn_range": [start, start + len(chunk_turns)],
        })

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Split long dialogues into shorter chunks for training."
    )
    parser.add_argument("--dialogues-dir", type=str, required=True,
                        help="Input directory with dialogue folders")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for split dialogue folders")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Number of turns per chunk (default 8, ~20-30s audio)")
    parser.add_argument("--min-chunk-size", type=int, default=4,
                        help="Drop trailing chunk if shorter than this (default 4)")
    parser.add_argument("--max-dialogues", type=int, default=0,
                        help="Max input dialogues to process (0 = all)")
    args = parser.parse_args()

    if args.min_chunk_size > args.chunk_size:
        parser.error("--min-chunk-size must be <= --chunk-size")

    dialogues_dir = Path(args.dialogues_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dialogue_dirs = sorted(
        [d for d in dialogues_dir.iterdir()
         if d.is_dir() and (d / "dialogue.json").exists()],
        key=lambda d: d.name,
    )

    if args.max_dialogues > 0:
        dialogue_dirs = dialogue_dirs[: args.max_dialogues]

    print(f"Found {len(dialogue_dirs)} input dialogues in {dialogues_dir}")
    print(f"Chunk size: {args.chunk_size} turns, min chunk: {args.min_chunk_size} turns")
    print(f"Output: {out_dir}")
    print()

    output_idx = 0
    n_input = 0
    n_dropped_chunks = 0
    total_turns_in = 0
    total_turns_out = 0

    for dialogue_dir in dialogue_dirs:
        with open(dialogue_dir / "dialogue.json") as f:
            dialogue = json.load(f)

        n_input += 1
        total_turns_in += len(dialogue["turns"])

        chunks = split_dialogue(dialogue, args.chunk_size, args.min_chunk_size)

        n_total_chunks = -(-len(dialogue["turns"]) // args.chunk_size)  # ceil div
        n_dropped_chunks += n_total_chunks - len(chunks)

        for chunk in chunks:
            chunk["dialogue_id"] = output_idx
            chunk_dir = out_dir / f"{output_idx:04d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            with open(chunk_dir / "dialogue.json", "w") as f:
                json.dump(chunk, f, indent=2, ensure_ascii=False)

            total_turns_out += len(chunk["turns"])
            output_idx += 1

        if n_input % 500 == 0:
            print(f"  Processed {n_input}/{len(dialogue_dirs)} -> {output_idx} chunks")

    avg_chunks = output_idx / n_input if n_input > 0 else 0
    print(f"\nDone!")
    print(f"  Input dialogues:  {n_input}")
    print(f"  Output chunks:    {output_idx} ({avg_chunks:.1f} chunks/dialogue)")
    print(f"  Turns in:         {total_turns_in}")
    print(f"  Turns out:        {total_turns_out} ({total_turns_in - total_turns_out} dropped in short trailing chunks)")
    print(f"  Dropped chunks:   {n_dropped_chunks} (too short)")
    print(f"\nNext step: synthesize audio with CosyVoice:")
    print(f"  python scripts/synthesize_dialogues.py --dialogues-dir {out_dir}")


if __name__ == "__main__":
    main()
