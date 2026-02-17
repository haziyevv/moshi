#!/usr/bin/env python3
"""Generate dialogue scripts using an LLM for Moshi training data creation.

Uses an OpenAI-compatible API (works with OpenAI, vLLM, TGI, Ollama, etc.)
to generate natural two-speaker dialogues. The output is a set of dialogue
directories, each containing a dialogue.json with turn-by-turn text.

Fires --concurrent requests in parallel for speed (~5-10x faster than serial).

Pipeline overview:
  1. [This script]                  Generate dialogue text
  2. [synthesize_dialogues.py]      Synthesize each turn with CosyVoice
  3. [build_stereo_dataset.py]      Combine into stereo WAVs + alignments + manifest
  4. [preencode_dataset.py]         Encode to training tensors
  5. [train_qwen_moshi.py]          Train

Usage::

    # Using Grok (xAI)
    python scripts/generate_dialogue_scripts.py \\
        --num-dialogues 500 \\
        --out-dir ./dialogues \\
        --base-url https://api.x.ai/v1 \\
        --api-key $XAI_API_KEY \\
        --model grok-3-mini-fast \\
        --concurrent 20

    # Using OpenAI API
    export OPENAI_API_KEY=your_key
    python scripts/generate_dialogue_scripts.py \\
        --num-dialogues 500 \\
        --out-dir ./dialogues \\
        --model gpt-4o-mini \\
        --concurrent 16

    # Using a local vLLM server
    python scripts/generate_dialogue_scripts.py \\
        --num-dialogues 500 \\
        --out-dir ./dialogues \\
        --base-url http://localhost:8000/v1 \\
        --api-key dummy \\
        --model meta-llama/Llama-3-8B-Instruct \\
        --concurrent 32

speakers.json format::

    [
        {"id": "speaker_00", "name": "Alice"},
        {"id": "speaker_01", "name": "Bob"},
        ...
    ]

Output structure::

    dialogues/
      0000/dialogue.json
      0001/dialogue.json
      ...
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# CosyVoice speaker pool. Override with --speakers-file.
# The "id" is the CosyVoice speaker ID, "name" is a friendly name for the LLM.
# ---------------------------------------------------------------------------
DEFAULT_SPEAKERS = [
    {"id": "Speaker_Samantha", "name": "Samantha"},
    {"id": "Speaker_Honey", "name": "Honey"},
    {"id": "Speaker_Autumn", "name": "Autumn"},
    {"id": "Speaker_Swiss", "name": "Swiss"},
    {"id": "Speaker_ThaddeaGraham", "name": "Thaddea"},
    {"id": "Speaker_EmmaMackey", "name": "Emma"},
    {"id": "Speaker_RakheeThakrar", "name": "Rakhee"},
    {"id": "Speaker_NataliePortman", "name": "Natalie"},
    {"id": "Speaker_AnthonyLexa", "name": "Anthony"},
    {"id": "Speaker_MimiKeene", "name": "Mimi"},
    {"id": "Speaker_PatriciaAllison", "name": "Patricia"},
    {"id": "Speaker_DoreeneBlackstock", "name": "Doreene"},
    {"id": "Speaker_AlexandraJames", "name": "Alexandra"},
    {"id": "Speaker_RachelMcAdams", "name": "Rachel"},
    {"id": "Speaker_ScarletJohanson", "name": "Scarlet"},
    {"id": "Speaker_JoanAllen", "name": "Joan"},
    {"id": "Speaker_ChinenyeEzeudu", "name": "Chinenye"},
    {"id": "Speaker_SimoneAshley", "name": "Simone"},
    {"id": "Speaker_TanyaReynolds", "name": "Tanya"},
    {"id": "Speaker_LisaMcGrillis", "name": "Lisa"},
    {"id": "Speaker_EvaGreen", "name": "Eva"},
    {"id": "Speaker_CateBlanchet", "name": "Cate"},
    {"id": "Speaker_JemimaKirke", "name": "Jemima"},
    {"id": "Speaker_GillianAnderson", "name": "Gillian"},
    {"id": "Speaker_DuaSaleh", "name": "Dua"},
    {"id": "Speaker_AbbieCornish", "name": "Abbie"},
    {"id": "Speaker_SharonDuncanBrewster", "name": "Sharon"},
    {"id": "Speaker_AimeeLouWood", "name": "Aimee"},
    {"id": "Speaker_AnneMarieDuff", "name": "Anne"},
    {"id": "Speaker_HannahGadsby", "name": "Hannah"},
    {"id": "Speaker_SamanthaSpiro", "name": "SamanthaS"},
    {"id": "Speaker_Despina", "name": "Despina"},
    {"id": "Speaker_Aoede", "name": "Aoede"},
    {"id": "Speaker_Autonoe", "name": "Autonoe"},
    {"id": "Speaker_Achernar", "name": "Achernar"},
    {"id": "Speaker_Callirhoe", "name": "Callirhoe"},
    {"id": "Speaker_Kore", "name": "Kore"},
    {"id": "Speaker_Pulcherrima", "name": "Pulcherrima"},
    {"id": "Speaker_Vindemiatrix", "name": "Vindemiatrix"},
    {"id": "Speaker_Leda", "name": "Leda"},
    {"id": "Speaker_Laomodeia", "name": "Laomodeia"},
    {"id": "Speaker_Sulafat", "name": "Sulafat"},
    {"id": "Speaker_Erinome", "name": "Erinome"},
    {"id": "Speaker_Zephyr", "name": "Zephyr"},
    {"id": "Speaker_Phoenix", "name": "Phoenix"},
]

# ---------------------------------------------------------------------------
# Diverse conversation topics / scenarios
# ---------------------------------------------------------------------------
TOPICS = [
    "ordering coffee and pastries at a busy café",
    "asking a stranger for directions to the nearest train station",
    "discussing weekend plans with a friend",
    "a job interview for a software engineering position",
    "a patient describing symptoms to a doctor",
    "planning a road trip across the country",
    "debating which movie to watch tonight",
    "calling tech support about a broken laptop",
    "checking into a hotel after a long flight",
    "shopping for groceries and debating what to cook for dinner",
    "talking about the weather and whether to cancel an outdoor event",
    "recommending books to each other",
    "planning a surprise birthday party for a mutual friend",
    "discussing car trouble at an auto repair shop",
    "inquiring about renting an apartment",
    "booking a flight and comparing options",
    "asking about gym membership packages",
    "discussing how to take care of a new puppy",
    "sharing a family recipe and cooking tips",
    "scheduling a work meeting and resolving a calendar conflict",
    "returning a defective product at a store",
    "discussing a recent news event",
    "a teacher and student discussing an assignment",
    "two coworkers brainstorming ideas for a project",
    "a customer ordering food at a fast food restaurant",
    "discussing travel experiences and favorite destinations",
    "neighbors meeting for the first time and introducing themselves",
    "asking a librarian for help finding a book",
    "discussing investment and savings strategies",
    "planning a home renovation project",
    "a parent and teenager negotiating curfew rules",
    "two friends discussing a concert they just attended",
    "ordering a custom birthday cake at a bakery",
    "discussing the pros and cons of remote work",
    "a real estate agent showing a house to a buyer",
    "two hikers discussing trail conditions and gear",
    "a barista training a new employee",
    "discussing pet adoption at an animal shelter",
    "planning a garden and choosing what to plant",
    "comparing smartphones and deciding which one to buy",
    "discussing a TV show they both watch",
    "coordinating volunteer work for a local charity event",
    "a mechanic explaining repairs to a car owner",
    "two students studying together for an exam",
    "discussing morning routines and productivity tips",
    "a tourist asking a local for restaurant recommendations",
    "negotiating a deal at a flea market",
    "discussing which streaming service to subscribe to",
    "planning a wedding and choosing a venue",
    "two friends catching up after not seeing each other for years",
]

SYSTEM_PROMPT = """\
You are a dialogue writer creating natural, realistic conversations for \
speech synthesis training data. Your dialogues should sound like real \
spoken conversations — use contractions, natural pacing, occasional \
filler words (like "um", "well", "you know"), and realistic sentence lengths.

Rules:
- Each turn should be 1-2 sentences — keep them SHORT (under 15 words each \
  is ideal). The full dialogue should fit within about 30 seconds of speech.
- Keep the conversation natural and on-topic.
- Both speakers should contribute meaningfully.
- Avoid overly formal or literary language — this is spoken dialogue.
- Do NOT include stage directions, emotions in brackets, or narration.
- Output ONLY a valid JSON array. No markdown, no explanation."""


def build_user_prompt(topic: str, speaker_a: str, speaker_b: str, num_turns: int) -> str:
    return (
        f"Generate a natural conversation between {speaker_a} and {speaker_b} "
        f"about: {topic}\n\n"
        f"The conversation should have exactly {num_turns} turns, alternating "
        f"between speakers. {speaker_a} speaks first.\n\n"
        f"Output a JSON array of objects with 'speaker' and 'text' fields:\n"
        f'[{{"speaker": "{speaker_a}", "text": "..."}}, '
        f'{{"speaker": "{speaker_b}", "text": "..."}}, ...]'
    )


def call_llm(client, model: str, topic: str, speaker_a: str, speaker_b: str,
             num_turns: int, max_retries: int = 3) -> list[dict] | None:
    """Call the LLM and parse the response into a list of turns."""
    prompt = build_user_prompt(topic, speaker_a, speaker_b, num_turns)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines)

            turns = json.loads(content)

            if not isinstance(turns, list) or len(turns) == 0:
                raise ValueError("Expected a non-empty list")

            for t in turns:
                if "speaker" not in t or "text" not in t:
                    raise ValueError(f"Missing 'speaker' or 'text' in turn: {t}")
                if not isinstance(t["text"], str) or len(t["text"].strip()) == 0:
                    raise ValueError(f"Empty text in turn: {t}")

            return turns

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


def generate_one(client, model, dialogue_idx, out_dir, speakers, topics,
                 min_turns, max_turns, skip_existing):
    """Generate a single dialogue. Called from thread pool."""
    dialogue_dir = out_dir / f"{dialogue_idx:04d}"
    dialogue_path = dialogue_dir / "dialogue.json"

    if skip_existing and dialogue_path.exists():
        return "skipped", dialogue_idx

    dialogue_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(dialogue_idx * 7919 + 42)
    spk_main, spk_user = rng.sample(speakers, 2)
    topic = rng.choice(topics)
    num_turns = rng.randrange(min_turns, max_turns + 1, 2)

    turns = call_llm(client, model, topic, spk_main["name"], spk_user["name"], num_turns)

    if turns is None:
        return "failed", dialogue_idx

    normalized_turns = []
    for idx, turn in enumerate(turns):
        if turn["speaker"] == spk_main["name"]:
            role = "main"
        elif turn["speaker"] == spk_user["name"]:
            role = "user"
        else:
            role = "main" if idx % 2 == 0 else "user"

        normalized_turns.append({
            "role": role,
            "speaker": turn["speaker"],
            "text": turn["text"].strip(),
            "turn_idx": idx,
        })

    dialogue_data = {
        "dialogue_id": dialogue_idx,
        "topic": topic,
        "speakers": {
            "main": spk_main,
            "user": spk_user,
        },
        "turns": normalized_turns,
    }

    with open(dialogue_path, "w") as f:
        json.dump(dialogue_data, f, indent=2, ensure_ascii=False)

    return "ok", dialogue_idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate dialogue scripts with an LLM for Moshi training data."
    )
    parser.add_argument("--num-dialogues", type=int, default=500,
                        help="Number of dialogues to generate")
    parser.add_argument("--out-dir", type=str, default="./dialogues",
                        help="Output directory for dialogue folders")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model name")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Custom API base URL (for local LLM servers)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (falls back to OPENAI_API_KEY env var)")
    parser.add_argument("--speakers-file", type=str, default=None,
                        help="JSON file with speaker list [{id, name}, ...]")
    parser.add_argument("--min-turns", type=int, default=4,
                        help="Minimum turns per dialogue (keep low for short dialogues)")
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Maximum turns per dialogue (8 turns ≈ 20-30s of audio)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip dialogues that already have a dialogue.json")
    parser.add_argument("--concurrent", type=int, default=16,
                        help="Number of concurrent API requests (default 16)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load speakers
    if args.speakers_file:
        with open(args.speakers_file) as f:
            speakers = json.load(f)
        print(f"Loaded {len(speakers)} speakers from {args.speakers_file}")
    else:
        speakers = DEFAULT_SPEAKERS
        print(f"Using {len(speakers)} default speakers")

    if len(speakers) < 2:
        raise ValueError("Need at least 2 speakers")

    # Init LLM client
    from openai import OpenAI
    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    elif os.environ.get("OPENAI_API_KEY"):
        client_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY env var or pass --api-key."
        )

    client = OpenAI(**client_kwargs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.num_dialogues} dialogues...")
    print(f"  Model:      {args.model}")
    print(f"  Output:     {out_dir}")
    print(f"  Turns:      {args.min_turns}-{args.max_turns}")
    print(f"  Concurrent: {args.concurrent}")
    print()

    n_generated = 0
    n_skipped = 0
    n_failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futures = {
            pool.submit(
                generate_one, client, args.model, i, out_dir, speakers, TOPICS,
                args.min_turns, args.max_turns, args.skip_existing,
            ): i
            for i in range(args.num_dialogues)
        }

        for future in as_completed(futures):
            status, idx = future.result()
            if status == "ok":
                n_generated += 1
            elif status == "skipped":
                n_skipped += 1
            else:
                n_failed += 1

            done = n_generated + n_skipped + n_failed
            if done % 20 == 0:
                elapsed = time.time() - t0
                rate = n_generated / elapsed if elapsed > 0 else 0
                print(f"  {done}/{args.num_dialogues}  "
                      f"generated={n_generated}  failed={n_failed}  "
                      f"skipped={n_skipped}  "
                      f"({rate:.1f} dialogues/sec)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s!")
    print(f"  Generated: {n_generated}")
    print(f"  Failed:    {n_failed}")
    print(f"  Skipped:   {n_skipped}")
    if n_generated > 0:
        print(f"  Speed:     {n_generated / elapsed:.1f} dialogues/sec")
    print(f"\nNext step: synthesize audio with CosyVoice:")
    print(f"  python scripts/synthesize_dialogues.py --dialogues-dir {out_dir}")


if __name__ == "__main__":
    main()
