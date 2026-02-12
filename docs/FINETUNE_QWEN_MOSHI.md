# Fine-tuning the Qwen-backed Moshi Pipeline

Kyutai provides an official fine-tuning repo for **standard Moshi** (7B / Moshiko):

- **[kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)** – LoRA (or full) fine-tuning with stereo wav + transcript data, FSDP, checkpointing, wandb.

Use that repo when fine-tuning the **original Moshi/Moshiko** models. It expects:

- **Data**: stereo wav (left = Moshi, right = user) + a `.jsonl` manifest and per-file `.json` transcripts with alignments.
- **Model**: loaded via `CheckpointInfo.from_hf_repo(...)` and the repo’s **SentencePiece** text tokenizer (32k vocab).

---

## Why the Qwen-backed model is different

The **Qwen-backed** Moshi in this fork uses:

- **Qwen 2.5 tokenizer** (vocab size 151936), not SentencePiece.
- **Different text special tokens** and padding IDs.

So moshi-finetune’s data pipeline (which uses SentencePiece and the 7B config) does **not** apply as-is. You have two paths:

1. **Use this repo’s script** (`scripts/train_qwen_moshi.py`) and feed it **precomputed `codes`** built with Mimi + **Qwen** tokenizer (see below).
2. **Extend moshi-finetune** (e.g. in a fork): add a “Qwen backbone” mode that loads via `get_qwen_moshi_lm`, and swap the interleaver/tokenizer to use the Qwen tokenizer so the same stereo wav + transcript format produces codes with Qwen text tokens.

---

## What gets trained (Qwen-backed model)

| Component | Initialization | Typical strategy |
|-----------|----------------|-------------------|
| **Transformer backbone** (Qwen) | Pretrained Qwen 2.5-3B | Freeze, or train with small LR / LoRA |
| **Depformer** (6 layers) | Random | **Train** (main focus) |
| **Audio embeddings** (`emb`) | Random | **Train** |
| **Audio output projections** (`linears`) | Random | **Train** |
| **Text embedding + output** | From Qwen | Usually freeze or small LR |

---

## Data format for Qwen-backed training

Training uses **code sequences** of shape `[B, K, T]`:

- **B**: batch size  
- **K**: 17 for `moshi_qwen_3b` (1 text + 16 audio codebooks)  
- **T**: time steps (e.g. 100–500 per sample)

- `codes[:, 0, :]`: **text token ids** from the **Qwen** tokenizer (vocab 151936). Use the model’s padding and `zero_token_id` (-1) where needed.
- `codes[:, 1:17, :]`: **audio codebook tokens** from Mimi (card 2048). Use `zero_token_id` (-1) for non-target positions.

To produce `codes` from raw data (e.g. stereo wav + transcripts, similar to moshi-finetune):

1. **User audio** → Mimi encode → fill the appropriate codebook positions.
2. **User transcript** → **Qwen** tokenizer → fill text codebook for input.
3. **Moshi response text** → **Qwen** tokenizer → fill text codebook for target.
4. **Moshi response audio** → Mimi encode → fill the response audio codebooks.

Layout and masking must match the config’s `delays`, `n_q`, and `dep_q`.

---

## Create the Qwen weights file (one-time)

The file `qwen_moshi_format.safetensors` is **not in the repo**. Generate it by running the conversion script (downloads Qwen 2.5-3B from HuggingFace and converts to Moshi format):

```bash
python scripts/import_qwen_pytorch.py \
  --checkpoint Qwen/Qwen2.5-3B \
  --out qwen_moshi_format.safetensors
```

Use the path you pass to `--out` as `--qwen-weights` in the commands below.

---

## Quick start: synthetic data (sanity check)

To verify the training loop without real data:

```bash
python scripts/train_qwen_moshi.py \
  --qwen-weights qwen_moshi_format.safetensors \
  --config configs/moshi_qwen_3b.json \
  --steps 100 \
  --batch-size 2 \
  --seq-length 32 \
  --freeze-backbone
```

---

## Training on real data (this repo)

1. **Build a dataset** that outputs `codes` of shape `(K, T)` per sample (or batch), using **Mimi for audio** and the **Qwen tokenizer for text** (same as the model).
2. **Run training** (example):

```bash
python scripts/train_qwen_moshi.py \
  --qwen-weights qwen_moshi_format.safetensors \
  --config configs/moshi_qwen_3b.json \
  --data-dir /path/to/codes_dataset \
  --epochs 3 \
  --batch-size 4 \
  --lr 1e-4 \
  --freeze-backbone \
  --out-dir runs/qwen_moshi_ft
```

(When `--data-dir` is implemented, it would load precomputed codes; until then you can use `--synthetic` or plug your own data loader into the script.)

---

## After fine-tuning

1. Save the full `state_dict` (e.g. `checkpoint_final.safetensors`).
2. Load with `get_qwen_moshi_lm(..., qwen_weights=<path>, config_path=configs/moshi_qwen_3b.json)`.
3. Run the benchmark and listen to the output:  
   `python scripts/benchmark_qwen_moshi.py --steps 50` → check `qwen_moshi_output.wav`.
