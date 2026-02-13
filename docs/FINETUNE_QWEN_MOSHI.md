# Fine-tuning the Qwen-backed Moshi Pipeline

Kyutai provides an official fine-tuning repo for **standard Moshi** (7B / Moshiko):

- **[kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)** -- LoRA (or full) fine-tuning with stereo wav + transcript data, FSDP, checkpointing, wandb.

Use that repo when fine-tuning the **original Moshi/Moshiko** models. It expects:

- **Data**: stereo wav (left = Moshi, right = user) + a `.jsonl` manifest and per-file `.json` transcripts with alignments.
- **Model**: loaded via `CheckpointInfo.from_hf_repo(...)` and the repo's **SentencePiece** text tokenizer (32k vocab).

---

## Why the Qwen-backed model is different

The **Qwen-backed** Moshi in this fork uses:

- **Qwen 2.5 tokenizer** (vocab size 151936), not SentencePiece.
- **Different text special tokens** and padding IDs.

So moshi-finetune's data pipeline (which uses SentencePiece and the 7B config) does **not** apply as-is. You have two paths:

1. **Use this repo's script** (`scripts/train_qwen_moshi.py`) and feed it **precomputed `codes`** built with Mimi + **Qwen** tokenizer (see below).
2. **Extend moshi-finetune** (e.g. in a fork): add a "Qwen backbone" mode that loads via `get_qwen_moshi_lm`, and swap the interleaver/tokenizer to use the Qwen tokenizer so the same stereo wav + transcript format produces codes with Qwen text tokens.

---

## What gets trained (Qwen-backed model)

| Component | Initialization | Phase 1 | Phase 2 |
|-----------|----------------|---------|---------|
| **Transformer backbone** (Qwen) | Pretrained Qwen 2.5-3B | Frozen | Train @ low LR (1e-5) |
| **Depformer** (6 layers) | Random | **Train @ 3e-4** | **Train @ 3e-4** |
| **Audio embeddings** (`emb`) | Random | **Train @ 3e-4** | **Train @ 3e-4** |
| **Audio output projections** (`linears`) | Random | **Train @ 3e-4** | **Train @ 3e-4** |
| **Text embedding + output** | From Qwen | Frozen | Train @ low LR |

**Why two phases?** The Qwen backbone was trained on text only -- it has zero understanding of audio codebook patterns. Phase 1 gives the Depformer a reasonable starting point. Phase 2 (critical!) teaches the backbone to understand the multimodal text+audio stream. Without Phase 2, the backbone gives the Depformer useless representations, and the generated audio will be noise/silence even if the training loss looks good.

---

## Data format for Qwen-backed training

Training uses **code sequences** of shape `[B, K, T]`:

- **B**: batch size  
- **K**: 17 for `moshi_qwen_3b` (1 text + 16 audio codebooks)  
- **T**: time steps (e.g. 100--500 per sample)

- `codes[:, 0, :]`: **text token ids** from the **Qwen** tokenizer (vocab 151936). Use the model's padding and `zero_token_id` (-1) where needed.
- `codes[:, 1:17, :]`: **audio codebook tokens** from Mimi (card 2048). Use `zero_token_id` (-1) for non-target positions.

To produce `codes` from raw data (e.g. stereo wav + transcripts, similar to moshi-finetune):

1. **User audio** -> Mimi encode -> fill the appropriate codebook positions.
2. **User transcript** -> **Qwen** tokenizer -> fill text codebook for input.
3. **Moshi response text** -> **Qwen** tokenizer -> fill text codebook for target.
4. **Moshi response audio** -> Mimi encode -> fill the response audio codebooks.

Layout and masking must match the config's `delays`, `n_q`, and `dep_q`.

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
  --steps 100 --batch-size 2 --seq-length 32 \
  --freeze-backbone --synthetic
```

---

## Training on real data

### Step 1: Download a dataset

**Ready-made dataset:** [kyutai/DailyTalkContiguous](https://huggingface.co/datasets/kyutai/DailyTalkContiguous) (~14 GB) -- stereo WAV (left = main speaker, right = other), word-level alignments.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "kyutai/DailyTalkContiguous",
    repo_type="dataset",
    local_dir="./daily-talk-contiguous",
)
```

### Step 2: Pre-encode the dataset

Use `scripts/preencode_dataset.py` to convert the raw dataset into `.pt` files of codes:

```bash
python scripts/preencode_dataset.py \
  --data-dir ./daily-talk-contiguous \
  --jsonl ./daily-talk-contiguous/dailytalk.jsonl \
  --out-dir ./encoded_codes \
  --duration-sec 30
```

This reads each stereo WAV + JSON alignment, encodes audio with Mimi and text with the Qwen tokenizer, and saves one `codes` tensor per sample (`[1, 17, T]`). Use `--max-samples N` for a quick test run.

### Step 3: Train with regularization (critical!)

Without regularization, the model memorizes training sequences but fails at autoregressive inference (exposure bias). The `--audio-noise-ratio` and `--label-smoothing` flags are essential.

**Phase 1: Warm up Depformer (backbone frozen):**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen_moshi.py \
  --qwen-weights /tmp/qwen_moshi_format.safetensors \
  --data-dir ./encoded_codes \
  --epochs 30 \
  --batch-size 8 \
  --seq-length 256 \
  --grad-accum 4 \
  --lr 3e-4 \
  --warmup-steps 100 \
  --freeze-backbone \
  --audio-noise-ratio 0.15 \
  --label-smoothing 0.1 \
  --out-dir runs/phase1
```

**Phase 2: Joint training with differential LR:**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen_moshi.py \
  --qwen-weights runs/phase1/checkpoint_final.safetensors \
  --data-dir ./encoded_codes \
  --epochs 50 \
  --batch-size 4 \
  --seq-length 256 \
  --grad-accum 8 \
  --lr 3e-4 \
  --backbone-lr 1e-5 \
  --warmup-steps 200 \
  --audio-noise-ratio 0.15 \
  --label-smoothing 0.1 \
  --out-dir runs/phase2
```

Key settings explained:

| Parameter | Value | Why |
|-----------|-------|-----|
| `--audio-noise-ratio 0.15` | 0.15 | Randomly replaces 15% of input audio codes during training. Prevents memorization and makes the model robust to its own errors during autoregressive inference. |
| `--label-smoothing 0.1` | 0.1 | Prevents the model from being overconfident in its predictions. |
| `--backbone-lr 1e-5` | 1e-5 | Phase 2 only: trains backbone at 30x lower LR than Depformer to preserve Qwen's language knowledge. |

### Step 4: Diagnose

Before running full training, use the diagnostic script to verify model behavior:

```bash
python scripts/diagnose_qwen_moshi.py \
  --qwen-weights runs/phase1/checkpoint_final.safetensors \
  --sample-pt ./encoded_codes/000000.pt \
  --input-wav ./daily-talk-contiguous/data_stereo/0.wav
```

This compares teacher-forced predictions (like training) vs autoregressive generation (like inference). A healthy model should show:
- Teacher-forced accuracy: 60-80% (not 98%+ which means memorization)
- Autoregressive tokens: diverse, not collapsing into repeating patterns

### Step 5: Test with real audio

Use `scripts/inference_qwen_moshi.py` to test with actual audio input (NOT `benchmark_qwen_moshi.py`, which feeds silence):

```bash
python scripts/inference_qwen_moshi.py \
  --qwen-weights runs/phase2/checkpoint_final.safetensors \
  --input-wav ./daily-talk-contiguous/data_stereo/0.wav \
  --out-wav generated_response.wav \
  --out-text generated_text.txt
```

For custom audio (mono WAV = user speaking):

```bash
python scripts/inference_qwen_moshi.py \
  --qwen-weights runs/phase2/checkpoint_final.safetensors \
  --input-wav my_question.wav \
  --out-wav model_answer.wav
```

---

## After fine-tuning

1. The script auto-saves `checkpoint_final.safetensors` and per-epoch checkpoints.
2. Load with `get_qwen_moshi_lm(..., qwen_weights=<path>, config_path=configs/moshi_qwen_3b.json)`.
3. Test with real audio input using `inference_qwen_moshi.py` (see Step 5 above).
