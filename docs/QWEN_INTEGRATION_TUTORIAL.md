# Integrating Qwen 2.5 into the Moshi Pipeline: An In-Depth Tutorial

This tutorial explains the complete process of replacing Moshi's original 7B backbone with Qwen 2.5-3B. It covers architecture understanding, weight conversion, configuration, training, and debugging.

## Table of Contents

1. [Understanding Moshi's Architecture](#1-understanding-moshis-architecture)
2. [Why Replace the Backbone?](#2-why-replace-the-backbone)
3. [Architecture Mapping: Moshi ↔ Qwen](#3-architecture-mapping-moshi--qwen)
4. [Weight Conversion Process](#4-weight-conversion-process)
5. [Configuration Changes](#5-configuration-changes)
6. [The Loading Process](#6-the-loading-process)
7. [Data Preparation](#7-data-preparation)
8. [Training Strategy](#8-training-strategy)
9. [Common Issues and Debugging](#9-common-issues-and-debugging)
10. [Key Lessons Learned](#10-key-lessons-learned)

---

## 1. Understanding Moshi's Architecture

Moshi is a full-duplex speech dialogue model with three main components:

### 1.1 Mimi (Neural Audio Codec)
- Encodes 24kHz audio into discrete tokens at 12.5 Hz
- Uses 8 codebooks, each with vocabulary size 2048
- Processes audio bidirectionally: encode (audio → tokens) and decode (tokens → audio)

### 1.2 Temporal Transformer (Backbone)
- Large transformer (originally 7B parameters)
- Processes the combined text + audio token streams
- Outputs representations used by the Depformer
- **This is what we replace with Qwen**

### 1.3 Depformer (Depth Transformer)
- Small 6-layer transformer
- Models inter-codebook dependencies at each timestep
- Predicts audio tokens autoregressively within a frame

### 1.4 Data Flow

```
                    ┌─────────────────────────────────────────┐
                    │           Input Codes [B, 17, T]        │
                    │  Channel 0: Text tokens                 │
                    │  Channels 1-8: Main speaker audio       │
                    │  Channels 9-16: User audio              │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │         Embedding Layers                │
                    │  text_emb: [vocab+1, dim] → text embed  │
                    │  emb[0-15]: [2049, dim] → audio embeds  │
                    └─────────────────┬───────────────────────┘
                                      │
                                      │ SUM all embeddings
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │      Temporal Transformer (Backbone)    │
                    │  - 36 layers (for Qwen 2.5-3B)          │
                    │  - Processes combined representation    │
                    │  - Outputs: transformer_out [B, T, dim] │
                    └─────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────┴────────────────────────┐
              │                                                │
    ┌─────────▼─────────┐                        ┌─────────────▼─────────────┐
    │   Text Linear     │                        │   Depformer (6 layers)    │
    │ [dim, text_vocab] │                        │  For each of 8 codebooks: │
    │                   │                        │  - Project transformer_out│
    │  → text_logits    │                        │  - Add previous token emb │
    └───────────────────┘                        │  - Run through Depformer  │
                                                 │  - Linear → audio_logits  │
                                                 └───────────────────────────┘
```

### 1.5 The Delay Pattern

Moshi uses a delay pattern to handle the temporal dependencies:

```python
delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
#        │  │  └─────────────────┘  │  └─────────────────┘
#        │  │   main audio cb1-7    │   user audio cb1-7
#        │  │   (delay=1)           │   (delay=1)
#        │  │                       │
#        │  main audio cb0          user audio cb0
#        │  (delay=0)               (delay=0)
#        │
#        text (delay=0)
```

This means:
- At time T, the model sees text[T], main_audio_cb0[T], user_audio_cb0[T]
- But for other codebooks, it sees the previous timestep: main_audio_cb1-7[T-1], user_audio_cb1-7[T-1]

---

## 2. Why Replace the Backbone?

### 2.1 Original Moshi Backbone
- 7B parameters, trained from scratch on audio+text
- Uses SentencePiece tokenizer (32k vocab)
- Deeply integrated with Moshi's training

### 2.2 Benefits of Using Qwen
- Leverage pretrained language understanding
- Smaller model (3B vs 7B) = faster training/inference
- Strong multilingual and reasoning capabilities
- Active community and updates

### 2.3 Challenges
- Qwen was trained on **text only** - no audio understanding
- Different tokenizer (151k+ vocab vs 32k)
- Different special tokens and padding conventions
- Need to teach the backbone about audio embeddings

---

## 3. Architecture Mapping: Moshi ↔ Qwen

### 3.1 Layer Correspondence

| Moshi Component | Qwen Component |
|-----------------|----------------|
| `transformer.layers.{i}.norm1` | `model.layers.{i}.input_layernorm` |
| `transformer.layers.{i}.norm2` | `model.layers.{i}.post_attention_layernorm` |
| `transformer.layers.{i}.self_attn.in_projs.0` | `model.layers.{i}.self_attn.{q,k,v}_proj` (concatenated) |
| `transformer.layers.{i}.self_attn.out_projs.0` | `model.layers.{i}.self_attn.o_proj` |
| `transformer.layers.{i}.gating.linear_in` | `model.layers.{i}.mlp.{gate,up}_proj` (concatenated) |
| `transformer.layers.{i}.gating.linear_out` | `model.layers.{i}.mlp.down_proj` |
| `out_norm` | `model.norm` |
| `text_emb` | `model.embed_tokens` |
| `text_linear` | `lm_head` (tied with embed_tokens in Qwen) |

### 3.2 Attention: Grouped Query Attention (GQA)

Qwen 2.5-3B uses GQA:
- 16 query heads
- 2 key-value heads
- `kv_repeat = 16 / 2 = 8`

This means K and V projections are smaller:
- Q: [2048, 2048]
- K: [256, 2048]  (2048 / 8)
- V: [256, 2048]

Concatenated in_proj: [2048 + 256 + 256, 2048] = [2560, 2048]

### 3.3 FFN: SwiGLU Gating

Both Moshi and Qwen use SwiGLU (SiLU activation with gating):

```python
# Qwen's FFN
gate = silu(x @ gate_proj)
up = x @ up_proj
out = (gate * up) @ down_proj

# Moshi combines gate_proj and up_proj into linear_in
linear_in = concat([gate_proj, up_proj], dim=0)  # [2 * intermediate, dim]
```

**Critical**: Moshi's gating module has a parameter reduction formula:
```python
if dim_feedforward == 4 * dim:
    hidden = (21 * dim) // 8
else:
    hidden = (2 * dim_feedforward) // 3  # <-- This is used
```

For Qwen's intermediate_size of 11008:
- We need `hidden = 11008`
- So: `dim_feedforward = (11008 * 3) // 2 = 16512`
- Therefore: `hidden_scale = 16512 / 2048 = 8.0625`

### 3.4 Normalization

Qwen uses RMSNorm with shape `[dim]`, but Moshi uses `[1, 1, dim]`:
```python
def reshape_norm(w):
    return w.unsqueeze(0).unsqueeze(0)  # [dim] → [1, 1, dim]
```

---

## 4. Weight Conversion Process

The script `scripts/import_qwen_pytorch.py` converts Qwen weights to Moshi format.

### 4.1 Key Conversions

```python
# Embeddings
model["text_emb.weight"] = tensors["model.embed_tokens.weight"]
model["text_linear.weight"] = tensors["lm_head.weight"]  # or embed_tokens if tied

# Final norm
model["out_norm.alpha"] = reshape_norm(tensors["model.norm.weight"])

# Per-layer conversions
for layer_idx in range(n_layers):
    # Norms
    model[f"transformer.layers.{i}.norm1.alpha"] = reshape_norm(input_layernorm)
    model[f"transformer.layers.{i}.norm2.alpha"] = reshape_norm(post_attention_layernorm)

    # Attention
    q, k, v = q_proj, k_proj, v_proj
    model[f"transformer.layers.{i}.self_attn.in_projs.0.weight"] = concat([q, k, v], dim=0)
    model[f"transformer.layers.{i}.self_attn.out_projs.0.weight"] = o_proj

    # FFN
    model[f"transformer.layers.{i}.gating.linear_in.weight"] = concat([gate_proj, up_proj], dim=0)
    model[f"transformer.layers.{i}.gating.linear_out.weight"] = down_proj
```

### 4.2 What's NOT Converted

- **Attention biases**: Qwen has Q/K/V biases, but Moshi doesn't support them. They're dropped with a warning.
- **Depformer weights**: These don't exist in Qwen and are randomly initialized.
- **Audio embeddings**: Randomly initialized (learned during fine-tuning).

### 4.3 Running the Conversion

```bash
python scripts/import_qwen_pytorch.py \
  --checkpoint Qwen/Qwen2.5-3B \
  --out /tmp/qwen_moshi_format.safetensors
```

---

## 5. Configuration Changes

### 5.1 The Config File: `configs/moshi_qwen_3b.json`

```json
{
    "dim": 2048,                    // Qwen's hidden_size
    "text_card": 151665,            // Qwen's full vocab (base + special tokens)
    "existing_text_padding_id": 151643,   // <|endoftext|>
    "existing_text_end_padding_id": 151645, // <|im_end|>
    "n_q": 16,                      // Total audio codebooks (8 main + 8 user)
    "dep_q": 8,                     // Codebooks predicted by Depformer
    "card": 2048,                   // Mimi audio vocabulary
    "num_heads": 16,                // Qwen's attention heads
    "num_layers": 36,               // Qwen's layer count
    "hidden_scale": 8.0625,         // See Section 3.3 for calculation
    "kv_repeat": 8,                 // GQA: 16 heads / 2 kv_heads
    "max_period": 1000000,          // Qwen's RoPE theta
    "gating": "silu",               // SwiGLU activation
    "norm": "rms_norm_f32",         // RMSNorm in float32

    // Depformer settings (unchanged from original Moshi)
    "depformer_dim": 1024,
    "depformer_num_layers": 6,
    ...
}
```

### 5.2 Critical Settings Explained

#### `text_card` and Special Tokens
```
Qwen vocab breakdown:
- Base vocab: 151643 tokens (IDs 0-151642)
- Special tokens: IDs 151643-151664
  - 151643: <|endoftext|> (we use for padding)
  - 151644: <|im_start|>
  - 151645: <|im_end|> (we use for end-of-padding)
  - ...

text_card must be >= highest token ID used (151665 to be safe)
```

**Why this matters**: Original Moshi used SentencePiece tokens 0-3 for special purposes. But in Qwen, tokens 0-3 are regular characters ('!', '"', '#', '$'). Using them as padding confuses the pretrained backbone.

#### `hidden_scale`
```
Qwen intermediate_size = 11008
Moshi formula: hidden = (2 * dim_feedforward) // 3

To get hidden = 11008:
  dim_feedforward = (11008 * 3) // 2 = 16512
  hidden_scale = 16512 / 2048 = 8.0625
```

**Why this matters**: Wrong hidden_scale = FFN weight shapes don't match = weights don't load.

---

## 6. The Loading Process

### 6.1 `get_qwen_moshi_lm()` Function

```python
def get_qwen_moshi_lm(qwen_weights, config_path, device, dtype):
    # 1. Load config
    with open(config_path) as f:
        lm_kwargs = json.load(f)

    # 2. Create model with random weights
    model = LMModel(device=device, dtype=dtype, **lm_kwargs)

    # 3. Load converted Qwen weights
    qwen_state = load_file(qwen_weights)

    # 4. Handle embedding size mismatch
    # Qwen might have more/fewer tokens than config specifies
    if qwen_emb.shape[0] < model_emb.shape[0]:
        # Pad with random vectors
        qwen_state["text_emb.weight"] = pad(qwen_emb)
    elif qwen_emb.shape[0] > model_emb.shape[0]:
        # Truncate
        qwen_state["text_emb.weight"] = qwen_emb[:model_size]

    # 5. Load weights (strict=False allows missing Depformer/audio keys)
    model.load_state_dict(qwen_state, strict=False, assign=True)

    return model
```

### 6.2 What Gets Loaded vs Randomly Initialized

**Loaded from Qwen (~219 tensors):**
- `transformer.layers.*.norm1.alpha`
- `transformer.layers.*.norm2.alpha`
- `transformer.layers.*.self_attn.in_projs.0.weight`
- `transformer.layers.*.self_attn.out_projs.0.weight`
- `transformer.layers.*.gating.linear_in.weight`
- `transformer.layers.*.gating.linear_out.weight`
- `out_norm.alpha`
- `text_emb.weight`
- `text_linear.weight`

**Randomly initialized (~244 tensors):**
- `emb[0-15].weight` (audio embeddings)
- `depformer.*` (entire Depformer)
- `depformer_emb[0-7].weight`
- `depformer_text_emb.weight`
- `depformer_in[0-7].weight`
- `linears[0-7].weight` (audio output projections)

---

## 7. Data Preparation

### 7.1 Data Format

Training data: `[B, 17, T]` tensor of token IDs

```
Channel 0:     Text tokens (Qwen tokenizer)
Channels 1-8:  Main speaker audio (8 Mimi codebooks)
Channels 9-16: User speaker audio (8 Mimi codebooks)
```

### 7.2 The Pre-encoding Script

`scripts/preencode_dataset.py` converts raw data to training format:

```python
# 1. Load stereo audio (left=main, right=user)
wav_main, wav_user = load_stereo_audio(path)

# 2. Encode both channels with Mimi
main_codes = mimi.encode(wav_main)  # [1, 8, T]
user_codes = mimi.encode(wav_user)  # [1, 8, T]

# 3. Build text stream from alignments
text_tokens = build_text_stream(alignments, tokenizer)  # [1, 1, T]

# 4. Concatenate
codes = torch.cat([text_tokens, main_codes, user_codes], dim=1)  # [1, 17, T]

# 5. Save
torch.save({"codes": codes}, output_path)
```

### 7.3 Text Stream Encoding

```python
def build_text_stream(alignments, tokenizer):
    # alignments: [(word, (start_time, end_time), speaker), ...]

    text_tokens = [padding_id] * T  # Fill with padding

    for word, (start, end), speaker in alignments:
        if speaker != "SPEAKER_MAIN":
            continue

        tokens = tokenizer.encode(word)
        frame_idx = int(start * frame_rate)

        for i, tok in enumerate(tokens):
            if frame_idx + i < T:
                text_tokens[frame_idx + i] = tok

    return torch.tensor(text_tokens)
```

### 7.4 Special Token IDs

```python
# Must match config!
text_padding_id = 151643       # <|endoftext|>
end_of_text_padding_id = 151645  # <|im_end|>
zero_token_id = -1             # "Don't predict/input"
```

---

## 8. Training Strategy

### 8.1 Why Two Phases?

The Qwen backbone was trained on **text only**. It has no understanding of audio embeddings. If we train everything together from the start:
- The backbone produces garbage representations
- The Depformer can't learn meaningful patterns
- Training diverges or converges to trivial solutions

### 8.2 Phase 1: Warm Up Depformer (Backbone Frozen)

```bash
python scripts/train_qwen_moshi.py \
  --qwen-weights /tmp/qwen_moshi_format.safetensors \
  --freeze-backbone \
  --epochs 30 \
  --lr 3e-4 \
  --audio-noise-ratio 0.15
```

**What happens:**
- Backbone weights are frozen
- Only Depformer, audio embeddings, and audio projections are trained
- The Depformer learns to predict audio tokens given (frozen) backbone representations

**Why it's not enough:**
- The frozen backbone doesn't understand audio
- It produces similar representations regardless of audio input
- The Depformer learns patterns, but they're not grounded in actual audio understanding

### 8.3 Phase 2: Joint Training (Backbone Unfrozen)

```bash
python scripts/train_qwen_moshi.py \
  --qwen-weights runs/phase1/checkpoint_final.safetensors \
  --backbone-lr 1e-5 \
  --lr 3e-4 \
  --epochs 50 \
  --audio-noise-ratio 0.15
```

**What happens:**
- Backbone trains at low LR (1e-5) to preserve language knowledge
- Depformer trains at higher LR (3e-4)
- The backbone learns to produce useful representations for audio

**Key insight:**
- The backbone must learn that audio embeddings carry information
- Without Phase 2, the model fails at inference (autoregressive collapse)

### 8.4 Audio Noise Ratio

```python
def inject_noise(codes, model, audio_noise_ratio=0.15):
    # Randomly replace 15% of audio codes with random values
    mask = torch.rand_like(codes) < audio_noise_ratio
    random_codes = torch.randint(0, model.card, codes.shape)
    return torch.where(mask, random_codes, codes)
```

**Why it matters:**
- Training: model sees perfect ground-truth inputs
- Inference: model sees its own (potentially wrong) predictions
- Noise injection bridges this gap by simulating imperfect inputs during training

---

## 9. Common Issues and Debugging

### 9.1 Shape Mismatches

**Symptom:** `RuntimeError: size mismatch for transformer.layers.0.gating.linear_in.weight`

**Cause:** Wrong `hidden_scale` in config

**Fix:** Calculate correct hidden_scale:
```python
hidden_scale = (qwen_intermediate_size * 3 // 2) / dim
# For Qwen 2.5-3B: (11008 * 3 // 2) / 2048 = 8.0625
```

### 9.2 CUDA Out-of-Bounds Error

**Symptom:** `CUDA error: device-side assert triggered`

**Cause:** Token ID exceeds embedding table size

**Fix:** Ensure `text_card` covers all tokens used:
```python
# Check max token in data
max_token = codes[:, 0, :].max()
assert max_token < model.text_card
```

### 9.3 Autoregressive Collapse

**Symptom:** Model outputs repetitive tokens like `[948, 948, 948, ...]`

**Possible causes:**
1. **No Phase 2 training** - backbone doesn't understand audio
2. **No audio noise** - exposure bias
3. **Imbalanced data** - model predicts most common token

**Diagnosis:**
```bash
python scripts/diagnose_qwen_moshi.py \
  --qwen-weights checkpoint.safetensors \
  --sample-pt encoded_codes/000000.pt
```

Compare teacher-forced vs autoregressive predictions.

### 9.4 Token ID Conflicts

**Symptom:** Model generates strange text or audio

**Cause:** Using wrong special token IDs

**Fix:** Use Qwen's actual special tokens:
```python
# Wrong (SentencePiece conventions):
padding_id = 3  # In Qwen, this is '$'

# Correct (Qwen conventions):
padding_id = 151643  # <|endoftext|>
```

### 9.5 Mode Collapse to Common Tokens

**Symptom:** Model always predicts the same few tokens

**Cause:** Imbalanced training data (e.g., 25% silence)

**Diagnosis:**
```python
# Check token distribution
from collections import Counter
counter = Counter(all_audio_tokens)
print(counter.most_common(10))
```

**Fixes:**
- Class-weighted loss
- Focal loss
- Data resampling
- Filter out silence-heavy sequences

---

## 10. Key Lessons Learned

### 10.1 Architecture
- Moshi's gating formula requires specific `hidden_scale` calculation
- GQA (kv_repeat) must match Qwen's configuration
- Norm weight shapes need reshaping [dim] → [1, 1, dim]

### 10.2 Tokenization
- Don't reuse token IDs 0-3 for special purposes
- Use Qwen's native special tokens (151643+)
- Ensure `text_card` covers all tokens including specials

### 10.3 Training
- Phase 1 alone is insufficient
- Phase 2 backbone training is critical
- Audio noise ratio prevents exposure bias
- Watch for data imbalance (silence tokens)

### 10.4 Debugging
- Always compare teacher-forced vs autoregressive
- Check token frequency distributions
- Verify weight loading (count loaded vs missing keys)
- Decode audio tokens to understand what they represent

---

## Appendix: Quick Reference

### File Locations
- Config: `configs/moshi_qwen_3b.json`
- Weight converter: `scripts/import_qwen_pytorch.py`
- Model loader: `moshi/moshi/models/loaders.py` (`get_qwen_moshi_lm`)
- Training script: `scripts/train_qwen_moshi.py`
- Data encoder: `scripts/preencode_dataset.py`
- Diagnostic: `scripts/diagnose_qwen_moshi.py`

### Key Parameters
```
Qwen 2.5-3B:
  hidden_size: 2048
  intermediate_size: 11008
  num_hidden_layers: 36
  num_attention_heads: 16
  num_key_value_heads: 2
  vocab_size: 151643 (+ special tokens)

Moshi config:
  dim: 2048
  hidden_scale: 8.0625
  num_layers: 36
  num_heads: 16
  kv_repeat: 8
  text_card: 151665
```

### Training Commands
```bash
# Phase 1
python scripts/train_qwen_moshi.py \
  --qwen-weights /tmp/qwen_moshi_format.safetensors \
  --freeze-backbone --epochs 30 --lr 3e-4 \
  --audio-noise-ratio 0.15 --out-dir runs/phase1

# Phase 2
python scripts/train_qwen_moshi.py \
  --qwen-weights runs/phase1/checkpoint_final.safetensors \
  --backbone-lr 1e-5 --epochs 50 --lr 3e-4 \
  --audio-noise-ratio 0.15 --out-dir runs/phase2
```
