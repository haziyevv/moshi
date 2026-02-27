#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Convert Qwen 2.5 HuggingFace weights to Moshi's LMModel format.

This maps the Qwen backbone weights into the naming convention expected by
Moshi's ``LMModel``.  Only the temporal-transformer weights (backbone, text
embedding, text linear projection, output norm) are converted.  Depformer and
audio-codec layers are left for random initialisation and subsequent
fine-tuning.

Works with any Qwen 2.5 size (3B, 7B, 14B, etc.) — dimensions are detected
automatically from the weight shapes.

Usage::

    python scripts/import_qwen_pytorch.py \
        --checkpoint Qwen/Qwen2.5-7B \
        --out qwen_7b_moshi_format.safetensors

The ``--checkpoint`` can be a local directory or a HuggingFace model id.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def load_qwen_state(checkpoint: str) -> dict[str, torch.Tensor]:
    """Load Qwen weights from a HF checkpoint directory or model id."""
    ckpt_path = Path(checkpoint)

    if ckpt_path.is_dir():
        # Local directory — look for safetensors first, then bin
        from safetensors.torch import load_file
        st_files = sorted(ckpt_path.glob("*.safetensors"))
        if st_files:
            state: dict[str, torch.Tensor] = {}
            for f in st_files:
                state.update(load_file(str(f)))
            return state
        bin_files = sorted(ckpt_path.glob("pytorch_model*.bin"))
        if bin_files:
            state = {}
            for f in bin_files:
                state.update(torch.load(str(f), map_location="cpu", weights_only=True))
            return state
        raise FileNotFoundError(f"No model files found in {ckpt_path}")

    # Assume it is a HuggingFace model id — download via transformers
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    return dict(model.state_dict())


def convert(qwen_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map Qwen HF keys → Moshi LMModel keys.

    Weight layout correspondence
    ----------------------------
    Attention (GQA, kv_repeat=8):
      Qwen  q_proj [2048, 2048] + k_proj [256, 2048] + v_proj [256, 2048]
      →  Moshi  in_projs.0 [2560, 2048]  (cat along dim-0)

      Qwen  o_proj [2048, 2048]
      →  Moshi  out_projs.0 [2048, 2048]

    FFN (SwiGLU):
      Qwen  gate_proj [11008, 2048] + up_proj [11008, 2048]
      →  Moshi  gating.linear_in [22016, 2048]  (cat along dim-0)

      Qwen  down_proj [2048, 11008]
      →  Moshi  gating.linear_out [2048, 11008]

    Norms:
      Qwen  input_layernorm  →  Moshi  norm1
      Qwen  post_attention_layernorm  →  Moshi  norm2

    Global:
      Qwen  model.embed_tokens  →  Moshi  text_emb  (+ 1 row for special token)
      Qwen  model.norm           →  Moshi  out_norm
      Qwen  lm_head              →  Moshi  text_linear  (or copied from embed_tokens if tied)
    """
    out: dict[str, torch.Tensor] = {}

    # --- Determine number of layers ---
    layer_indices = set()
    for k in qwen_state:
        if k.startswith("model.layers."):
            idx = int(k.split(".")[2])
            layer_indices.add(idx)
    num_layers = max(layer_indices) + 1
    print(f"  Qwen layers detected: {num_layers}")

    # --- Per-layer conversion ---
    for i in range(num_layers):
        prefix_q = f"model.layers.{i}"
        prefix_m = f"transformer.layers.{i}"

        # Attention: pack Q + K + V into single in_proj
        q = qwen_state[f"{prefix_q}.self_attn.q_proj.weight"]
        k = qwen_state[f"{prefix_q}.self_attn.k_proj.weight"]
        v = qwen_state[f"{prefix_q}.self_attn.v_proj.weight"]
        out[f"{prefix_m}.self_attn.in_projs.0.weight"] = torch.cat([q, k, v], dim=0)

        out[f"{prefix_m}.self_attn.out_projs.0.weight"] = (
            qwen_state[f"{prefix_q}.self_attn.o_proj.weight"]
        )

        # FFN: pack gate + up into single linear_in
        gate = qwen_state[f"{prefix_q}.mlp.gate_proj.weight"]
        up = qwen_state[f"{prefix_q}.mlp.up_proj.weight"]
        out[f"{prefix_m}.gating.linear_in.weight"] = torch.cat([gate, up], dim=0)

        out[f"{prefix_m}.gating.linear_out.weight"] = (
            qwen_state[f"{prefix_q}.mlp.down_proj.weight"]
        )

        # Norms — Moshi's RMSNorm uses .alpha with shape [1, 1, dim],
        # while Qwen uses .weight with shape [dim].
        norm1_w = qwen_state[f"{prefix_q}.input_layernorm.weight"]
        norm2_w = qwen_state[f"{prefix_q}.post_attention_layernorm.weight"]
        out[f"{prefix_m}.norm1.alpha"] = norm1_w.view(1, 1, -1)
        out[f"{prefix_m}.norm2.alpha"] = norm2_w.view(1, 1, -1)

    # --- Global weights ---
    embed = qwen_state["model.embed_tokens.weight"]  # [151936, 2048]
    # Moshi expects text_card + 1 rows (extra row for the initial/special token)
    pad_row = torch.randn(1, embed.shape[1], dtype=embed.dtype) * 0.02
    out["text_emb.weight"] = torch.cat([embed, pad_row], dim=0)  # [151937, 2048]

    final_norm = qwen_state["model.norm.weight"]
    out["out_norm.alpha"] = final_norm.view(1, 1, -1)

    # lm_head — if tied, it won't exist as a separate key
    if "lm_head.weight" in qwen_state:
        out["text_linear.weight"] = qwen_state["lm_head.weight"]
    else:
        # Tied weights: reuse embed_tokens (without the extra pad row)
        out["text_linear.weight"] = embed.clone()
        print("  lm_head tied to embed_tokens — copying embedding as text_linear")

    # --- Summary ---
    total_params = sum(v.numel() for v in out.values())
    print(f"  Converted {len(out)} tensors, {total_params / 1e6:.1f}M parameters")
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen 2.5 to Moshi format")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="HuggingFace model id (e.g. Qwen/Qwen2.5-3B) or local directory",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output safetensors path",
    )
    args = parser.parse_args()

    print(f"Loading Qwen weights from: {args.checkpoint}")
    qwen_state = load_qwen_state(args.checkpoint)
    print(f"  Loaded {len(qwen_state)} tensors")

    print("Converting to Moshi format...")
    moshi_state = convert(qwen_state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(moshi_state, str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
