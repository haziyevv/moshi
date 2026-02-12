# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert Qwen 2.5-3B HuggingFace weights to Moshi's internal LMModel safetensors format.

This follows the same pattern as ``import_helium_mlx.py`` but targets Qwen 2.5
and handles GQA (Grouped Query Attention) and tied word embeddings.

Usage::

    python scripts/import_qwen_pytorch.py --out qwen_moshi_format.safetensors
    python scripts/import_qwen_pytorch.py --checkpoint Qwen/Qwen2.5-3B --out qwen_moshi.safetensors

Notes:
    - Qwen 2.5-3B uses GQA with 16 query heads and 2 KV heads (kv_repeat=8).
      The in_proj weights are stored with the reduced KV dimensions, matching
      Moshi's ``kv_repeat`` support.
    - Qwen 2.5-3B uses ``tie_word_embeddings=True``, so ``lm_head.weight`` is
      the same as ``model.embed_tokens.weight``. Both ``text_emb.weight`` and
      ``text_linear.weight`` are set from the embedding tensor.
    - Qwen 2.5-3B has bias on Q/K/V attention projections. Since Moshi's
      ``StreamingMultiheadAttention`` uses ``bias=False``, attention biases are
      dropped during conversion with a warning. This is a lossy conversion for
      attention biases only.
    - Moshi's RMSNorm uses ``alpha`` parameter with shape ``[1, 1, dim]``,
      while Qwen uses ``weight`` with shape ``[dim]``. The conversion reshapes
      accordingly.
    - Moshi's attention uses ``in_projs.0.weight`` and ``out_projs.0.weight``
      naming (not ``in_proj.weight``/``out_proj.weight``).
"""

import argparse
import glob
import json
import warnings
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_CHECKPOINT = "Qwen/Qwen2.5-3B"


def _load_tensors_from_sharded(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load tensors from potentially sharded safetensors files."""
    safetensor_files = sorted(glob.glob(str(model_dir / "model*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    tensors: dict[str, torch.Tensor] = {}
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def _load_tensors_from_single(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load tensors from a single safetensors file."""
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        return {key: f.get_tensor(key) for key in f.keys()}


def _reshape_norm_weight(w: torch.Tensor) -> torch.Tensor:
    """Reshape a 1D norm weight [dim] to Moshi's RMSNorm alpha format [1, 1, dim]."""
    return w.unsqueeze(0).unsqueeze(0)


def import_model(
    tensors: dict[str, torch.Tensor],
    out_path: Path,
    silent: bool = False,
) -> None:
    """Convert Qwen HF weights to Moshi format and save.

    Args:
        tensors: Dictionary of source tensors from HuggingFace checkpoint.
        out_path: Output safetensors file path.
        silent: If True, suppress progress output.
    """
    model: dict[str, torch.Tensor] = {}

    # Embedding weights
    embed_weight = tensors["model.embed_tokens.weight"]
    model["text_emb.weight"] = embed_weight

    # Qwen 2.5-3B uses tied word embeddings: lm_head.weight == embed_tokens.weight
    if "lm_head.weight" in tensors:
        model["text_linear.weight"] = tensors["lm_head.weight"]
    else:
        if not silent:
            print("lm_head.weight not found (tied embeddings), using embed_tokens.weight")
        model["text_linear.weight"] = embed_weight.clone()

    # Final layer norm -- Moshi uses 'alpha' with shape [1, 1, dim]
    model["out_norm.alpha"] = _reshape_norm_weight(tensors["model.norm.weight"])

    # Discover number of layers
    n_layers = -1
    for key in tensors.keys():
        if key.startswith("model.layers."):
            layer_idx = int(key.split(".")[2])
            n_layers = max(layer_idx, n_layers)
    n_layers += 1
    if not silent:
        print(f"Found {n_layers} transformer layers")

    # Check for attention biases
    has_attn_bias = "model.layers.0.self_attn.q_proj.bias" in tensors
    if has_attn_bias and not silent:
        warnings.warn(
            "Qwen model has attention Q/K/V biases, but Moshi's "
            "StreamingMultiheadAttention uses bias=False. "
            "Attention biases will be DROPPED during conversion. "
            "This is a lossy conversion for the attention bias terms.",
            stacklevel=2,
        )

    for layer_idx in range(n_layers):
        dst_prefix = f"transformer.layers.{layer_idx}."
        src_prefix = f"model.layers.{layer_idx}."

        # Layer norms -- Moshi uses 'alpha' with shape [1, 1, dim]
        model[dst_prefix + "norm1.alpha"] = _reshape_norm_weight(
            tensors[src_prefix + "input_layernorm.weight"]
        )
        model[dst_prefix + "norm2.alpha"] = _reshape_norm_weight(
            tensors[src_prefix + "post_attention_layernorm.weight"]
        )

        # Attention output projection -- Moshi uses out_projs.0.weight
        model[dst_prefix + "self_attn.out_projs.0.weight"] = tensors[
            src_prefix + "self_attn.o_proj.weight"
        ]

        # MLP down projection
        model[dst_prefix + "gating.linear_out.weight"] = tensors[
            src_prefix + "mlp.down_proj.weight"
        ]

        # MLP gate + up projections -> concatenated gating.linear_in
        gate_proj = tensors[src_prefix + "mlp.gate_proj.weight"]
        up_proj = tensors[src_prefix + "mlp.up_proj.weight"]
        linear_in = torch.cat([gate_proj, up_proj], dim=0)
        model[dst_prefix + "gating.linear_in.weight"] = linear_in

        # Attention Q/K/V projections -> concatenated self_attn.in_projs.0.weight
        # Qwen 2.5-3B: Q=[2048, 2048], K=[256, 2048], V=[256, 2048] (GQA with 2 KV heads)
        # Moshi format with kv_repeat: in_projs.0 = cat([Q, K, V], dim=0) = [2560, 2048]
        q = tensors[src_prefix + "self_attn.q_proj.weight"]
        k = tensors[src_prefix + "self_attn.k_proj.weight"]
        v = tensors[src_prefix + "self_attn.v_proj.weight"]
        in_proj = torch.cat([q, k, v], dim=0)
        model[dst_prefix + "self_attn.in_projs.0.weight"] = in_proj

        if not silent and layer_idx == 0:
            print(f"  Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
            print(f"  in_projs.0 shape: {in_proj.shape}")
            print(f"  gate_proj shape: {gate_proj.shape}, up_proj shape: {up_proj.shape}")
            print(f"  linear_in shape: {linear_in.shape}")
            print(f"  linear_out shape: {model[dst_prefix + 'gating.linear_out.weight'].shape}")

    if not silent:
        total_params = sum(v.numel() for v in model.values())
        print(f"Converted {len(model)} tensors, {total_params / 1e9:.2f}B parameters")

    save_file(model, str(out_path))
    if not silent:
        print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen 2.5-3B HuggingFace weights to Moshi LMModel format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="HuggingFace repo id or local path to checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output safetensors file path",
    )
    parser.add_argument(
        "-s", "--silent",
        action="store_true",
        help="Only print the output checkpoint path",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)

    if out_path.exists():
        print(f"Output file {out_path} already exists, skipping conversion.")
        print(out_path)
        return

    # Load source tensors
    if ckpt_path.is_dir():
        if not args.silent:
            print(f"Loading from local directory: {ckpt_path}")
        tensors = _load_tensors_from_sharded(ckpt_path)
    elif ckpt_path.is_file():
        if not args.silent:
            print(f"Loading from local file: {ckpt_path}")
        tensors = _load_tensors_from_single(ckpt_path)
    else:
        if not args.silent:
            print(f"Downloading from HuggingFace: {args.checkpoint}")
        try:
            single_path = hf_hub_download(
                repo_id=args.checkpoint, filename="model.safetensors"
            )
            tensors = _load_tensors_from_single(Path(single_path))
        except Exception:
            if not args.silent:
                print("Single model.safetensors not found, downloading full snapshot...")
            model_dir = snapshot_download(
                repo_id=args.checkpoint,
                allow_patterns=["model*.safetensors", "config.json"],
            )
            tensors = _load_tensors_from_sharded(Path(model_dir))

    if not args.silent:
        print(f"Loaded {len(tensors)} tensors from source checkpoint")

    import_model(tensors, out_path, silent=args.silent)
    print(out_path)


if __name__ == "__main__":
    main()
