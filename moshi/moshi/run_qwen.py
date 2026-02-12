# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone text generation script using Qwen 2.5-3B via HuggingFace.

This is the PyTorch equivalent of ``moshi_mlx.run_helium``, using Qwen 2.5-3B
as a drop-in replacement for Helium.

Usage::

    python -m moshi.run_qwen --prompt "Once upon a time"
    python -m moshi.run_qwen --hf-repo Qwen/Qwen2.5-3B --nsteps 100 --device cuda
    python -m moshi.run_qwen --quantize-bits 4 --prompt "Hello world"
"""

import argparse
import sys

import torch

from .models.qwen_wrapper import QwenWrapper, DEFAULT_HF_REPO


def main():
    parser = argparse.ArgumentParser(
        description="Text generation with Qwen 2.5-3B (replacement for Helium)."
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help="HuggingFace model repository (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--quantize-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help="Load model with bitsandbytes quantization (4 or 8 bit)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50, 0 to disable)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top-p nucleus sampling (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print token ids alongside generated text",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Resolve dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load model
    print(f"Loading {args.hf_repo} on {device} ({args.dtype})...", file=sys.stderr)
    wrapper = QwenWrapper.from_pretrained(
        hf_repo=args.hf_repo,
        device=device,
        dtype=dtype,
        quantize_bits=args.quantize_bits,
    )
    print("Model loaded.", file=sys.stderr)

    # Generate
    use_sampling = not args.greedy
    if args.verbose:
        print(f"prompt: {args.prompt}")
    else:
        print(args.prompt, end="", flush=True)

    for token_id, text_piece in wrapper.generate_step_by_step(
        prompt=args.prompt,
        max_new_tokens=args.nsteps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_sampling=use_sampling,
    ):
        if args.verbose:
            print(f"  token={token_id:6d}  {repr(text_piece)}")
        else:
            print(text_piece, end="", flush=True)

    if not args.verbose:
        print()  # Final newline


if __name__ == "__main__":
    main()
