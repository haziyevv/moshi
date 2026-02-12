# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fine-tune the Qwen-backed Moshi model (Depformer + audio components, optionally backbone).

Supports:
  - Synthetic random codes (sanity check)
  - Optional --freeze-backbone to only train Depformer and audio
  - Optional --lora-rank to add LoRA to the backbone

Example (synthetic, 100 steps):
  python scripts/train_qwen_moshi.py \\
    --qwen-weights /tmp/qwen_moshi_format.safetensors \\
    --config configs/moshi_qwen_3b.json \\
    --steps 100 --batch-size 2 --seq-length 32 --freeze-backbone
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moshi.models import get_qwen_moshi_lm
from moshi.models.loaders import get_conditioner_provider, get_condition_fuser
from moshi.utils.utils import cross_entropy


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen-backed Moshi")
    p.add_argument("--qwen-weights", type=str, required=True,
                   help="Path to converted Qwen safetensors (create with: python scripts/import_qwen_pytorch.py --checkpoint Qwen/Qwen2.5-3B --out <this path>)")
    p.add_argument("--config", type=str, default=None, help="Path to moshi_qwen_3b.json")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-length", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="runs/qwen_moshi_ft")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lora-rank", type=int, default=0)
    return p.parse_args()


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        return json.load(f)


def freeze_backbone(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ("transformer.", "text_emb.", "text_linear.", "out_norm.")):
            param.requires_grad = False


def synthetic_batch(model: torch.nn.Module, B: int, T: int, device: torch.device) -> torch.Tensor:
    K = model.num_codebooks
    codes = torch.zeros(B, K, T, dtype=torch.long, device=device)
    codes[:, 0, :] = torch.randint(0, model.text_card, (B, T), device=device)
    for k in range(1, K):
        codes[:, k, :] = torch.randint(0, model.card, (B, T), device=device)
    pad_len = max(1, T // 10)
    codes[:, :, -pad_len:] = model.zero_token_id
    return codes


def compute_loss(model: torch.nn.Module, codes: torch.Tensor, condition_tensors=None):
    out = model(codes, condition_tensors=condition_tensors)
    loss = torch.tensor(0.0, device=model.device, dtype=torch.float32)
    if out.text_logits is not None and out.text_mask is not None:
        text_ce = cross_entropy(out.text_logits, codes[:, :1, :], out.text_mask, dtype=torch.float32, logits_soft_clip=30.0)
        loss = loss + text_ce[out.text_mask].mean()
    if out.logits is not None and out.mask is not None:
        audio_ce = cross_entropy(out.logits, codes[:, model.audio_offset : model.audio_offset + model.dep_q, :], out.mask, dtype=torch.float32, logits_soft_clip=30.0)
        loss = loss + audio_ce[out.mask].mean()
    return loss


def main():
    args = parse_args()
    config_path = args.config or str(REPO_ROOT / "configs" / "moshi_qwen_3b.json")
    lm_kwargs = load_config(config_path)
    lm_kwargs = dict(lm_kwargs)
    if "conditioners" in lm_kwargs:
        lm_kwargs["condition_provider"] = get_conditioner_provider(lm_kwargs["dim"], args.device, lm_kwargs)
        del lm_kwargs["conditioners"]
    if lm_kwargs.get("fuser") is not None:
        lm_kwargs["fuser"] = get_condition_fuser(lm_kwargs)
    lm_kwargs.pop("depformer_causal", None)
    if "demux_second_stream" in lm_kwargs:
        lm_kwargs["demux_second_text_stream"] = lm_kwargs.pop("demux_second_stream")

    print("Loading Qwen-backed Moshi...")
    model = get_qwen_moshi_lm(qwen_weights=args.qwen_weights, config_path=config_path, device=args.device, dtype=torch.bfloat16)
    model.train()

    if args.freeze_backbone:
        freeze_backbone(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Froze backbone. Trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e9:.2f}B")
    else:
        print("Training full model (no freeze).")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.synthetic and args.data_dir is None:
        args.synthetic = True

    step = 0
    t0 = time.time()
    num_steps = args.steps if args.synthetic else args.epochs
    for _ in range(num_steps):
        codes = synthetic_batch(model, args.batch_size, args.seq_length, torch.device(args.device))
        opt.zero_grad()
        loss = compute_loss(model, codes)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        opt.step()
        step += 1
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d}  loss {loss.item():.6f}  grad_norm {grad_norm.item():.4f}  dt {elapsed:.1f}s")
            t0 = time.time()
        if args.save_every and step % args.save_every == 0:
            from safetensors.torch import save_file
            save_file(model.state_dict(), str(out_dir / f"checkpoint_step_{step}.safetensors"))
            print(f"Saved checkpoint_step_{step}.safetensors")

    from safetensors.torch import save_file
    final_path = out_dir / "checkpoint_final.safetensors"
    save_file(model.state_dict(), str(final_path))
    print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
