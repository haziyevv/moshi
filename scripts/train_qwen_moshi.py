# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fine-tune the Qwen-backed Moshi model (Depformer + audio components, optionally backbone).

Supports:
  - Synthetic random codes (sanity check)
  - Pre-encoded .pt codes (from scripts/preencode_dataset.py)
  - Optional --freeze-backbone to only train Depformer and audio
  - Gradient accumulation for larger effective batch sizes
  - Learning rate warmup + cosine decay
  - Separate text/audio loss logging

Example (synthetic, 100 steps):
  python scripts/train_qwen_moshi.py \\
    --qwen-weights /tmp/qwen_moshi_format.safetensors \\
    --steps 100 --batch-size 2 --seq-length 32 --freeze-backbone --synthetic

Example (Phase 1: warm up Depformer with frozen backbone):
  python scripts/train_qwen_moshi.py \\
    --qwen-weights /tmp/qwen_moshi_format.safetensors \\
    --data-dir ./encoded_codes \\
    --epochs 20 --batch-size 8 --seq-length 256 \\
    --grad-accum 4 --lr 3e-4 --warmup-steps 100 \\
    --freeze-backbone --out-dir runs/phase1

Example (Phase 2: joint training with differential LR):
  python scripts/train_qwen_moshi.py \\
    --qwen-weights runs/phase1/checkpoint_final.safetensors \\
    --data-dir ./encoded_codes \\
    --epochs 50 --batch-size 4 --seq-length 256 \\
    --grad-accum 8 --lr 3e-4 --backbone-lr 1e-5 \\
    --warmup-steps 200 --out-dir runs/phase2
"""

import argparse
import json
import math
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
                   help="Path to converted Qwen safetensors")
    p.add_argument("--config", type=str, default=None, help="Path to moshi_qwen_3b.json")
    p.add_argument("--device", type=str, default="cuda")

    # Training schedule
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of epochs (for real data mode)")
    p.add_argument("--steps", type=int, default=0,
                   help="Max steps (0 = no limit, run full epochs). For synthetic mode, defaults to 500.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-length", type=int, default=256,
                   help="Sequence length in frames (256 = ~20s at 12.5Hz)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5,
                   help="Minimum LR for cosine decay")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear warmup steps")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # Model
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--backbone-lr", type=float, default=0.0,
                   help="Separate LR for backbone (0 = use --lr for everything). "
                        "When set, backbone trains at this LR while Depformer/audio use --lr.")
    p.add_argument("--lora-rank", type=int, default=0)

    # Loss weighting (following official moshi-finetune)
    p.add_argument("--first-codebook-weight", type=float, default=100.0,
                   help="Weight multiplier for first audio codebook (semantic). "
                        "Official moshi-finetune uses 100.0.")
    p.add_argument("--text-padding-weight", type=float, default=0.5,
                   help="Weight multiplier for text padding tokens (EOS/END). "
                        "Official moshi-finetune uses 0.5 to avoid model just predicting padding.")

    # Regularization
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing for cross-entropy (0.0 = none)")
    p.add_argument("--audio-noise-ratio", type=float, default=0.0,
                   help="Fraction of audio codes to randomly replace in input (0.0 = off).")
    p.add_argument("--text-noise-ratio", type=float, default=0.0,
                   help="Fraction of text codes to randomly replace in input (usually keep at 0).")

    # Data
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--data-dir", type=str, default=None)

    # Logging & checkpointing
    p.add_argument("--out-dir", type=str, default="runs/qwen_moshi_ft")
    p.add_argument("--save-every", type=int, default=0,
                   help="Save checkpoint every N steps (0 = save per epoch)")
    p.add_argument("--log-every", type=int, default=10)
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
    """Freeze the temporal transformer backbone and text embedding/projection."""
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ("transformer.", "text_emb.", "text_linear.", "out_norm.")):
            param.requires_grad = False


class PreEncodedDataset(torch.utils.data.Dataset):
    """Dataset of pre-encoded .pt files from scripts/preencode_dataset.py.

    Each file contains {"codes": tensor of shape [1, K, T]}.
    The dataset returns codes of shape [K, seq_length] (randomly cropped).
    """
    def __init__(self, data_dir: str, seq_length: int, zero_token_id: int = -1):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        self.seq_length = seq_length
        self.zero_token_id = zero_token_id
        print(f"PreEncodedDataset: {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(str(self.files[idx]), map_location="cpu", weights_only=True)
        codes = data["codes"].squeeze(0)  # [K, T]
        K, T = codes.shape
        if T >= self.seq_length:
            # Random crop
            start = torch.randint(0, T - self.seq_length + 1, (1,)).item()
            codes = codes[:, start : start + self.seq_length]
        else:
            # Pad with zero_token_id
            pad = torch.full((K, self.seq_length - T), self.zero_token_id, dtype=codes.dtype)
            codes = torch.cat([codes, pad], dim=-1)
        return codes


def synthetic_batch(model: torch.nn.Module, B: int, T: int, device: torch.device) -> torch.Tensor:
    K = model.num_codebooks
    codes = torch.zeros(B, K, T, dtype=torch.long, device=device)
    codes[:, 0, :] = torch.randint(0, model.text_card, (B, T), device=device)
    for k in range(1, K):
        codes[:, k, :] = torch.randint(0, model.card, (B, T), device=device)
    pad_len = max(1, T // 10)
    codes[:, :, -pad_len:] = model.zero_token_id
    return codes


def inject_noise(codes, model, audio_noise_ratio=0.0, text_noise_ratio=0.0):
    """Randomly replace a fraction of input codes with random values.

    This is a simple form of scheduled sampling / input perturbation that
    prevents the model from memorizing exact sequences and makes it robust
    to its own prediction errors during autoregressive inference.
    """
    if audio_noise_ratio <= 0 and text_noise_ratio <= 0:
        return codes

    codes = codes.clone()
    B, K, T = codes.shape
    device = codes.device

    # Audio noise: replace random positions in audio codebooks with random codes
    if audio_noise_ratio > 0 and K > 1:
        audio_codes = codes[:, model.audio_offset:, :]  # [B, n_q, T]
        mask = torch.rand(audio_codes.shape, device=device) < audio_noise_ratio
        # Don't corrupt zero_token_id positions
        mask &= (audio_codes != model.zero_token_id)
        random_codes = torch.randint(0, model.card, audio_codes.shape, device=device)
        audio_codes = torch.where(mask, random_codes, audio_codes)
        codes[:, model.audio_offset:, :] = audio_codes

    # Text noise: replace random positions in text stream
    if text_noise_ratio > 0:
        text_codes = codes[:, :1, :]  # [B, 1, T]
        mask = torch.rand(text_codes.shape, device=device) < text_noise_ratio
        mask &= (text_codes != model.zero_token_id)
        random_text = torch.randint(0, model.text_card, text_codes.shape, device=device)
        text_codes = torch.where(mask, random_text, text_codes)
        codes[:, :1, :] = text_codes

    return codes


def compute_loss(model, codes, label_smoothing=0.0, audio_noise_ratio=0.0,
                 text_noise_ratio=0.0, condition_tensors=None,
                 first_codebook_weight=100.0, text_padding_weight=0.5):
    """Compute text + audio cross-entropy loss with per-codebook and text padding weighting.

    Following the official moshi-finetune approach:
    - First audio codebook gets `first_codebook_weight` multiplier (default 100x)
    - Text padding tokens (EOS/END) get `text_padding_weight` multiplier (default 0.5x)

    Returns (total_loss, text_loss, audio_loss).
    """
    # Inject noise into the INPUT codes (the model still predicts clean targets)
    noisy_codes = inject_noise(codes, model, audio_noise_ratio, text_noise_ratio)
    out = model(noisy_codes, condition_tensors=condition_tensors)

    text_loss = torch.tensor(0.0, device=model.device, dtype=torch.float32)
    audio_loss = torch.tensor(0.0, device=model.device, dtype=torch.float32)

    # --- Text loss with padding down-weighting ---
    if out.text_logits is not None and out.text_mask is not None:
        text_targets = codes[:, :1, :]  # [B, 1, T]
        text_ce = cross_entropy(
            out.text_logits, text_targets, out.text_mask,
            dtype=torch.float32, logits_soft_clip=30.0,
        )
        if out.text_mask.any():
            # Build per-position weights: 1.0 for real text, text_padding_weight for padding
            text_weights = out.text_mask.float()  # [B, 1, T]
            if text_padding_weight != 1.0:
                padding_id = model.existing_text_padding_id
                end_padding_id = model.existing_text_end_padding_id
                is_padding = (text_targets == padding_id) | (text_targets == end_padding_id)
                text_weights = torch.where(is_padding & out.text_mask,
                                           text_weights * text_padding_weight,
                                           text_weights)
            # Weighted mean
            weighted_ce = text_ce * text_weights
            text_loss = weighted_ce.sum() / text_weights.sum().clamp(min=1.0)

    # --- Audio loss with first codebook upweighting ---
    if out.logits is not None and out.mask is not None:
        audio_targets = codes[:, model.audio_offset:model.audio_offset + model.dep_q, :]
        audio_ce = cross_entropy(
            out.logits, audio_targets, out.mask,
            dtype=torch.float32, logits_soft_clip=30.0,
        )
        if out.mask.any():
            # Build per-codebook weights: first codebook gets higher weight
            audio_weights = out.mask.float()  # [B, dep_q, T]
            if first_codebook_weight != 1.0:
                audio_weights[:, 0, :] = audio_weights[:, 0, :] * first_codebook_weight
            # Weighted mean
            weighted_ce = audio_ce * audio_weights
            audio_loss = weighted_ce.sum() / audio_weights.sum().clamp(min=1.0)

    total_loss = text_loss + audio_loss
    return total_loss, text_loss, audio_loss


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return max_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    config_path = args.config or str(REPO_ROOT / "configs" / "moshi_qwen_3b.json")

    print("=" * 70)
    print("TRAINING CONFIG")
    print("=" * 70)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:20s}: {v}")
    print("=" * 70)

    # ---- Load model ----
    print("\nLoading Qwen-backed Moshi...")
    model = get_qwen_moshi_lm(
        qwen_weights=args.qwen_weights,
        config_path=config_path,
        device=args.device,
        dtype=torch.bfloat16,
    )
    model.train()

    if args.freeze_backbone:
        freeze_backbone(model)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_trainable / 1e6:.1f}M trainable / {n_total / 1e6:.1f}M total")
    if args.freeze_backbone:
        print("  (backbone frozen: only Depformer + audio embeddings + audio projections are trained)")

    # ---- Optimizer with optional differential LR ----
    backbone_prefixes = ("transformer.", "text_emb.", "text_linear.", "out_norm.")
    if args.backbone_lr > 0 and not args.freeze_backbone:
        # Separate param groups: backbone at lower LR, rest at --lr
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(name.startswith(pfx) for pfx in backbone_prefixes):
                backbone_params.append(param)
            else:
                other_params.append(param)
        param_groups = [
            {"params": backbone_params, "lr": args.backbone_lr, "label": "backbone"},
            {"params": other_params, "lr": args.lr, "label": "depformer+audio"},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        trainable_params = backbone_params + other_params
        n_bb = sum(p.numel() for p in backbone_params)
        n_other = sum(p.numel() for p in other_params)
        print(f"  Differential LR: backbone ({n_bb / 1e6:.0f}M) @ {args.backbone_lr}, "
              f"Depformer+audio ({n_other / 1e6:.0f}M) @ {args.lr}")
        use_differential_lr = True
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        use_differential_lr = False

    # ---- Data ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.synthetic and args.data_dir is None:
        print("No --data-dir provided, falling back to --synthetic mode.")
        args.synthetic = True

    if args.synthetic:
        max_steps = args.steps if args.steps > 0 else 500
        total_steps = max_steps
        print(f"\nSynthetic mode: {max_steps} steps")
    else:
        dataset = PreEncodedDataset(args.data_dir, args.seq_length, zero_token_id=model.zero_token_id)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        batches_per_epoch = len(dataloader)
        # Account for gradient accumulation in step counting
        steps_per_epoch = batches_per_epoch // args.grad_accum
        total_steps = steps_per_epoch * args.epochs
        if args.steps > 0:
            total_steps = min(total_steps, args.steps)
        eff_batch = args.batch_size * args.grad_accum
        print(f"\nReal data: {len(dataset)} samples, {batches_per_epoch} batches/epoch")
        print(f"Effective batch size: {args.batch_size} x {args.grad_accum} (grad_accum) = {eff_batch}")
        print(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, Epochs: {args.epochs}")
        print(f"LR schedule: warmup {args.warmup_steps} steps -> peak {args.lr} -> cosine to {args.min_lr}")

    # ---- Training ----
    step = 0
    best_loss = float("inf")
    t0 = time.time()
    log_losses = {"total": 0.0, "text": 0.0, "audio": 0.0, "count": 0}

    def log_step(step, loss_t, loss_text, loss_audio, grad_norm, epoch=None):
        nonlocal t0, log_losses, best_loss
        log_losses["total"] += loss_t
        log_losses["text"] += loss_text
        log_losses["audio"] += loss_audio
        log_losses["count"] += 1

        if step % args.log_every == 0 and log_losses["count"] > 0:
            n = log_losses["count"]
            avg_t = log_losses["total"] / n
            avg_text = log_losses["text"] / n
            avg_audio = log_losses["audio"] / n
            elapsed = time.time() - t0
            current_lr = get_lr(step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            ep_str = f"ep {epoch:3d}  " if epoch is not None else ""
            print(
                f"  {ep_str}step {step:6d}/{total_steps}  "
                f"loss {avg_t:.4f} (text {avg_text:.4f} + audio {avg_audio:.4f})  "
                f"gnorm {grad_norm:.3f}  lr {current_lr:.2e}  dt {elapsed:.1f}s"
            )
            if avg_t < best_loss:
                best_loss = avg_t
            log_losses = {"total": 0.0, "text": 0.0, "audio": 0.0, "count": 0}
            t0 = time.time()

    def maybe_save(step, tag=None):
        if tag:
            path = out_dir / f"checkpoint_{tag}.safetensors"
        else:
            path = out_dir / f"checkpoint_step_{step}.safetensors"
        from safetensors.torch import save_file
        save_file(model.state_dict(), str(path))
        print(f"  -> Saved {path}")

    def update_lr(step):
        """Update learning rates for all param groups, respecting differential LR."""
        scale = get_lr(step, args.warmup_steps, total_steps, 1.0, args.min_lr / args.lr)
        for pg in opt.param_groups:
            if use_differential_lr and pg.get("label") == "backbone":
                pg["lr"] = args.backbone_lr * scale
            else:
                pg["lr"] = args.lr * scale

    print(f"\nStarting training...\n")

    if args.synthetic:
        max_steps = total_steps
        for s in range(max_steps):
            update_lr(step)

            codes = synthetic_batch(model, args.batch_size, args.seq_length, torch.device(args.device))
            opt.zero_grad()
            loss, text_loss, audio_loss = compute_loss(
                model, codes,
                label_smoothing=args.label_smoothing,
                audio_noise_ratio=args.audio_noise_ratio,
                text_noise_ratio=args.text_noise_ratio,
                first_codebook_weight=args.first_codebook_weight,
                text_padding_weight=args.text_padding_weight,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
            opt.step()
            step += 1
            log_step(step, loss.item(), text_loss.item(), audio_loss.item(), grad_norm.item())
            if args.save_every and step % args.save_every == 0:
                maybe_save(step)
    else:
        done = False
        for epoch in range(1, args.epochs + 1):
            if done:
                break
            print(f"--- Epoch {epoch}/{args.epochs} ---")
            epoch_loss = 0.0
            epoch_steps = 0
            micro_step = 0

            opt.zero_grad()
            for batch_codes in dataloader:
                batch_codes = batch_codes.to(args.device)

                # Forward + backward (accumulate)
                loss, text_loss, audio_loss = compute_loss(
                    model, batch_codes,
                    label_smoothing=args.label_smoothing,
                    audio_noise_ratio=args.audio_noise_ratio,
                    text_noise_ratio=args.text_noise_ratio,
                    first_codebook_weight=args.first_codebook_weight,
                    text_padding_weight=args.text_padding_weight,
                )
                scaled_loss = loss / args.grad_accum
                scaled_loss.backward()
                micro_step += 1

                epoch_loss += loss.item()

                # Step optimizer every grad_accum micro-steps
                if micro_step % args.grad_accum == 0:
                    # Update LR
                    update_lr(step)

                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    step += 1
                    epoch_steps += 1

                    log_step(step, loss.item(), text_loss.item(), audio_loss.item(), grad_norm.item(), epoch)

                    if args.save_every and step % args.save_every == 0:
                        maybe_save(step)
                    if args.steps > 0 and step >= args.steps:
                        done = True
                        break

            avg_epoch_loss = epoch_loss / max(micro_step, 1)
            print(f"  Epoch {epoch} complete. Avg loss: {avg_epoch_loss:.4f} ({epoch_steps} optimizer steps)\n")

            # Save per-epoch checkpoint
            maybe_save(step, tag=f"epoch_{epoch}")

            if done:
                break

    # Final checkpoint
    maybe_save(step, tag="final")
    print(f"\nTraining complete. Best avg loss: {best_loss:.4f}")
    print(f"Total optimizer steps: {step}")


if __name__ == "__main__":
    main()
