#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Fine-tune the Qwen-backed Moshi model (Depformer + audio, optionally backbone).

Key design decisions (lessons learned from previous attempts):
  - Noise injection targets ONLY Moshi-side codebooks (CB0-7), not user audio,
    because user audio is always clean at inference.
  - Text noise skips padding tokens so the padding/text boundary stays intact.
  - First codebook weight defaults to 100 (matching official moshi-finetune).
  - Backbone LR default is 5e-5 (not 1e-5) so the backbone actually adapts.

Phase 0 (embedding warmup, trains audio embeddings + depformer only)::

    python scripts/train_qwen_moshi.py \
        --qwen-weights qwen_7b_moshi_format.safetensors \
        --data-dir ./encoded_codes_all \
        --epochs 20 --batch-size 8 --seq-length 256 \
        --grad-accum 4 --lr 3e-4 --warmup-steps 200 \
        --freeze-backbone --embed-warmup-steps 500 \
        --out-dir runs/phase0_phase1 \
        --eval-every 250 --eval-sample 36000

Phase 1 (warm up Depformer, frozen backbone — runs automatically after Phase 0)::

    # If running separately without Phase 0:
    python scripts/train_qwen_moshi.py \
        --qwen-weights qwen_7b_moshi_format.safetensors \
        --data-dir ./encoded_codes_all \
        --epochs 20 --batch-size 8 --seq-length 256 \
        --grad-accum 4 --lr 3e-4 --warmup-steps 200 \
        --freeze-backbone --out-dir runs/phase1 \
        --eval-every 500 --eval-sample 36000

Phase 2 (joint training with noise injection)::

    python scripts/train_qwen_moshi.py \
        --qwen-weights runs/phase0_phase1/checkpoint_final.safetensors \
        --data-dir ./encoded_codes_all \
        --epochs 50 --batch-size 4 --seq-length 256 \
        --grad-accum 8 --lr 3e-4 --backbone-lr 5e-5 \
        --warmup-steps 200 \
        --audio-noise-ratio 0.10 --text-noise-ratio 0.02 \
        --out-dir runs/phase2  --eval-every 500 --eval-sample 36000


Resume from checkpoint::

    python scripts/train_qwen_moshi.py \
        --qwen-weights runs/phase2/checkpoint_step_500.safetensors \
        --resume runs/phase2/training_state.pt \
        --data-dir ./encoded_codes_all \
        --out-dir runs/phase2
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from moshi.models import get_qwen_moshi_lm, LMGen
from moshi.models.loaders import CheckpointInfo
from moshi.utils.utils import cross_entropy


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen-backed Moshi (single GPU)")
    p.add_argument("--qwen-weights", type=str, required=True,
                   help="Path to converted Qwen safetensors (or prior checkpoint)")
    p.add_argument("--config", type=str, default=None,
                   help="Path to model config (default: moshi_qwen_7b.json)")
    p.add_argument("--device", type=str, default="cuda")

    # Training schedule
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps", type=int, default=0,
                   help="Max steps (0 = no limit, run full epochs)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-length", type=int, default=256,
                   help="Sequence length in frames (256 ~ 20s at 12.5Hz)")
    p.add_argument("--grad-accum", type=int, default=1)

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # Model
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--backbone-lr", type=float, default=0.0,
                   help="Separate LR for backbone (0 = use --lr for everything). "
                        "Recommended: 5e-5 for Phase 2.")
    p.add_argument("--embed-warmup-steps", type=int, default=0,
                   help="Phase 0: train only audio embeddings + depformer for N steps "
                        "before switching to the main freeze mode. "
                        "Recommended: 300-500 steps. 0 = skip Phase 0.")

    # Loss weighting (matching official moshi-finetune)
    p.add_argument("--first-codebook-weight", type=float, default=100.0,
                   help="Weight for first audio codebook (semantic). Official default: 100.")
    p.add_argument("--early-codebook-weight", type=float, default=1.0,
                   help="Weight for codebooks 1-3 (pitch/formants). "
                        "Recommended: 10.0 for Phase 3 to improve audio quality.")
    p.add_argument("--text-padding-weight", type=float, default=0.5,
                   help="Weight for text padding tokens. Official default: 0.5.")

    # Noise injection for exposure bias reduction
    p.add_argument("--audio-noise-ratio", type=float, default=0.0,
                   help="Fraction of MOSHI-SIDE audio codes to randomly replace. "
                        "Recommended: 0.10 for Phase 2.")
    p.add_argument("--text-noise-ratio", type=float, default=0.0,
                   help="Fraction of text codes to randomly replace. "
                        "Recommended: 0.02 for Phase 2.")

    # Data
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Single data directory (ignored if --data-dirs is set)")
    p.add_argument("--data-dirs", nargs="+", default=None,
                   help="Multiple data directories to combine")

    # Logging & checkpointing
    p.add_argument("--out-dir", type=str, default="runs/qwen_moshi_ft")
    p.add_argument("--save-every", type=int, default=0,
                   help="Save checkpoint every N steps (0 = save per epoch)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to training_state.pt to resume optimizer/step/epoch")

    # Qualitative eval during training
    p.add_argument("--eval-every", type=int, default=0,
                   help="Run qualitative eval every N optimizer steps (0 = disabled). "
                        "Generates WAVs and decoded text so you can hear/read progress.")
    p.add_argument("--eval-sample", type=int, default=0,
                   help="Index of the sample in --data-dir to use for eval (default: 0)")
    p.add_argument("--eval-ar-frames", type=int, default=100,
                   help="Number of AR frames to generate during eval (100 ~ 8s)")
    return p.parse_args()


class PreEncodedDataset(torch.utils.data.Dataset):
    """Dataset of pre-encoded .pt files from scripts/preencode_dataset.py.

    Each file contains {"codes": tensor [1, K, T]}.
    Returns codes of shape [K, seq_length] (randomly cropped).
    """
    def __init__(self, data_dir: str | list[str], seq_length: int, zero_token_id: int = -1):
        if isinstance(data_dir, (list, tuple)):
            dirs = [Path(d) for d in data_dir]
            self.files = sorted(f for d in dirs for f in d.glob("*.pt"))
            if not self.files:
                raise FileNotFoundError(f"No .pt files found in {data_dir}")
            print(f"PreEncodedDataset: {len(self.files)} files from {len(dirs)} dirs")
        else:
            self.data_dir = Path(data_dir)
            self.files = sorted(self.data_dir.glob("*.pt"))
            if not self.files:
                raise FileNotFoundError(f"No .pt files found in {data_dir}")
            print(f"PreEncodedDataset: {len(self.files)} files from {data_dir}")
        self.seq_length = seq_length
        self.zero_token_id = zero_token_id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(str(self.files[idx]), map_location="cpu", weights_only=True)
        codes = data["codes"].squeeze(0)  # [K, T]
        K, T = codes.shape
        if T >= self.seq_length:
            start = torch.randint(0, T - self.seq_length + 1, (1,)).item()
            codes = codes[:, start : start + self.seq_length]
        else:
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
    """Replace a fraction of input codes with random values for exposure bias reduction.

    Audio noise targets ONLY Moshi-side codebooks (CB0-7), not user audio.
    Text noise skips padding and end-padding tokens to preserve boundary signals.
    """
    if audio_noise_ratio <= 0 and text_noise_ratio <= 0:
        return codes

    codes = codes.clone()
    B, K, T = codes.shape
    device = codes.device

    if audio_noise_ratio > 0:
        moshi_start = model.audio_offset
        moshi_end = model.audio_offset + model.dep_q
        moshi_codes = codes[:, moshi_start:moshi_end, :]
        mask = torch.rand(moshi_codes.shape, device=device) < audio_noise_ratio
        mask &= (moshi_codes != model.zero_token_id)
        random_codes = torch.randint(0, model.card, moshi_codes.shape, device=device)
        moshi_codes = torch.where(mask, random_codes, moshi_codes)
        codes[:, moshi_start:moshi_end, :] = moshi_codes

    if text_noise_ratio > 0:
        text_codes = codes[:, :1, :]
        mask = torch.rand(text_codes.shape, device=device) < text_noise_ratio
        mask &= (text_codes != model.zero_token_id)
        mask &= (text_codes != model.existing_text_padding_id)
        mask &= (text_codes != model.existing_text_end_padding_id)
        random_text = torch.randint(0, model.text_card, text_codes.shape, device=device)
        text_codes = torch.where(mask, random_text, text_codes)
        codes[:, :1, :] = text_codes

    return codes


def compute_loss(model, codes, audio_noise_ratio=0.0, text_noise_ratio=0.0,
                 condition_tensors=None, first_codebook_weight=100.0,
                 early_codebook_weight=1.0, text_padding_weight=0.5):
    """Compute text + audio loss with per-codebook weighting.

    Returns (total_loss, text_loss, audio_loss).
    """
    noisy_codes = inject_noise(codes, model, audio_noise_ratio, text_noise_ratio)
    out = model(noisy_codes, condition_tensors=condition_tensors)

    text_loss = torch.tensor(0.0, device=codes.device, dtype=torch.float32)
    audio_loss = torch.tensor(0.0, device=codes.device, dtype=torch.float32)

    if out.text_logits is not None and out.text_mask is not None:
        text_targets = codes[:, :1, :]
        text_ce = cross_entropy(
            out.text_logits, text_targets, out.text_mask,
            dtype=torch.float32, logits_soft_clip=30.0,
        )
        if out.text_mask.any():
            text_weights = out.text_mask.float()
            if text_padding_weight != 1.0:
                is_padding = (
                    (text_targets == model.existing_text_padding_id)
                    | (text_targets == model.existing_text_end_padding_id)
                )
                text_weights = torch.where(
                    is_padding & out.text_mask,
                    text_weights * text_padding_weight,
                    text_weights,
                )
            weighted_ce = text_ce * text_weights
            text_loss = weighted_ce.sum() / text_weights.sum().clamp(min=1.0)

    if out.logits is not None and out.mask is not None:
        audio_targets = codes[:, model.audio_offset:model.audio_offset + model.dep_q, :]
        audio_ce = cross_entropy(
            out.logits, audio_targets, out.mask,
            dtype=torch.float32, logits_soft_clip=30.0,
        )
        if out.mask.any():
            audio_weights = out.mask.float()
            if first_codebook_weight != 1.0:
                audio_weights[:, 0, :] = audio_weights[:, 0, :] * first_codebook_weight
            if early_codebook_weight != 1.0:
                n_early = min(3, audio_weights.shape[1] - 1)
                audio_weights[:, 1:1 + n_early, :] = (
                    audio_weights[:, 1:1 + n_early, :] * early_codebook_weight
                )
            weighted_ce = audio_ce * audio_weights
            audio_loss = weighted_ce.sum() / audio_weights.sum().clamp(min=1.0)

    total_loss = text_loss + audio_loss
    return total_loss, text_loss, audio_loss


def get_lr(step: int, warmup_steps: int, total_steps: int,
           max_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return max_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def run_eval(model, eval_codes, step_num, out_dir, mimi, tokenizer,
             max_ar_frames=100):
    """Qualitative eval: teacher-forced accuracy/audio + AR generation + text decode."""
    import sphn

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    sr = mimi.sample_rate

    eval_dir = out_dir / "eval_samples"
    eval_dir.mkdir(exist_ok=True)

    T = min(eval_codes.shape[-1], max_ar_frames)
    codes = eval_codes[:, :, :T].to(device)

    print(f"\n{'=' * 70}")
    print(f"  QUALITATIVE EVAL @ step {step_num}  ({T} frames = {T / 12.5:.1f}s)")
    print(f"{'=' * 70}")

    # ---- Teacher-forced ----
    out = model(codes)

    if out.text_logits is not None and out.text_mask is not None:
        text_preds = out.text_logits.argmax(dim=-1)
        text_targets = codes[:, :1, :]
        valid = out.text_mask[0, 0]
        if valid.any():
            text_acc = (text_preds[0, 0, valid] == text_targets[0, 0, valid]).float().mean().item()
            non_pad = valid & (text_targets[0, 0] != model.existing_text_padding_id)
            n_np = non_pad.sum().item()
            text_acc_np = ((text_preds[0, 0, non_pad] == text_targets[0, 0, non_pad])
                          .float().mean().item()) if n_np > 0 else float("nan")
            print(f"  Text acc: {text_acc:.1%} (all) | {text_acc_np:.1%} (non-pad, n={n_np})")

            if tokenizer is not None and n_np > 0:
                gt_text = tokenizer.decode(text_targets[0, 0, non_pad].tolist(),
                                           skip_special_tokens=True)
                pred_text = tokenizer.decode(text_preds[0, 0, non_pad].tolist(),
                                             skip_special_tokens=True)
                print(f"  GT text:   {gt_text[:120]}")
                print(f"  Pred text: {pred_text[:120]}")

    cb_accs = []
    if out.logits is not None and out.mask is not None:
        audio_preds = out.logits.argmax(dim=-1)
        audio_targets = codes[:, model.audio_offset:model.audio_offset + model.dep_q, :]
        for cb in range(model.dep_q):
            cb_mask = out.mask[0, cb]
            if cb_mask.any():
                cb_accs.append((audio_preds[0, cb, cb_mask] == audio_targets[0, cb, cb_mask])
                               .float().mean().item())
            else:
                cb_accs.append(0.0)
        print(f"  Audio acc: CB0 {cb_accs[0]:.1%} | Overall {np.mean(cb_accs):.1%}")

        pred_clamped = audio_preds.clamp(0, model.card - 1)
        pred_pcm = mimi.decode(pred_clamped)
        sphn.write_wav(str(eval_dir / f"step{step_num:06d}_teacher.wav"),
                       pred_pcm[0, 0].cpu().float().numpy(), sr)

        gt_path = eval_dir / "ground_truth_moshi.wav"
        if not gt_path.exists():
            gt_clamped = audio_targets.clamp(0, model.card - 1)
            gt_pcm = mimi.decode(gt_clamped)
            sphn.write_wav(str(gt_path), gt_pcm[0, 0].cpu().float().numpy(), sr)

            user_codes_gt = codes[:, model.audio_offset + model.dep_q:, :]
            user_clamped = user_codes_gt.clamp(0, model.card - 1)
            user_pcm = mimi.decode(user_clamped)
            sphn.write_wav(str(eval_dir / "ground_truth_user.wav"),
                           user_pcm[0, 0].cpu().float().numpy(), sr)
            print(f"  Saved ground truth WAVs (one-time)")

    # ---- AR generation ----
    user_codes_ar = codes[:, model.audio_offset + model.dep_q:, :]
    lm_gen = LMGen(model, use_sampling=True, temp=0.8, temp_text=0.7,
                   top_k=250, top_k_text=25)
    gen_audio_chunks = []
    gen_text_tokens = []

    with mimi.streaming(1), lm_gen.streaming(1):
        for t in range(T):
            user_frame = user_codes_ar[:, :, t : t + 1]
            tokens = lm_gen.step(user_frame)
            if tokens is None:
                continue
            gen_audio_chunks.append(mimi.decode(tokens[:, 1:, :]))
            gen_text_tokens.append(tokens[0, 0, 0].item())

    if gen_audio_chunks:
        gen_pcm = torch.cat(gen_audio_chunks, dim=-1)
        rms = torch.sqrt(torch.mean(gen_pcm ** 2)).item()
        sphn.write_wav(str(eval_dir / f"step{step_num:06d}_ar.wav"),
                       gen_pcm[0, 0].cpu().float().numpy(), sr)

        non_pad = [t for t in gen_text_tokens if t != model.existing_text_padding_id]
        if tokenizer is not None and non_pad:
            ar_text = tokenizer.decode(non_pad, skip_special_tokens=True)
            print(f"  AR text:   {ar_text[:120]}")
        print(f"  AR audio:  RMS={rms:.5f}, {len(gen_audio_chunks)} frames")

    print(f"  Saved: step{step_num:06d}_teacher.wav, step{step_num:06d}_ar.wav")
    print(f"{'=' * 70}\n")

    if was_training:
        model.train()


# ---- Freeze functions ----

def freeze_for_phase0(model: torch.nn.Module) -> None:
    """Phase 0: freeze everything EXCEPT audio embeddings + depformer + audio heads.

    This teaches the backbone what audio tokens mean in its embedding
    space before we train the Depformer to rely on those representations.

    Unfrozen:
      - emb.0 - emb.15          (backbone audio codebook embeddings)
      - depformer*               (depformer everything)
      - linears.*                (audio output projections)

    Frozen:
      - transformer.*            (backbone transformer layers)
      - text_emb.*               (backbone text embedding - already good from Qwen)
      - text_linear.*            (backbone text output projection)
      - out_norm.*               (backbone output norm)
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then selectively unfreeze
    unfrozen_names = []

    for name, param in model.named_parameters():
        should_unfreeze = False

        # Audio embeddings for the backbone (emb.0 through emb.15)
        if name.startswith("emb.") and not name.startswith("emb_"):
            should_unfreeze = True

        # All depformer parameters
        if name.startswith("depformer"):
            should_unfreeze = True

        # Audio output linear projections
        if name.startswith("linears."):
            should_unfreeze = True

        if should_unfreeze:
            param.requires_grad = True
            unfrozen_names.append(name)

    unfrozen_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Phase 0 freeze: {unfrozen_count / 1e6:.1f}M unfrozen, "
          f"{frozen_count / 1e6:.1f}M frozen")
    print(f"  Unfrozen groups: audio emb ({sum(1 for n in unfrozen_names if n.startswith('emb.'))}), "
          f"depformer ({sum(1 for n in unfrozen_names if n.startswith('depformer'))}), "
          f"linears ({sum(1 for n in unfrozen_names if n.startswith('linears.'))})")


def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze the temporal transformer backbone and text embedding/projection."""
    for name, param in model.named_parameters():
        if any(name.startswith(pfx) for pfx in
               ("transformer.", "text_emb.", "text_linear.", "out_norm.")):
            param.requires_grad = False


def apply_freeze_mode(model, args, phase0_active):
    """Apply the correct freeze mode based on current training phase."""
    if phase0_active:
        freeze_for_phase0(model)
    elif args.freeze_backbone:
        freeze_backbone(model)
    # else: everything unfrozen (Phase 2 / joint training)


def build_optimizer(model, args, backbone_prefixes):
    """Build AdamW with proper weight-decay exclusion and optional differential LR."""
    use_differential_lr = args.backbone_lr > 0 and not args.freeze_backbone

    # Exclude biases and 1D params (LayerNorm/RMSNorm) from weight decay
    decay_params, no_decay_params = [], []
    backbone_decay, backbone_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = any(name.startswith(pfx) for pfx in backbone_prefixes)
        no_wd = param.ndim <= 1 or name.endswith(".bias")

        if use_differential_lr and is_backbone:
            (backbone_no_decay if no_wd else backbone_decay).append(param)
        else:
            (no_decay_params if no_wd else decay_params).append(param)

    groups = [
        {"params": decay_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "lr": args.lr, "weight_decay": 0.0},
    ]
    if use_differential_lr:
        groups += [
            {"params": backbone_decay, "lr": args.backbone_lr,
             "weight_decay": args.weight_decay, "label": "backbone"},
            {"params": backbone_no_decay, "lr": args.backbone_lr,
             "weight_decay": 0.0, "label": "backbone"},
        ]
        n_bb = sum(p.numel() for p in backbone_decay + backbone_no_decay)
        n_other = sum(p.numel() for p in decay_params + no_decay_params)
        print(f"  Differential LR: backbone ({n_bb / 1e6:.0f}M) @ {args.backbone_lr}, "
              f"Depformer+audio ({n_other / 1e6:.0f}M) @ {args.lr}")

    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    n_wd = sum(p.numel() for p in decay_params + backbone_decay)
    n_no_wd = sum(p.numel() for p in no_decay_params + backbone_no_decay)
    print(f"  Optimizer: {n_trainable / 1e6:.1f}M trainable params")
    print(f"  Weight decay: {n_wd / 1e6:.0f}M params with wd={args.weight_decay}, "
          f"{n_no_wd / 1e6:.0f}M without")

    return torch.optim.AdamW(groups, betas=(0.9, 0.95)), use_differential_lr


def main():
    args = parse_args()
    config_path = args.config or str(REPO_ROOT / "configs" / "moshi_qwen_3b.json")
    device = torch.device(args.device)

    print("=" * 70)
    print("TRAINING CONFIG")
    print("=" * 70)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:24s}: {v}")
    print(f"  {'device':24s}: {device}")
    print("=" * 70)

    # ---- Load model ----
    print("\nLoading Qwen-backed Moshi...")
    model = get_qwen_moshi_lm(
        qwen_weights=args.qwen_weights,
        config_path=config_path,
        device=device,
        dtype=torch.bfloat16,
    )
    model.train()

    # ---- Apply freeze mode ----
    phase0_active = args.embed_warmup_steps > 0
    if phase0_active:
        print(f"\nPhase 0: Embedding warmup for {args.embed_warmup_steps} steps")
        freeze_for_phase0(model)
    elif args.freeze_backbone:
        print("\nPhase 1: Backbone frozen")
        freeze_backbone(model)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    phase_str = "Phase 0 (embed warmup)" if phase0_active else \
                ("backbone frozen" if args.freeze_backbone else "all unfrozen")
    print(f"Parameters: {n_trainable / 1e6:.1f}M trainable / {n_total / 1e6:.1f}M total"
          f" ({phase_str})")

    # ---- Optimizer ----
    backbone_prefixes = ("transformer.", "text_emb.", "text_linear.", "out_norm.")
    opt, use_differential_lr = build_optimizer(model, args, backbone_prefixes)

    # ---- Data ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside checkpoints for reproducibility
    import shutil
    shutil.copy2(config_path, out_dir / "model_config.json")

    data_dir_or_dirs = args.data_dirs if args.data_dirs else args.data_dir
    if not args.synthetic and data_dir_or_dirs is None:
        print("No --data-dir / --data-dirs provided, falling back to --synthetic mode.")
        args.synthetic = True

    if args.synthetic:
        total_steps = args.steps if args.steps > 0 else 500
        print(f"\nSynthetic mode: {total_steps} steps")
    else:
        dataset = PreEncodedDataset(data_dir_or_dirs, args.seq_length,
                                    zero_token_id=model.zero_token_id)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
        )
        batches_per_epoch = len(dataloader)
        steps_per_epoch = batches_per_epoch // args.grad_accum
        total_steps = steps_per_epoch * args.epochs
        if args.steps > 0:
            total_steps = min(total_steps, args.steps)
        eff_batch = args.batch_size * args.grad_accum
        n_dropped = len(dataset) - batches_per_epoch * args.batch_size
        print(
            f"\nReal data: {len(dataset)} samples, {batches_per_epoch} batches/epoch"
            f" ({n_dropped} samples dropped per epoch)\n"
            f"Effective batch size: {args.batch_size} x {args.grad_accum} accum = {eff_batch}\n"
            f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, Epochs: {args.epochs}\n"
            f"LR: warmup {args.warmup_steps} -> peak {args.lr} -> cosine to {args.min_lr}"
        )

    # ---- Resume ----
    start_step = 0
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        start_epoch = state.get("epoch", 1)
        best_loss = state.get("best_loss", float("inf"))
        # If resuming past Phase 0, deactivate it
        if phase0_active and start_step >= args.embed_warmup_steps:
            print(f"  Resumed past Phase 0 ({start_step} >= {args.embed_warmup_steps})")
            phase0_active = False
            if args.freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = True
                freeze_backbone(model)
                opt, use_differential_lr = build_optimizer(model, args, backbone_prefixes)
        print(f"  Resumed at step {start_step}, epoch {start_epoch}, best_loss {best_loss:.4f}")

    # ---- Eval setup ----
    mimi = None
    tokenizer = None
    eval_codes = None

    if args.eval_every > 0 and not args.synthetic:
        print("\nLoading Mimi codec for eval...")
        ckpt = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        mimi = ckpt.get_mimi(device=device)
        mimi.eval()

        print("Loading Qwen tokenizer for eval...")
        from transformers import AutoTokenizer
        tok_name = "Qwen/Qwen2.5-7B" if "7b" in config_path.lower() else "Qwen/Qwen2.5-3B"
        tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        eval_path = dataset.files[min(args.eval_sample, len(dataset.files) - 1)]
        eval_data = torch.load(str(eval_path), map_location="cpu", weights_only=True)
        eval_codes = eval_data["codes"]
        print(f"Eval sample: {eval_path.name} ({eval_codes.shape[-1]} frames = "
              f"{eval_codes.shape[-1] / 12.5:.1f}s), eval every {args.eval_every} steps")

    # ---- Training loop ----
    step = start_step
    t0 = time.time()
    log_losses = {"total": 0.0, "text": 0.0, "audio": 0.0, "count": 0}

    def log_step(step_num, loss_t, loss_text, loss_audio, grad_norm, epoch=None):
        nonlocal t0, log_losses, best_loss
        log_losses["total"] += loss_t
        log_losses["text"] += loss_text
        log_losses["audio"] += loss_audio
        log_losses["count"] += 1

        if step_num % args.log_every == 0 and log_losses["count"] > 0:
            n = log_losses["count"]
            avg_t = log_losses["total"] / n
            avg_text = log_losses["text"] / n
            avg_audio = log_losses["audio"] / n
            elapsed = time.time() - t0
            current_lr = get_lr(step_num, args.warmup_steps, total_steps, args.lr, args.min_lr)
            ep_str = f"ep {epoch:3d}  " if epoch is not None else ""
            phase_tag = "[P0] " if phase0_active else ""
            print(
                f"  {phase_tag}{ep_str}step {step_num:6d}/{total_steps}  "
                f"loss {avg_t:.4f} (text {avg_text:.4f} + audio {avg_audio:.4f})  "
                f"gnorm {grad_norm:.3f}  lr {current_lr:.2e}  dt {elapsed:.1f}s"
            )
            if avg_t < best_loss:
                best_loss = avg_t
            log_losses = {"total": 0.0, "text": 0.0, "audio": 0.0, "count": 0}
            t0 = time.time()

    def save_checkpoint(step_num, tag=None):
        path = out_dir / (f"checkpoint_{tag}.safetensors" if tag
                          else f"checkpoint_step_{step_num}.safetensors")
        from safetensors.torch import save_file
        save_file(model.state_dict(), str(path))
        print(f"  -> Saved {path}")

    def save_training_state(step_num, epoch):
        state_path = out_dir / "training_state.pt"
        torch.save({
            "step": step_num,
            "epoch": epoch,
            "optimizer": opt.state_dict(),
            "best_loss": best_loss,
        }, str(state_path))

    def update_lr(step_num):
        for pg in opt.param_groups:
            if use_differential_lr and pg.get("label") == "backbone":
                min_bb_lr = max(args.min_lr * (args.backbone_lr / args.lr), 1e-6)
                pg["lr"] = get_lr(step_num, args.warmup_steps, total_steps,
                                  args.backbone_lr, min_bb_lr)
            else:
                pg["lr"] = get_lr(step_num, args.warmup_steps, total_steps,
                                  args.lr, args.min_lr)

    def maybe_transition_from_phase0(step_num, epoch):
        """Check if Phase 0 is done and switch to Phase 1."""
        nonlocal phase0_active, opt, use_differential_lr
        if not phase0_active:
            return
        if step_num < args.embed_warmup_steps:
            return

        print(f"\n{'=' * 70}")
        print(f"  PHASE 0 COMPLETE at step {step_num}")
        print(f"  Transitioning to {'Phase 1 (backbone frozen)' if args.freeze_backbone else 'joint training'}...")
        print(f"{'=' * 70}")

        # Save Phase 0 checkpoint
        save_checkpoint(step_num, tag="phase0_done")

        # Switch freeze mode
        phase0_active = False

        # Unfreeze everything first, then apply target freeze
        for param in model.parameters():
            param.requires_grad = True

        if args.freeze_backbone:
            freeze_backbone(model)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Now trainable: {n_trainable / 1e6:.1f}M parameters")

        # Rebuild optimizer for new param groups
        opt, use_differential_lr = build_optimizer(model, args, backbone_prefixes)
        print()

    print(f"\nStarting training...\n")

    if args.synthetic:
        opt.zero_grad()
        for _ in range(total_steps):
            update_lr(step)
            codes = synthetic_batch(model, args.batch_size, args.seq_length, device)
            loss, text_loss, audio_loss = compute_loss(
                model, codes,
                audio_noise_ratio=args.audio_noise_ratio,
                text_noise_ratio=args.text_noise_ratio,
                first_codebook_weight=args.first_codebook_weight,
                early_codebook_weight=args.early_codebook_weight,
                text_padding_weight=args.text_padding_weight,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
            opt.step()
            opt.zero_grad()
            step += 1
            log_step(step, loss.item(), text_loss.item(), audio_loss.item(), grad_norm)
            maybe_transition_from_phase0(step, 0)
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(step)
    else:
        done = False
        grad_norm = 0.0

        for epoch in range(start_epoch, args.epochs + 1):
            if done:
                break
            print(f"--- Epoch {epoch}/{args.epochs} ---")
            epoch_loss = 0.0
            epoch_steps = 0
            micro_step = 0
            accum_losses = {"total": 0.0, "text": 0.0, "audio": 0.0, "count": 0}

            opt.zero_grad()
            for batch_codes in dataloader:
                batch_codes = batch_codes.to(device)
                loss, text_loss, audio_loss = compute_loss(
                    model, batch_codes,
                    audio_noise_ratio=args.audio_noise_ratio,
                    text_noise_ratio=args.text_noise_ratio,
                    first_codebook_weight=args.first_codebook_weight,
                    early_codebook_weight=args.early_codebook_weight,
                    text_padding_weight=args.text_padding_weight,
                )
                (loss / args.grad_accum).backward()
                micro_step += 1
                epoch_loss += loss.item()

                accum_losses["total"] += loss.item()
                accum_losses["text"] += text_loss.item()
                accum_losses["audio"] += audio_loss.item()
                accum_losses["count"] += 1

                if micro_step % args.grad_accum == 0:
                    update_lr(step)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    ).item()
                    opt.step()
                    opt.zero_grad()
                    step += 1
                    epoch_steps += 1

                    n = accum_losses["count"]
                    log_step(
                        step,
                        accum_losses["total"] / n,
                        accum_losses["text"] / n,
                        accum_losses["audio"] / n,
                        grad_norm, epoch,
                    )
                    accum_losses = {"total": 0.0, "text": 0.0, "audio": 0.0, "count": 0}

                    # Phase 0 -> Phase 1 transition
                    maybe_transition_from_phase0(step, epoch)

                    if args.save_every and step % args.save_every == 0:
                        save_checkpoint(step)
                        save_training_state(step, epoch)
                    if args.eval_every > 0 and step % args.eval_every == 0 and eval_codes is not None:
                        run_eval(model, eval_codes, step, out_dir, mimi, tokenizer,
                                 max_ar_frames=args.eval_ar_frames)
                    if args.steps > 0 and step >= args.steps:
                        done = True
                        break

            avg_epoch_loss = epoch_loss / max(micro_step, 1)
            print(f"  Epoch {epoch} complete. Avg micro-step loss: {avg_epoch_loss:.4f} "
                  f"({epoch_steps} optimizer steps)\n")
            save_checkpoint(step, tag=f"epoch_{epoch}")
            save_training_state(step, epoch)

            if done:
                break

    if eval_codes is not None:
        run_eval(model, eval_codes, step, out_dir, mimi, tokenizer,
                 max_ar_frames=args.eval_ar_frames)
    save_checkpoint(step, tag="final")
    print(f"\nTraining complete. Best avg loss: {best_loss:.4f}")
    print(f"Total optimizer steps: {step}")


if __name__ == "__main__":
    main()