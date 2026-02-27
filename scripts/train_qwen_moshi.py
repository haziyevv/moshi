#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Fine-tune the Qwen-backed Moshi model (Depformer + audio, optionally backbone).

Key design decisions (lessons learned from previous attempts):
  - Noise injection targets ONLY Moshi-side codebooks (CB0-7), not user audio,
    because user audio is always clean at inference.
  - Text noise is enabled by default (--text-noise-ratio 0.02) so the Depformer
    learns to handle imperfect text conditioning during AR.
  - First codebook weight defaults to 100 (matching official moshi-finetune).
  - Backbone LR default is 5e-5 (not 1e-5) so the backbone actually adapts.

Phase 1 (warm up Depformer, frozen backbone)::

    CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 scripts/train_qwen_moshi.py \\
        --qwen-weights /tmp/qwen_moshi_format.safetensors \\
        --data-dir ./encoded_codes_all \\
        --epochs 20 --batch-size 8 --seq-length 256 \\
        --grad-accum 4 --lr 3e-4 --warmup-steps 200 \\
        --freeze-backbone --out-dir runs/phase1

Phase 2 (joint training with noise injection)::

    CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 scripts/train_qwen_moshi.py \\
        --qwen-weights runs/phase1/checkpoint_final.safetensors \\
        --data-dir ./encoded_codes_all \\
        --epochs 50 --batch-size 4 --seq-length 256 \\
        --grad-accum 8 --lr 3e-4 --backbone-lr 5e-5 \\
        --warmup-steps 200 \\
        --audio-noise-ratio 0.10 --text-noise-ratio 0.02 \\
        --out-dir runs/phase2

Single GPU::

    python scripts/train_qwen_moshi.py --qwen-weights ... --data-dir ...
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from moshi.models import get_qwen_moshi_lm, LMGen
from moshi.models.loaders import CheckpointInfo
from moshi.utils.utils import cross_entropy


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module from a DDP/accelerate wrapper (or the module itself)."""
    return m.module if hasattr(m, "module") else m


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen-backed Moshi")
    p.add_argument("--qwen-weights", type=str, required=True,
                   help="Path to converted Qwen safetensors (or Phase 1 checkpoint)")
    p.add_argument("--config", type=str, default=None, help="Path to moshi_qwen_3b.json")
    p.add_argument("--mixed-precision", type=str, default="no",
                   choices=["no", "fp16", "bf16"],
                   help="Accelerate mixed-precision mode. Default 'no' since the model is "
                        "already loaded as bfloat16.")

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
                   help="Multiple data directories to combine (e.g. encoded_codes_all encoded_ami)")

    # Logging & checkpointing
    p.add_argument("--out-dir", type=str, default="runs/qwen_moshi_ft")
    p.add_argument("--save-every", type=int, default=0,
                   help="Save checkpoint every N steps (0 = save per epoch)")
    p.add_argument("--log-every", type=int, default=10)

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
    Can be built from one or multiple directories.
    """
    def __init__(self, data_dir: str | list[str], seq_length: int, zero_token_id: int = -1):
        if isinstance(data_dir, (list, tuple)):
            dirs = [Path(d) for d in data_dir]
            self.files = sorted(f for d in dirs for f in d.glob("*.pt"))
            if not self.files:
                raise FileNotFoundError(f"No .pt files found in {data_dir}")
            self.data_dir = None
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
    m = _unwrap(model)
    K = m.num_codebooks
    codes = torch.zeros(B, K, T, dtype=torch.long, device=device)
    codes[:, 0, :] = torch.randint(0, m.text_card, (B, T), device=device)
    for k in range(1, K):
        codes[:, k, :] = torch.randint(0, m.card, (B, T), device=device)
    pad_len = max(1, T // 10)
    codes[:, :, -pad_len:] = m.zero_token_id
    return codes


def inject_noise(codes, model, audio_noise_ratio=0.0, text_noise_ratio=0.0):
    """Replace a fraction of input codes with random values for exposure bias reduction.

    IMPORTANT: Audio noise is applied ONLY to Moshi-side codebooks (CB0-7 = channels
    1 through dep_q), NOT to user audio (channels dep_q+1 onward).  User audio comes
    from the microphone at inference and is always clean.
    """
    if audio_noise_ratio <= 0 and text_noise_ratio <= 0:
        return codes

    codes = codes.clone()
    B, K, T = codes.shape
    device = codes.device
    m = _unwrap(model)

    if audio_noise_ratio > 0:
        # Only corrupt Moshi-side codebooks: channels [audio_offset, audio_offset + dep_q)
        moshi_start = m.audio_offset
        moshi_end = m.audio_offset + m.dep_q
        moshi_codes = codes[:, moshi_start:moshi_end, :]  # [B, dep_q, T]
        mask = torch.rand(moshi_codes.shape, device=device) < audio_noise_ratio
        mask &= (moshi_codes != m.zero_token_id)
        random_codes = torch.randint(0, m.card, moshi_codes.shape, device=device)
        moshi_codes = torch.where(mask, random_codes, moshi_codes)
        codes[:, moshi_start:moshi_end, :] = moshi_codes

    if text_noise_ratio > 0:
        text_codes = codes[:, :1, :]  # [B, 1, T]
        mask = torch.rand(text_codes.shape, device=device) < text_noise_ratio
        mask &= (text_codes != m.zero_token_id)
        random_text = torch.randint(0, m.text_card, text_codes.shape, device=device)
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
    m = _unwrap(model)

    text_loss = torch.tensor(0.0, device=m.device, dtype=torch.float32)
    audio_loss = torch.tensor(0.0, device=m.device, dtype=torch.float32)

    # --- Text loss ---
    if out.text_logits is not None and out.text_mask is not None:
        text_targets = codes[:, :1, :]  # clean targets
        text_ce = cross_entropy(
            out.text_logits, text_targets, out.text_mask,
            dtype=torch.float32, logits_soft_clip=30.0,
        )
        if out.text_mask.any():
            text_weights = out.text_mask.float()
            if text_padding_weight != 1.0:
                padding_id = m.existing_text_padding_id
                end_padding_id = m.existing_text_end_padding_id
                is_padding = (text_targets == padding_id) | (text_targets == end_padding_id)
                text_weights = torch.where(
                    is_padding & out.text_mask,
                    text_weights * text_padding_weight,
                    text_weights,
                )
            weighted_ce = text_ce * text_weights
            text_loss = weighted_ce.sum() / text_weights.sum().clamp(min=1.0)

    # --- Audio loss ---
    if out.logits is not None and out.mask is not None:
        audio_targets = codes[:, m.audio_offset:m.audio_offset + m.dep_q, :]
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
    """Qualitative eval: teacher-forced accuracy/audio + AR generation + text decode.

    Produces WAV files and prints decoded text so you can hear and read the
    model's progress during training.
    """
    import sphn

    m = _unwrap(model)
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
            non_pad = valid & (text_targets[0, 0] != m.existing_text_padding_id)
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
        audio_targets = codes[:, m.audio_offset:m.audio_offset + m.dep_q, :]
        for cb in range(m.dep_q):
            cb_mask = out.mask[0, cb]
            if cb_mask.any():
                cb_accs.append((audio_preds[0, cb, cb_mask] == audio_targets[0, cb, cb_mask])
                               .float().mean().item())
            else:
                cb_accs.append(0.0)
        print(f"  Audio acc: CB0 {cb_accs[0]:.1%} | Overall {np.mean(cb_accs):.1%}")

        pred_clamped = audio_preds.clamp(0, m.card - 1)
        pred_pcm = mimi.decode(pred_clamped)
        sphn.write_wav(str(eval_dir / f"step{step_num:06d}_teacher.wav"),
                       pred_pcm[0, 0].cpu().float().numpy(), sr)

        gt_path = eval_dir / "ground_truth_moshi.wav"
        if not gt_path.exists():
            gt_clamped = audio_targets.clamp(0, m.card - 1)
            gt_pcm = mimi.decode(gt_clamped)
            sphn.write_wav(str(gt_path), gt_pcm[0, 0].cpu().float().numpy(), sr)

            user_codes_gt = codes[:, m.audio_offset + m.dep_q:, :]
            user_clamped = user_codes_gt.clamp(0, m.card - 1)
            user_pcm = mimi.decode(user_clamped)
            sphn.write_wav(str(eval_dir / "ground_truth_user.wav"),
                           user_pcm[0, 0].cpu().float().numpy(), sr)
            print(f"  Saved ground truth WAVs (one-time)")

    # ---- AR generation ----
    user_codes_ar = codes[:, m.audio_offset + m.dep_q:, :]  # [1, 8, T]
    lm_gen = LMGen(m, use_sampling=True, temp=0.8, temp_text=0.7,
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

        non_pad = [t for t in gen_text_tokens if t != m.existing_text_padding_id]
        if tokenizer is not None and non_pad:
            ar_text = tokenizer.decode(non_pad, skip_special_tokens=True)
            print(f"  AR text:   {ar_text[:120]}")
        print(f"  AR audio:  RMS={rms:.5f}, {len(gen_audio_chunks)} frames")

    print(f"  Saved: step{step_num:06d}_teacher.wav, step{step_num:06d}_ar.wav")
    print(f"{'=' * 70}\n")

    if was_training:
        model.train()


def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze the temporal transformer backbone and text embedding/projection."""
    for name, param in model.named_parameters():
        if any(name.startswith(pfx) for pfx in
               ("transformer.", "text_emb.", "text_linear.", "out_norm.")):
            param.requires_grad = False


def main():
    args = parse_args()
    config_path = args.config or str(REPO_ROOT / "configs" / "moshi_qwen_3b.json")

    # Accelerate handles distributed init, device placement, and mixed precision.
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    is_main = accelerator.is_main_process
    device = accelerator.device

    if is_main:
        print("=" * 70)
        print("TRAINING CONFIG")
        print("=" * 70)
        for k, v in sorted(vars(args).items()):
            print(f"  {k:24s}: {v}")
        print(f"  {'num_processes':24s}: {accelerator.num_processes}")
        print(f"  {'device':24s}: {device}")
        print("=" * 70)

    # ---- Load model ----
    accelerator.print("\nLoading Qwen-backed Moshi...")
    model = get_qwen_moshi_lm(
        qwen_weights=args.qwen_weights,
        config_path=config_path,
        device=device,
        dtype=torch.bfloat16,
    )
    model.train()

    if args.freeze_backbone:
        freeze_backbone(model)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Parameters: {n_trainable / 1e6:.1f}M trainable / {n_total / 1e6:.1f}M total"
                      + (" (backbone frozen)" if args.freeze_backbone else ""))

    # ---- Optimizer ----
    # Build optimizer BEFORE accelerator.prepare() so param names have no "module." prefix.
    backbone_prefixes = ("transformer.", "text_emb.", "text_linear.", "out_norm.")
    use_differential_lr = args.backbone_lr > 0 and not args.freeze_backbone

    if use_differential_lr:
        backbone_params, other_params = [], []
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
        n_bb = sum(p.numel() for p in backbone_params)
        n_other = sum(p.numel() for p in other_params)
        accelerator.print(f"  Differential LR: backbone ({n_bb / 1e6:.0f}M) @ {args.backbone_lr}, "
                          f"Depformer+audio ({n_other / 1e6:.0f}M) @ {args.lr}")
    else:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
        )

    # ---- Data ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir_or_dirs = args.data_dirs if args.data_dirs else args.data_dir
    if not args.synthetic and data_dir_or_dirs is None:
        accelerator.print("No --data-dir / --data-dirs provided, falling back to --synthetic mode.")
        args.synthetic = True

    if args.synthetic:
        model, opt = accelerator.prepare(model, opt)
        total_steps = args.steps if args.steps > 0 else 500
        accelerator.print(f"\nSynthetic mode: {total_steps} steps")
    else:
        dataset = PreEncodedDataset(data_dir_or_dirs, args.seq_length,
                                    zero_token_id=_unwrap(model).zero_token_id)
        # Single DataLoader path — accelerate.prepare() adds DistributedSampler automatically.
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
        )
        model, opt, dataloader = accelerator.prepare(model, opt, dataloader)

        # len(dataloader) is per-process after prepare.
        batches_per_epoch = len(dataloader)
        steps_per_epoch = batches_per_epoch // args.grad_accum
        total_steps = steps_per_epoch * args.epochs
        if args.steps > 0:
            total_steps = min(total_steps, args.steps)
        eff_batch = args.batch_size * args.grad_accum * accelerator.num_processes
        accelerator.print(
            f"\nReal data: {len(dataset)} samples, {batches_per_epoch} batches/epoch/GPU\n"
            f"Effective batch size: {args.batch_size} x {args.grad_accum} accum "
            f"x {accelerator.num_processes} GPUs = {eff_batch}\n"
            f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, Epochs: {args.epochs}\n"
            f"LR: warmup {args.warmup_steps} -> peak {args.lr} -> cosine to {args.min_lr}"
        )

    # ---- Eval setup (lazy: only loads Mimi/tokenizer if --eval-every > 0) ----
    mimi = None
    tokenizer = None
    eval_codes = None

    if args.eval_every > 0 and not args.synthetic and is_main:
        accelerator.print("\nLoading Mimi codec for eval...")
        ckpt = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        mimi = ckpt.get_mimi(device=device)
        mimi.eval()

        accelerator.print("Loading Qwen tokenizer for eval...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
        eval_path = dataset.files[min(args.eval_sample, len(dataset.files) - 1)]
        eval_data = torch.load(str(eval_path), map_location="cpu", weights_only=True)
        eval_codes = eval_data["codes"]  # [1, K, T] — full length, run_eval will crop
        accelerator.print(f"Eval sample: {eval_path.name} ({eval_codes.shape[-1]} frames = "
                          f"{eval_codes.shape[-1] / 12.5:.1f}s), eval every {args.eval_every} steps")

    # ---- Training loop ----
    step = 0
    best_loss = float("inf")
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
            if is_main:
                print(
                    f"  {ep_str}step {step_num:6d}/{total_steps}  "
                    f"loss {avg_t:.4f} (text {avg_text:.4f} + audio {avg_audio:.4f})  "
                    f"gnorm {grad_norm:.3f}  lr {current_lr:.2e}  dt {elapsed:.1f}s"
                )
            if avg_t < best_loss:
                best_loss = avg_t
            log_losses = {"total": 0.0, "text": 0.0, "audio": 0.0, "count": 0}
            t0 = time.time()

    def maybe_save(step_num, tag=None):
        if not is_main:
            return
        path = out_dir / (f"checkpoint_{tag}.safetensors" if tag
                          else f"checkpoint_step_{step_num}.safetensors")
        from safetensors.torch import save_file
        save_file(accelerator.unwrap_model(model).state_dict(), str(path))
        print(f"  -> Saved {path}")

    def update_lr(step_num):
        for pg in opt.param_groups:
            if use_differential_lr and pg.get("label") == "backbone":
                min_bb_lr = max(args.min_lr * (args.backbone_lr / args.lr), 1e-6)
                pg["lr"] = get_lr(step_num, args.warmup_steps, total_steps,
                                  args.backbone_lr, min_bb_lr)
            else:
                pg["lr"] = get_lr(step_num, args.warmup_steps, total_steps,
                                  args.lr, args.min_lr)

    accelerator.print(f"\nStarting training...\n")

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
            accelerator.backward(loss)
            grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm))
            opt.step()
            opt.zero_grad()
            step += 1
            log_step(step, loss.item(), text_loss.item(), audio_loss.item(), grad_norm)
            if args.save_every and step % args.save_every == 0:
                maybe_save(step)
    else:
        done = False
        grad_norm = 0.0

        for epoch in range(1, args.epochs + 1):
            if done:
                break
            if is_main:
                print(f"--- Epoch {epoch}/{args.epochs} ---")
            epoch_loss = 0.0
            epoch_steps = 0
            micro_step = 0

            opt.zero_grad()
            for batch_codes in dataloader:
                loss, text_loss, audio_loss = compute_loss(
                    model, batch_codes,
                    audio_noise_ratio=args.audio_noise_ratio,
                    text_noise_ratio=args.text_noise_ratio,
                    first_codebook_weight=args.first_codebook_weight,
                    early_codebook_weight=args.early_codebook_weight,
                    text_padding_weight=args.text_padding_weight,
                )
                # Scale loss for gradient accumulation; accelerator.backward handles AMP.
                accelerator.backward(loss / args.grad_accum)
                micro_step += 1
                epoch_loss += loss.item()

                if micro_step % args.grad_accum == 0:
                    update_lr(step)
                    grad_norm = float(
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    )
                    opt.step()
                    opt.zero_grad()
                    step += 1
                    epoch_steps += 1

                    log_step(step, loss.item(), text_loss.item(), audio_loss.item(),
                             grad_norm, epoch)

                    if args.save_every and step % args.save_every == 0:
                        maybe_save(step)
                    if args.eval_every > 0 and step % args.eval_every == 0 and eval_codes is not None:
                        run_eval(model, eval_codes, step, out_dir, mimi, tokenizer,
                                 max_ar_frames=args.eval_ar_frames)
                    if args.steps > 0 and step >= args.steps:
                        done = True
                        break

            avg_epoch_loss = epoch_loss / max(micro_step, 1)
            if is_main:
                print(f"  Epoch {epoch} complete. Avg loss: {avg_epoch_loss:.4f} "
                      f"({epoch_steps} optimizer steps)\n")
            maybe_save(step, tag=f"epoch_{epoch}")

            if done:
                break

    if eval_codes is not None:
        run_eval(model, eval_codes, step, out_dir, mimi, tokenizer,
                 max_ar_frames=args.eval_ar_frames)
    maybe_save(step, tag="final")
    if is_main:
        print(f"\nTraining complete. Best avg loss: {best_loss:.4f}")
        print(f"Total optimizer steps: {step}")


if __name__ == "__main__":
    main()
