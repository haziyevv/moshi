#!/usr/bin/env python3
"""Diagnose the teacher-forced vs autoregressive gap in Qwen-Moshi.

This script runs multiple targeted tests to identify WHY autoregressive
generation produces garbled audio while teacher-forcing works.

Tests:
  1. Teacher-forced accuracy and logit entropy (per codebook)
  2. Greedy AR generation (temp=0) — is sampling the issue?
  3. Sampled AR generation (temp=0.8) — baseline comparison
  4. Semi-AR: feed ground-truth codebook 0, generate rest — is CB0 the bottleneck?
  5. Progressive degradation: measure token agreement over time

Usage::

    python scripts/diagnose_ar_gap.py \
        --qwen-weights runs/qwen_moshi_ft/checkpoint_final.safetensors \
        --sample-pt ./encoded_codes/000000.pt \
        --input-wav ./daily-talk-contiguous/data_stereo/0.wav
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moshi.models import get_qwen_moshi_lm, LMGen
from moshi.models.loaders import CheckpointInfo


def run_ar_generation(lm, lm_gen_kwargs, mimi, user_tensor, frame_size, n_frames):
    """Run autoregressive generation, return tokens and audio."""
    lm_gen = LMGen(lm, **lm_gen_kwargs)
    gen_audio = []
    gen_tokens_all = []  # list of [dep_q+1] per frame

    with torch.no_grad():
        with mimi.streaming(1), lm_gen.streaming(1):
            for i in range(n_frames):
                chunk = user_tensor[:, i * frame_size:(i + 1) * frame_size].unsqueeze(0)
                user_codes = mimi.encode(chunk)
                tokens = lm_gen.step(user_codes[:, :, :1])
                if tokens is None:
                    continue
                gen_audio.append(mimi.decode(tokens[:, 1:, :]))
                gen_tokens_all.append(tokens[0, :, 0].cpu())  # [dep_q+1]

    if gen_audio:
        gen_pcm = torch.cat(gen_audio, dim=-1)
    else:
        gen_pcm = None
    if gen_tokens_all:
        gen_tokens = torch.stack(gen_tokens_all, dim=1)  # [dep_q+1, n_generated]
    else:
        gen_tokens = None
    return gen_pcm, gen_tokens


def main():
    parser = argparse.ArgumentParser(description="Diagnose AR gap in Qwen-Moshi")
    parser.add_argument("--qwen-weights", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sample-pt", type=str, required=True,
                        help="A .pt file from encoded_codes/")
    parser.add_argument("--input-wav", type=str, default=None,
                        help="Original stereo WAV (right=user)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Max frames to process")
    args = parser.parse_args()

    config_path = args.config or str(REPO_ROOT / "configs" / "moshi_qwen_3b.json")

    # --- Load models ---
    print("Loading Mimi codec...")
    ckpt = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    mimi = ckpt.get_mimi(device=args.device)
    sample_rate = mimi.sample_rate
    frame_rate = mimi.frame_rate
    frame_size = int(sample_rate / frame_rate)

    print("Loading Qwen-Moshi LM...")
    lm = get_qwen_moshi_lm(
        qwen_weights=args.qwen_weights,
        config_path=config_path,
        device=args.device,
        dtype=torch.bfloat16,
    )

    # --- Load sample ---
    print(f"\nLoading sample: {args.sample_pt}")
    data = torch.load(args.sample_pt, map_location="cpu", weights_only=True)
    codes = data["codes"]  # [1, 17, T]
    T = min(codes.shape[-1], args.max_frames)
    codes = codes[:, :, :T].to(args.device)
    print(f"  Codes shape: {codes.shape}, using {T} frames ({T / frame_rate:.1f}s)")

    # =====================================================================
    # TEST 1: Teacher-forced analysis
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST 1: TEACHER-FORCED ANALYSIS")
    print("=" * 70)

    lm.eval()
    with torch.no_grad():
        out = lm(codes)

    # --- Text ---
    text_acc = 0.0
    if out.text_logits is not None and out.text_mask is not None:
        text_preds = out.text_logits.argmax(dim=-1)  # [B, 1, T]
        text_targets = codes[:, :1, :]
        valid = out.text_mask[0, 0]
        n_valid = valid.sum().item()

        if n_valid > 0:
            text_acc = (text_preds[0, 0, valid] == text_targets[0, 0, valid]).float().mean().item()

            # Non-padding text accuracy
            padding_id = lm.existing_text_padding_id
            non_pad = valid & (text_targets[0, 0] != padding_id)
            n_non_pad = non_pad.sum().item()
            if n_non_pad > 0:
                text_acc_nonpad = (text_preds[0, 0, non_pad] == text_targets[0, 0, non_pad]).float().mean().item()
            else:
                text_acc_nonpad = float('nan')

            # Text logit entropy
            text_probs = torch.softmax(out.text_logits[0, 0, valid].float(), dim=-1)
            text_entropy = -(text_probs * (text_probs + 1e-10).log()).sum(dim=-1).mean().item()

            print(f"\n  Text accuracy (all valid): {text_acc:.4f} ({n_valid} positions)")
            print(f"  Text accuracy (non-padding): {text_acc_nonpad:.4f} ({n_non_pad} positions)")
            print(f"  Text logit entropy: {text_entropy:.2f} (lower=more confident)")

    # --- Audio ---
    audio_acc_per_cb = []
    if out.logits is not None and out.mask is not None:
        audio_preds = out.logits.argmax(dim=-1)  # [B, dep_q, T]
        audio_targets = codes[:, lm.audio_offset:lm.audio_offset + lm.dep_q, :]

        print(f"\n  Per-codebook analysis:")
        print(f"  {'CB':>4s}  {'Accuracy':>8s}  {'Entropy':>8s}  {'Valid':>6s}")
        print(f"  {'----':>4s}  {'--------':>8s}  {'--------':>8s}  {'------':>6s}")

        for cb in range(lm.dep_q):
            cb_mask = out.mask[0, cb]
            if cb_mask.any():
                cb_acc = (audio_preds[0, cb, cb_mask] == audio_targets[0, cb, cb_mask]).float().mean().item()
                # Entropy
                cb_probs = torch.softmax(out.logits[0, cb, cb_mask].float(), dim=-1)
                cb_entropy = -(cb_probs * (cb_probs + 1e-10).log()).sum(dim=-1).mean().item()
                n_cb_valid = cb_mask.sum().item()
                print(f"  {cb:>4d}  {cb_acc:>8.4f}  {cb_entropy:>8.2f}  {n_cb_valid:>6d}")
                audio_acc_per_cb.append(cb_acc)
            else:
                audio_acc_per_cb.append(0.0)

        overall_audio_acc = np.mean(audio_acc_per_cb)
        print(f"\n  Overall audio accuracy: {overall_audio_acc:.4f}")
        print(f"  CB0 accuracy (most important): {audio_acc_per_cb[0]:.4f}")

    # --- Decode teacher-forced prediction ---
    try:
        import sphn
        pred_codes = audio_preds.detach().clamp(0, lm.card - 1)
        pred_pcm = mimi.decode(pred_codes)
        orig_codes = audio_targets.detach().clamp(0, lm.card - 1)
        orig_pcm = mimi.decode(orig_codes)

        sphn.write_wav("diag_teacher_pred.wav",
                       pred_pcm[0, 0].cpu().numpy().astype(np.float32), sample_rate)
        sphn.write_wav("diag_original.wav",
                       orig_pcm[0, 0].cpu().numpy().astype(np.float32), sample_rate)
        print(f"\n  Saved: diag_teacher_pred.wav (model predictions)")
        print(f"  Saved: diag_original.wav (ground truth)")
    except Exception as e:
        print(f"\n  Decode/save error: {e}")

    # =====================================================================
    # TEST 2 & 3: Autoregressive generation (greedy vs sampled)
    # =====================================================================
    if args.input_wav is None:
        print("\n  [Skipping AR tests — no --input-wav provided]")
        print("  Run with --input-wav to enable AR diagnostics.")
        return

    # Load user audio
    try:
        import sphn
        wav, wav_sr = sphn.read(args.input_wav)
    except Exception:
        import soundfile as sf
        wav, wav_sr = sf.read(args.input_wav, dtype="float32")
        wav = wav.T

    if wav_sr != sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.tensor(wav, dtype=torch.float32), wav_sr, sample_rate
        ).numpy()

    if wav.shape[0] >= 2:
        user_wav = wav[1:2]
    else:
        user_wav = wav[0:1]

    max_samples = T * frame_size
    user_wav = user_wav[:, :max_samples]
    user_tensor = torch.tensor(user_wav, dtype=torch.float32, device=args.device)
    n_frames = user_tensor.shape[1] // frame_size

    # --- TEST 2: Greedy AR ---
    print("\n" + "=" * 70)
    print("TEST 2: GREEDY AUTOREGRESSIVE (temp=0)")
    print("=" * 70)

    greedy_pcm, greedy_tokens = run_ar_generation(
        lm,
        dict(use_sampling=False, temp=0.0, temp_text=0.0, top_k=0, top_k_text=0),
        mimi, user_tensor, frame_size, n_frames,
    )

    if greedy_pcm is not None:
        rms = torch.sqrt(torch.mean(greedy_pcm ** 2)).item()
        peak = torch.max(torch.abs(greedy_pcm)).item()
        print(f"  Audio: RMS={rms:.6f}, Peak={peak:.6f}")
        sphn.write_wav("diag_ar_greedy.wav",
                       greedy_pcm[0, 0].cpu().numpy().astype(np.float32), sample_rate)
        print(f"  Saved: diag_ar_greedy.wav")

        # Token analysis
        audio_toks = greedy_tokens[1:]  # [dep_q, T]
        for cb in range(min(3, audio_toks.shape[0])):
            vals = audio_toks[cb].numpy()
            unique = len(set(vals.tolist()))
            print(f"  CB{cb}: unique={unique}/{len(vals)}, "
                  f"min={vals.min()}, max={vals.max()}, mean={vals.mean():.0f}")
    else:
        print("  No audio generated!")

    # --- TEST 3: Sampled AR ---
    print("\n" + "=" * 70)
    print("TEST 3: SAMPLED AUTOREGRESSIVE (temp=0.8, top_k=250)")
    print("=" * 70)

    sampled_pcm, sampled_tokens = run_ar_generation(
        lm,
        dict(use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25),
        mimi, user_tensor, frame_size, n_frames,
    )

    if sampled_pcm is not None:
        rms = torch.sqrt(torch.mean(sampled_pcm ** 2)).item()
        peak = torch.max(torch.abs(sampled_pcm)).item()
        print(f"  Audio: RMS={rms:.6f}, Peak={peak:.6f}")
        sphn.write_wav("diag_ar_sampled.wav",
                       sampled_pcm[0, 0].cpu().numpy().astype(np.float32), sample_rate)
        print(f"  Saved: diag_ar_sampled.wav")

    # --- TEST 4: Low-temp AR ---
    print("\n" + "=" * 70)
    print("TEST 4: LOW-TEMP AUTOREGRESSIVE (temp=0.3, top_k=50)")
    print("=" * 70)

    lowtemp_pcm, lowtemp_tokens = run_ar_generation(
        lm,
        dict(use_sampling=True, temp=0.3, temp_text=0.3, top_k=50, top_k_text=10),
        mimi, user_tensor, frame_size, n_frames,
    )

    if lowtemp_pcm is not None:
        rms = torch.sqrt(torch.mean(lowtemp_pcm ** 2)).item()
        peak = torch.max(torch.abs(lowtemp_pcm)).item()
        print(f"  Audio: RMS={rms:.6f}, Peak={peak:.6f}")
        sphn.write_wav("diag_ar_lowtemp.wav",
                       lowtemp_pcm[0, 0].cpu().numpy().astype(np.float32), sample_rate)
        print(f"  Saved: diag_ar_lowtemp.wav")

    # =====================================================================
    # TEST 5: Progressive degradation analysis
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST 5: PROGRESSIVE ERROR ANALYSIS")
    print("=" * 70)

    if greedy_tokens is not None and out.logits is not None:
        # Compare AR tokens to ground truth over time
        gt_audio = codes[0, lm.audio_offset:lm.audio_offset + lm.dep_q, :]  # [dep_q, T]

        ar_audio = greedy_tokens[1:]  # [dep_q, n_generated]
        max_delay = max(lm.delays)
        n_compare = min(ar_audio.shape[1], gt_audio.shape[1])

        # Split into chunks and measure agreement
        chunk_size = max(n_compare // 10, 1)
        print(f"\n  Token agreement (greedy AR vs ground truth) over time:")
        print(f"  {'Chunk':>8s}  {'CB0':>6s}  {'CB1':>6s}  {'CB2':>6s}  {'AllCB':>6s}")
        print(f"  {'--------':>8s}  {'------':>6s}  {'------':>6s}  {'------':>6s}  {'------':>6s}")

        for chunk_start in range(0, n_compare - chunk_size + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_compare)
            ar_chunk = ar_audio[:, chunk_start:chunk_end].to(args.device)
            # Offset GT by max_delay to align with AR output
            gt_offset = chunk_start + max_delay
            gt_end = gt_offset + (chunk_end - chunk_start)
            if gt_end > gt_audio.shape[1]:
                break
            gt_chunk = gt_audio[:, gt_offset:gt_end]

            accs = []
            for cb in range(min(3, lm.dep_q)):
                acc = (ar_chunk[cb] == gt_chunk[cb]).float().mean().item()
                accs.append(acc)
            all_acc = (ar_chunk == gt_chunk).float().mean().item()
            time_sec = chunk_start / frame_rate

            line = f"  {time_sec:>6.1f}s  "
            for acc in accs:
                line += f"{acc:>6.2%}  "
            line += f"{all_acc:>6.2%}"
            print(line)

    # =====================================================================
    # TEST 6: Check if text tokens diverge
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST 6: TEXT TOKEN ANALYSIS")
    print("=" * 70)

    if greedy_tokens is not None:
        text_tokens = greedy_tokens[0].tolist()
        padding_id = lm.existing_text_padding_id
        non_padding = [t for t in text_tokens if t != padding_id]
        print(f"  Total text tokens: {len(text_tokens)}")
        print(f"  Non-padding tokens: {len(non_padding)} ({100 * len(non_padding) / max(len(text_tokens), 1):.1f}%)")

        # Try decode
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
            if non_padding:
                decoded = tok.decode(non_padding, skip_special_tokens=True)
                print(f"  Decoded text (first 200 chars): {decoded[:200]}")
            else:
                print("  All text tokens are padding!")
        except Exception as e:
            print(f"  Could not decode: {e}")

        # Ground truth text for comparison
        gt_text = codes[0, 0, :].cpu().tolist()
        gt_non_padding = [t for t in gt_text if t != padding_id]
        if gt_non_padding:
            try:
                gt_decoded = tok.decode(gt_non_padding, skip_special_tokens=True)
                print(f"  Ground truth text:              {gt_decoded[:200]}")
            except Exception:
                pass

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    if audio_acc_per_cb:
        print(f"\n  Teacher-forced CB0 accuracy: {audio_acc_per_cb[0]:.2%}")
        print(f"  Teacher-forced overall accuracy: {np.mean(audio_acc_per_cb):.2%}")
        print(f"  Teacher-forced text accuracy: {text_acc:.2%}")

    print("\n  Generated WAV files for manual listening:")
    print("    diag_original.wav          - Ground truth audio")
    print("    diag_teacher_pred.wav      - Teacher-forced model predictions")
    print("    diag_ar_greedy.wav         - Greedy AR (temp=0)")
    print("    diag_ar_sampled.wav        - Sampled AR (temp=0.8)")
    print("    diag_ar_lowtemp.wav        - Low-temp AR (temp=0.3)")

    print("\n  Next steps based on results:")
    print("  - If greedy sounds MUCH better than sampled -> lower temperature")
    print("  - If all AR sounds bad -> error accumulation / exposure bias")
    print("  - If CB0 accuracy < 90% -> backbone needs more training")
    print("  - If CB0 accuracy > 95% but AR bad -> Depformer conditioning issue")
    print("  - If tokens show progressive degradation -> need noise injection or scheduled sampling")
    print("  - If text diverges -> text prediction feeding wrong context to Depformer")


if __name__ == "__main__":
    main()