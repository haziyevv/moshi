#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Diagnose a trained Qwen-Moshi checkpoint: teacher-forced accuracy, AR generation,
progressive error analysis, and text token quality.

Generates multiple WAV files for manual listening comparison.

Usage::

    python scripts/diagnose_qwen_moshi.py \\
        --qwen-weights runs/phase2/checkpoint_final.safetensors \\
        --sample-pt ./encoded_codes_all/000000.pt \\
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
    gen_tokens_all = []

    with torch.no_grad():
        with mimi.streaming(1), lm_gen.streaming(1):
            for i in range(n_frames):
                chunk = user_tensor[:, i * frame_size:(i + 1) * frame_size].unsqueeze(0)
                user_codes = mimi.encode(chunk)
                tokens = lm_gen.step(user_codes[:, :, :1])
                if tokens is None:
                    continue
                gen_audio.append(mimi.decode(tokens[:, 1:, :]))
                gen_tokens_all.append(tokens[0, :, 0].cpu())

    if gen_audio:
        gen_pcm = torch.cat(gen_audio, dim=-1)
    else:
        gen_pcm = None
    if gen_tokens_all:
        gen_tokens = torch.stack(gen_tokens_all, dim=1)
    else:
        gen_tokens = None
    return gen_pcm, gen_tokens


def main():
    parser = argparse.ArgumentParser(description="Diagnose Qwen-Moshi checkpoint")
    parser.add_argument("--qwen-weights", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sample-pt", type=str, required=True,
                        help="A .pt file from encoded_codes/")
    parser.add_argument("--input-wav", type=str, default=None,
                        help="Original stereo WAV (right=user)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-frames", type=int, default=200)
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
    codes = data["codes"]
    T = min(codes.shape[-1], args.max_frames)
    codes = codes[:, :, :T].to(args.device)
    print(f"  Codes shape: {codes.shape}, using {T} frames ({T / frame_rate:.1f}s)")

    # =================================================================
    # TEST 1: Teacher-forced accuracy
    # =================================================================
    print("\n" + "=" * 70)
    print("TEST 1: TEACHER-FORCED ANALYSIS")
    print("=" * 70)

    lm.eval()
    with torch.no_grad():
        out = lm(codes)

    text_acc = 0.0
    if out.text_logits is not None and out.text_mask is not None:
        text_preds = out.text_logits.argmax(dim=-1)
        text_targets = codes[:, :1, :]
        valid = out.text_mask[0, 0]
        n_valid = valid.sum().item()

        if n_valid > 0:
            text_acc = (text_preds[0, 0, valid] == text_targets[0, 0, valid]).float().mean().item()
            padding_id = lm.existing_text_padding_id
            non_pad = valid & (text_targets[0, 0] != padding_id)
            n_non_pad = non_pad.sum().item()
            if n_non_pad > 0:
                text_acc_nonpad = (text_preds[0, 0, non_pad] == text_targets[0, 0, non_pad]).float().mean().item()
            else:
                text_acc_nonpad = float('nan')
            text_probs = torch.softmax(out.text_logits[0, 0, valid].float(), dim=-1)
            text_entropy = -(text_probs * (text_probs + 1e-10).log()).sum(dim=-1).mean().item()
            print(f"\n  Text accuracy (all valid): {text_acc:.4f} ({n_valid} positions)")
            print(f"  Text accuracy (non-padding): {text_acc_nonpad:.4f} ({n_non_pad} positions)")
            print(f"  Text logit entropy: {text_entropy:.2f} (lower=more confident)")

    audio_acc_per_cb = []
    if out.logits is not None and out.mask is not None:
        audio_preds = out.logits.argmax(dim=-1)
        audio_targets = codes[:, lm.audio_offset:lm.audio_offset + lm.dep_q, :]

        print(f"\n  Per-codebook analysis:")
        print(f"  {'CB':>4s}  {'Accuracy':>8s}  {'Entropy':>8s}  {'Valid':>6s}")
        print(f"  {'----':>4s}  {'--------':>8s}  {'--------':>8s}  {'------':>6s}")

        for cb in range(lm.dep_q):
            cb_mask = out.mask[0, cb]
            if cb_mask.any():
                cb_acc = (audio_preds[0, cb, cb_mask] == audio_targets[0, cb, cb_mask]).float().mean().item()
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

    # Decode teacher-forced predictions
    try:
        import sphn
        pred_codes = audio_preds.detach().clamp(0, lm.card - 1)
        pred_pcm = mimi.decode(pred_codes)
        orig_codes = audio_targets.detach().clamp(0, lm.card - 1)
        orig_pcm = mimi.decode(orig_codes)

        sphn.write_wav("diag_teacher_pred.wav",
                       pred_pcm[0, 0].detach().cpu().numpy().astype(np.float32), sample_rate)
        sphn.write_wav("diag_original.wav",
                       orig_pcm[0, 0].detach().cpu().numpy().astype(np.float32), sample_rate)
        print(f"\n  Saved: diag_teacher_pred.wav, diag_original.wav")
    except Exception as e:
        print(f"\n  Decode/save error: {e}")

    # =================================================================
    # TEST 2-4: AR generation at different temperatures
    # =================================================================
    if args.input_wav is None:
        print("\n  [Skipping AR tests — no --input-wav provided]")
        _print_summary(audio_acc_per_cb, text_acc)
        return

    import sphn
    wav, wav_sr = sphn.read(args.input_wav)

    if wav_sr != sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.tensor(wav, dtype=torch.float32), wav_sr, sample_rate
        ).numpy()

    user_wav = wav[1:2] if wav.shape[0] >= 2 else wav[0:1]
    max_samples = T * frame_size
    user_wav = user_wav[:, :max_samples]
    user_tensor = torch.tensor(user_wav, dtype=torch.float32, device=args.device)
    n_frames = user_tensor.shape[1] // frame_size

    ar_configs = [
        ("GREEDY (temp=0)", dict(use_sampling=False, temp=0.0, temp_text=0.0, top_k=0, top_k_text=0), "diag_ar_greedy.wav"),
        ("SAMPLED (temp=0.8, top_k=250)", dict(use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25), "diag_ar_sampled.wav"),
        ("LOW-TEMP (temp=0.3, top_k=50)", dict(use_sampling=True, temp=0.3, temp_text=0.3, top_k=50, top_k_text=10), "diag_ar_lowtemp.wav"),
    ]

    greedy_tokens = None
    for i, (label, kwargs, filename) in enumerate(ar_configs):
        print(f"\n{'=' * 70}")
        print(f"TEST {i + 2}: AUTOREGRESSIVE {label}")
        print("=" * 70)

        pcm, tokens = run_ar_generation(lm, kwargs, mimi, user_tensor, frame_size, n_frames)
        if i == 0:
            greedy_tokens = tokens

        if pcm is not None:
            rms = torch.sqrt(torch.mean(pcm ** 2)).item()
            peak = torch.max(torch.abs(pcm)).item()
            print(f"  Audio: RMS={rms:.6f}, Peak={peak:.6f}")
            sphn.write_wav(filename, pcm[0, 0].cpu().numpy().astype(np.float32), sample_rate)
            print(f"  Saved: {filename}")

            if i == 0 and tokens is not None:
                audio_toks = tokens[1:]
                for cb in range(min(3, audio_toks.shape[0])):
                    vals = audio_toks[cb].numpy()
                    unique = len(set(vals.tolist()))
                    print(f"  CB{cb}: unique={unique}/{len(vals)}, "
                          f"min={vals.min()}, max={vals.max()}, mean={vals.mean():.0f}")

    # =================================================================
    # TEST 5: Progressive error analysis
    # =================================================================
    print(f"\n{'=' * 70}")
    print("TEST 5: PROGRESSIVE ERROR ANALYSIS")
    print("=" * 70)

    if greedy_tokens is not None:
        gt_audio = codes[0, lm.audio_offset:lm.audio_offset + lm.dep_q, :]
        ar_audio = greedy_tokens[1:]
        max_delay = max(lm.delays)
        n_compare = min(ar_audio.shape[1], gt_audio.shape[1])
        chunk_size = max(n_compare // 10, 1)

        print(f"\n  Token agreement (greedy AR vs ground truth) over time:")
        print(f"  {'Chunk':>8s}  {'CB0':>6s}  {'CB1':>6s}  {'CB2':>6s}  {'AllCB':>6s}")
        print(f"  {'--------':>8s}  {'------':>6s}  {'------':>6s}  {'------':>6s}  {'------':>6s}")

        for chunk_start in range(0, n_compare - chunk_size + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_compare)
            ar_chunk = ar_audio[:, chunk_start:chunk_end].to(args.device)
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

    # =================================================================
    # TEST 6: Text token analysis
    # =================================================================
    print(f"\n{'=' * 70}")
    print("TEST 6: TEXT TOKEN ANALYSIS")
    print("=" * 70)

    if greedy_tokens is not None:
        text_tokens = greedy_tokens[0].tolist()
        padding_id = lm.existing_text_padding_id
        non_padding = [t for t in text_tokens if t != padding_id]
        print(f"  Total text tokens: {len(text_tokens)}")
        print(f"  Non-padding tokens: {len(non_padding)} "
              f"({100 * len(non_padding) / max(len(text_tokens), 1):.1f}%)")

        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
            if non_padding:
                decoded = tok.decode(non_padding, skip_special_tokens=True)
                print(f"  Decoded text (first 200 chars): {decoded[:200]}")
        except Exception as e:
            print(f"  Could not decode: {e}")

        gt_text = codes[0, 0, :].cpu().tolist()
        gt_non_padding = [t for t in gt_text if t != padding_id]
        if gt_non_padding:
            try:
                gt_decoded = tok.decode(gt_non_padding, skip_special_tokens=True)
                print(f"  Ground truth text:              {gt_decoded[:200]}")
            except Exception:
                pass

    _print_summary(audio_acc_per_cb, text_acc)


def _print_summary(audio_acc_per_cb, text_acc):
    print(f"\n{'=' * 70}")
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    if audio_acc_per_cb:
        print(f"\n  Teacher-forced CB0 accuracy: {audio_acc_per_cb[0]:.2%}")
        print(f"  Teacher-forced overall accuracy: {np.mean(audio_acc_per_cb):.2%}")
        print(f"  Teacher-forced text accuracy: {text_acc:.2%}")
    print("\n  Generated WAV files for manual listening:")
    print("    diag_original.wav          - Ground truth audio")
    print("    diag_teacher_pred.wav      - Teacher-forced predictions")
    print("    diag_ar_greedy.wav         - Greedy AR (temp=0)")
    print("    diag_ar_sampled.wav        - Sampled AR (temp=0.8)")
    print("    diag_ar_lowtemp.wav        - Low-temp AR (temp=0.3)")
    print("\n  Interpretation guide:")
    print("  - CB0 accuracy < 90% -> backbone needs more training")
    print("  - CB0 > 95% but AR bad -> Depformer conditioning / exposure bias")
    print("  - Text non-padding < 80% -> backbone LR too low")
    print("  - Progressive degradation -> increase noise injection")
    print("  - Garbled text -> add --text-noise-ratio in training")


if __name__ == "__main__":
    main()
