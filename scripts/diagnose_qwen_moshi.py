# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Diagnose fine-tuned Qwen-Moshi: compare teacher-forced vs autoregressive inference.

This script loads a training sample, runs it through the model in two modes:
1. Teacher-forced (like training): feeds ground-truth codes, checks predictions
2. Autoregressive (like inference): feeds only user audio, generates response

By comparing these, we can identify if the issue is:
- (a) Model can predict teacher-forced but fails autoregressive -> exposure bias
- (b) Model can't predict teacher-forced either -> training or architecture bug
- (c) Predictions look good but Mimi decoding fails -> codec issue

Usage::

    python scripts/diagnose_qwen_moshi.py \\
        --qwen-weights runs/qwen_moshi_ft/checkpoint_epoch_35.safetensors \\
        --sample-pt ./encoded_codes/000000.pt \\
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


def main():
    parser = argparse.ArgumentParser(description="Diagnose Qwen-Moshi model quality")
    parser.add_argument("--qwen-weights", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sample-pt", type=str, required=True,
                        help="A .pt file from encoded_codes/ to use as the test sample")
    parser.add_argument("--input-wav", type=str, default=None,
                        help="Original WAV file (for autoregressive test)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Max frames to test (for speed)")
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
    print(f"  Codes shape: {codes.shape}")

    T = min(codes.shape[-1], args.max_frames)
    codes = codes[:, :, :T].to(args.device)
    print(f"  Using first {T} frames")

    # =========================================================================
    # TEST 1: Teacher-forced forward pass (like training)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: TEACHER-FORCED (like training)")
    print("=" * 70)

    lm.eval()
    with torch.no_grad():
        out = lm(codes)

    # Check text predictions
    if out.text_logits is not None and out.text_mask is not None:
        text_preds = out.text_logits.argmax(dim=-1)  # [B, 1, T]
        text_targets = codes[:, :1, :]
        valid_mask = out.text_mask[0, 0]  # [T]
        n_valid = valid_mask.sum().item()

        if n_valid > 0:
            text_acc = (text_preds[0, 0, valid_mask] == text_targets[0, 0, valid_mask]).float().mean().item()
            print(f"\n  Text accuracy (teacher-forced): {text_acc:.4f} ({n_valid} valid positions)")
            # Show some predictions vs targets
            valid_indices = valid_mask.nonzero(as_tuple=True)[0][:10]
            print("  First 10 valid text positions (target -> predicted):")
            for idx in valid_indices:
                t_tgt = text_targets[0, 0, idx].item()
                t_pred = text_preds[0, 0, idx].item()
                match = "OK" if t_tgt == t_pred else "MISS"
                print(f"    frame {idx.item():4d}: {t_tgt:6d} -> {t_pred:6d}  [{match}]")
        else:
            print("  No valid text positions in mask!")

    # Check audio predictions
    if out.logits is not None and out.mask is not None:
        audio_preds = out.logits.argmax(dim=-1)  # [B, dep_q, T]
        audio_targets = codes[:, lm.audio_offset:lm.audio_offset + lm.dep_q, :]
        valid_mask = out.mask[0]  # [dep_q, T]
        n_valid = valid_mask.sum().item()

        if n_valid > 0:
            audio_acc = (audio_preds[0][valid_mask[0:lm.dep_q]] == audio_targets[0][valid_mask[0:lm.dep_q]]).float().mean().item()
            # Per-codebook accuracy
            print(f"\n  Audio accuracy (teacher-forced, all codebooks): {audio_acc:.4f}")
            for cb in range(lm.dep_q):
                cb_mask = valid_mask[cb]
                if cb_mask.any():
                    cb_acc = (audio_preds[0, cb, cb_mask] == audio_targets[0, cb, cb_mask]).float().mean().item()
                    print(f"    Codebook {cb}: accuracy={cb_acc:.4f} ({cb_mask.sum().item()} valid)")

            # Decode teacher-forced audio predictions
            print("\n  Decoding teacher-forced audio predictions with Mimi...")
            pred_audio_codes = audio_preds.clamp(0, lm.card - 1)  # [1, dep_q, T]
            try:
                pred_pcm = mimi.decode(pred_audio_codes)
                pred_rms = torch.sqrt(torch.mean(pred_pcm ** 2)).item()
                pred_peak = torch.max(torch.abs(pred_pcm)).item()
                print(f"    Predicted audio: RMS={pred_rms:.6f}, Peak={pred_peak:.6f}")

                # Decode original main speaker audio for comparison
                orig_audio_codes = audio_targets.clamp(0, lm.card - 1)
                orig_pcm = mimi.decode(orig_audio_codes)
                orig_rms = torch.sqrt(torch.mean(orig_pcm ** 2)).item()
                orig_peak = torch.max(torch.abs(orig_pcm)).item()
                print(f"    Original audio:  RMS={orig_rms:.6f}, Peak={orig_peak:.6f}")

                # Save both
                import sphn
                sphn.write_wav("diag_teacher_forced_pred.wav",
                               pred_pcm[0, 0].cpu().numpy().astype(np.float32), sample_rate)
                sphn.write_wav("diag_original_main.wav",
                               orig_pcm[0, 0].cpu().numpy().astype(np.float32), sample_rate)
                print("    Saved: diag_teacher_forced_pred.wav, diag_original_main.wav")
            except Exception as e:
                print(f"    Mimi decode failed: {e}")
        else:
            print("  No valid audio positions in mask!")

    # =========================================================================
    # TEST 2: Autoregressive inference (like real usage)
    # =========================================================================
    if args.input_wav:
        print("\n" + "=" * 70)
        print("TEST 2: AUTOREGRESSIVE (like inference)")
        print("=" * 70)

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

        # Right channel = user
        if wav.shape[0] >= 2:
            user_wav = wav[1:2]
        else:
            user_wav = wav[0:1]

        max_samples = T * frame_size
        user_wav = user_wav[:, :max_samples]
        user_tensor = torch.tensor(user_wav, dtype=torch.float32, device=args.device)

        lm_gen = LMGen(lm, use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)
        gen_audio = []
        gen_text_tokens = []
        gen_audio_tokens = []

        with torch.no_grad():
            with mimi.streaming(1), lm_gen.streaming(1):
                n_frames = user_tensor.shape[1] // frame_size
                for i in range(n_frames):
                    chunk = user_tensor[:, i * frame_size:(i + 1) * frame_size].unsqueeze(0)
                    user_codes = mimi.encode(chunk)
                    tokens = lm_gen.step(user_codes[:, :, :1])
                    if tokens is None:
                        continue
                    gen_audio.append(mimi.decode(tokens[:, 1:, :]))
                    gen_text_tokens.append(tokens[0, 0, 0].item())
                    gen_audio_tokens.append(tokens[0, 1:, 0].cpu().tolist())

        if gen_audio:
            gen_pcm = torch.cat(gen_audio, dim=-1)
            gen_rms = torch.sqrt(torch.mean(gen_pcm ** 2)).item()
            gen_peak = torch.max(torch.abs(gen_pcm)).item()
            print(f"\n  Generated audio: RMS={gen_rms:.6f}, Peak={gen_peak:.6f}")
            sphn.write_wav("diag_autoregressive.wav",
                           gen_pcm[0, 0].cpu().numpy().astype(np.float32), sample_rate)
            print(f"  Saved: diag_autoregressive.wav")

            # Analyze generated tokens
            non_padding_text = [t for t in gen_text_tokens if t not in (0, 3, -1)]
            print(f"\n  Text tokens: {len(gen_text_tokens)} total, {len(non_padding_text)} non-padding")

            # Check audio token distribution
            all_audio_flat = [t for frame in gen_audio_tokens for t in frame]
            if all_audio_flat:
                audio_arr = np.array(all_audio_flat)
                print(f"  Audio tokens: min={audio_arr.min()}, max={audio_arr.max()}, "
                      f"mean={audio_arr.mean():.1f}, std={audio_arr.std():.1f}")
                # Check if tokens are concentrated (sign of degenerate model)
                unique = len(set(all_audio_flat))
                total = len(all_audio_flat)
                print(f"  Unique audio tokens: {unique}/{total} ({100 * unique / total:.1f}%)")

            # Compare first codebook tokens: auto vs teacher-forced
            if out.logits is not None:
                print("\n  Comparing codebook 0 tokens (first 20 frames):")
                tf_cb0 = audio_preds[0, 0, :20].cpu().tolist()
                ar_cb0 = [frame[0] for frame in gen_audio_tokens[:20]]
                gt_cb0 = codes[0, 1, :20].cpu().tolist()
                print(f"    Ground truth:     {gt_cb0}")
                print(f"    Teacher-forced:   {tf_cb0}")
                print(f"    Autoregressive:   {ar_cb0}")
        else:
            print("  No audio generated (all frames in warmup)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    if out.logits is not None and out.mask is not None:
        if audio_acc > 0.5:
            print("  Teacher-forced audio accuracy is HIGH (>{:.0f}%).".format(audio_acc * 100))
            print("  -> Model learned the training data well.")
            if args.input_wav and gen_audio:
                if gen_peak < 0.01:
                    print("  BUT autoregressive output is SILENCE.")
                    print("  -> Likely exposure bias or inference code issue.")
                elif gen_rms < orig_rms * 0.1:
                    print("  BUT autoregressive output is very quiet.")
                    print("  -> Audio token distribution may be degenerate.")
                else:
                    print("  Autoregressive output has audio content.")
                    print("  -> Quality issue may be in backbone audio understanding.")
        else:
            print("  Teacher-forced audio accuracy is LOW ({:.0f}%).".format(audio_acc * 100))
            print("  -> Model has NOT learned audio patterns well.")
            print("  -> Need more training data/epochs, or architecture issue.")


if __name__ == "__main__":
    main()
