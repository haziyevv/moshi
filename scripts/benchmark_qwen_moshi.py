# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the Qwen-backed Moshi full pipeline: Mimi encode -> LM step -> Mimi decode.

Measures per-step latency, GPU time, real-time factor, and saves generated audio.

Usage::

    python scripts/benchmark_qwen_moshi.py --steps 100
    python scripts/benchmark_qwen_moshi.py --steps 200 --device cuda
"""

import argparse
import time

import numpy as np
import torch

from moshi.models import get_qwen_moshi_lm, LMGen
from moshi.models.loaders import CheckpointInfo

parser = argparse.ArgumentParser(description="Benchmark Qwen-backed Moshi pipeline.")
parser.add_argument(
    "--qwen-weights", type=str, default="/tmp/qwen_moshi_format.safetensors",
    help="Path to the converted Qwen safetensors weights.",
)
parser.add_argument(
    "--config", type=str,
    help="Path to the Moshi-Qwen config JSON.",
)
parser.add_argument("--steps", type=int, default=100, help="Number of streaming steps to run.")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--warmup", type=int, default=5, help="Warmup steps (excluded from timing).")
args = parser.parse_args()

torch.manual_seed(42)

# --- Load Mimi codec ---
print("Loading Mimi codec...")
ckpt = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
mimi = ckpt.get_mimi(device=args.device)
print(f"  Mimi: sample_rate={mimi.sample_rate}, frame_rate={mimi.frame_rate}")

# --- Load Qwen-backed Moshi ---
print("Loading Qwen-backed Moshi LM...")
config_path = args.config
if config_path is None:
    # Auto-detect config next to the script
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    config_path = str(repo_root / "configs" / "moshi_qwen_3b.json")

lm = get_qwen_moshi_lm(
    qwen_weights=args.qwen_weights,
    config_path=config_path,
    device=args.device,
    dtype=torch.bfloat16,
)
total_params = sum(p.numel() for p in lm.parameters())
print(f"  LM: {total_params / 1e9:.2f}B params, dim={lm.dim}, layers={len(lm.transformer.layers)}")

lm_gen = LMGen(lm, use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)
print("  LMGen created.")

# --- Benchmark ---
bs = 1
frame_size = int(mimi.sample_rate / mimi.frame_rate)  # 1920 samples per frame
frame_duration_ms = 1000.0 / mimi.frame_rate  # 80ms per frame

print(f"\nFrame size: {frame_size} samples ({frame_duration_ms:.0f}ms at {mimi.sample_rate}Hz)")
print(f"Running {args.steps} steps (+ {args.warmup} warmup)...\n")

main_audio = []
step_times = []
gpu_times = []
lm_times = []

def run_step(step_idx, record=True):
    start = time.time()

    # Simulate silence input
    chunk = torch.zeros((bs, 1, frame_size), dtype=torch.float, device=args.device)
    codes = mimi.encode(chunk)

    # LM step with GPU timing
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    lm_start = time.time()
    tokens = lm_gen.step(codes[:, :, :1])
    lm_dt = time.time() - lm_start
    ev_end.record()

    if tokens is None:
        print(f"  Step {step_idx:4d}: warming up (delay buffer)")
        return

    # Decode audio
    audio_tokens = tokens[:, 1:, :]
    main_pcm = mimi.decode(audio_tokens)
    main_audio.append(main_pcm[0].cpu())

    ev_end.synchronize()
    gpu_dt = ev_start.elapsed_time(ev_end)  # milliseconds
    total_dt = time.time() - start

    if record:
        step_times.append(total_dt * 1000)
        gpu_times.append(gpu_dt)
        lm_times.append(lm_dt * 1000)

    text_token = tokens[0, 0, 0].item()
    rtf = (total_dt * 1000) / frame_duration_ms

    print(
        f"  Step {step_idx:4d}: "
        f"total={total_dt * 1000:6.1f}ms  "
        f"lm={lm_dt * 1000:6.1f}ms  "
        f"gpu={gpu_dt:6.1f}ms  "
        f"RTF={rtf:.3f}  "
        f"text_tok={text_token}"
    )


with torch.no_grad():
    with mimi.streaming(bs), lm_gen.streaming(bs):
        # Warmup
        for step in range(args.warmup):
            run_step(step, record=False)

        # Timed run
        bench_start = time.time()
        for step in range(args.warmup, args.warmup + args.steps):
            run_step(step, record=True)
        bench_total = time.time() - bench_start

# --- Results ---
print("\n" + "=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

if step_times:
    step_arr = np.array(step_times)
    gpu_arr = np.array(gpu_times)
    lm_arr = np.array(lm_times)

    print(f"  Steps:              {len(step_times)}")
    print(f"  Frame duration:     {frame_duration_ms:.0f}ms")
    print(f"  Total wall time:    {bench_total:.2f}s")
    print()
    print(f"  Total step time:    mean={step_arr.mean():.1f}ms  "
          f"median={np.median(step_arr):.1f}ms  "
          f"p95={np.percentile(step_arr, 95):.1f}ms  "
          f"max={step_arr.max():.1f}ms")
    print(f"  LM step time:       mean={lm_arr.mean():.1f}ms  "
          f"median={np.median(lm_arr):.1f}ms  "
          f"p95={np.percentile(lm_arr, 95):.1f}ms")
    print(f"  GPU time:           mean={gpu_arr.mean():.1f}ms  "
          f"median={np.median(gpu_arr):.1f}ms  "
          f"p95={np.percentile(gpu_arr, 95):.1f}ms")
    print()

    rtf = step_arr.mean() / frame_duration_ms
    print(f"  Real-Time Factor:   {rtf:.3f}x  "
          f"({'REAL-TIME OK' if rtf < 1.0 else 'SLOWER THAN REAL-TIME'})")
    print(f"  Throughput:         {1000.0 / step_arr.mean():.1f} steps/sec")
    audio_duration = len(step_times) * frame_duration_ms / 1000.0
    print(f"  Audio generated:    {audio_duration:.1f}s in {bench_total:.1f}s")

# Save generated audio
if main_audio:
    try:
        import sphn
        audio_cat = torch.cat(main_audio, dim=-1)
        out_path = "qwen_moshi_output.wav"
        sphn.write_wav(out_path, audio_cat[0].numpy().astype(np.float32), mimi.sample_rate)
        print(f"\n  Output audio saved: {out_path} ({audio_cat.shape[-1] / mimi.sample_rate:.1f}s)")
    except ImportError:
        print("\n  (sphn not available, skipping audio save)")

print()
