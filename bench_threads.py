"""
Measure what a SAC gradient update actually costs on this machine, as a
function of torch thread count and batch size.

Why this exists
---------------
Two decisions depend on this number and nothing in the project has ever
measured it:

  1. Wall-clock. With the replay-ratio fix (verify_replay_ratio.py) a run does
     ~4x more gradient updates, and updates dominate wall-clock (a physics step
     is ~7 us). The cost per update therefore sets the whole compute budget.
  2. Thermal headroom. The machine hard-locked on 2026-08-14 running four
     workers x 3 torch threads. For small MLPs ([256, 256]) torch often gets
     SLOWER above ~4 threads because synchronisation overhead exceeds the
     parallel gain — in which case fewer threads is both faster and cooler,
     and the parallelism that killed the machine was never buying anything.

Measures the real SAC.train() path on the real replay buffer, not a synthetic
matmul, so the number includes actor+critic+entropy updates and optimiser steps.

Run:  python bench_threads.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import config
from rl import algorithms as algo_registry
from rl.train import _make_vec_env


def bench(threads: int, batch_size: int, n_updates: int = 120,
          warmup: int = 30) -> float:
    """Median seconds per SAC gradient update at the given thread count."""
    torch.set_num_threads(threads)

    envs = _make_vec_env(seed=0, domain_rand=True, n_envs=1)
    model = algo_registry.build(
        "SAC", envs, seed=0, device="cpu",
        batch_size=batch_size, learning_starts=100, buffer_size=20_000,
    )
    # Fill the buffer so train() has real data to sample.
    model.learn(total_timesteps=1500, progress_bar=False)

    for _ in range(warmup):
        model.train(gradient_steps=1, batch_size=batch_size)

    times = []
    for _ in range(n_updates):
        t0 = time.perf_counter()
        model.train(gradient_steps=1, batch_size=batch_size)
        times.append(time.perf_counter() - t0)

    envs.close()
    return float(np.median(times))


def main():
    n_logical = os.cpu_count()
    print("=" * 74)
    print(f"SAC GRADIENT-UPDATE COST   ({n_logical} logical CPUs, "
          f"torch default {torch.get_num_threads()} threads)")
    print("=" * 74)

    batch_sizes = [256, 512]
    thread_counts = [1, 2, 4, 8]

    print(f"  {'threads':>8} " + "".join(f"{'bs=' + str(b):>14}" for b in batch_sizes))
    print("  " + "-" * 68)

    table = {}
    for th in thread_counts:
        row = []
        for bs in batch_sizes:
            ms = bench(th, bs) * 1e3
            table[(th, bs)] = ms
            row.append(f"{ms:>11.2f} ms")
        print(f"  {th:>8} " + "".join(f"{c:>14}" for c in row))

    print()
    print("=" * 74)
    print("PROJECTED WALL-CLOCK  (batch 512, replay ratio 0.933 after the fix)")
    print("=" * 74)
    best_th = min(thread_counts, key=lambda t: table[(t, 512)])
    for th in thread_counts:
        ms = table[(th, 512)]
        # 1e6 env steps at ratio 0.933 -> ~933k updates
        hours_1m = 933_000 * ms / 1e3 / 3600
        # a 300k-step run at the same ratio
        min_300k = 280_000 * ms / 1e3 / 60
        mark = "   <- fastest" if th == best_th else ""
        print(f"  {th} threads: {ms:6.2f} ms/update  ->  "
              f"300k steps = {min_300k:5.1f} min | 1e6 steps = {hours_1m:4.1f} h{mark}")

    print()
    slowest = max(thread_counts, key=lambda t: table[(t, 512)])
    speedup = table[(slowest, 512)] / table[(best_th, 512)]
    print(f"  Fastest thread count: {best_th}  "
          f"({speedup:.2f}x faster than {slowest} threads)")
    if best_th < 8:
        print(f"  => Using {best_th} threads instead of the default 8 is both faster")
        print(f"     AND leaves cores idle, cutting the sustained thermal load.")
    print("=" * 74)


if __name__ == "__main__":
    main()
