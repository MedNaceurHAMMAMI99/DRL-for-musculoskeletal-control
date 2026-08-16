"""
Does the GPU actually help? Measure, do not assume.

The networks here are small ([256, 256], 393k parameters, batch 128). For
networks this size, GPU kernel-launch overhead can exceed the compute, and the
device ends up SLOWER than a CPU. Small-network reinforcement learning is
routinely CPU-bound for exactly this reason.

Against that: 14.4 ms per SAC update on 8 CPU threads is slow for six passes
through a two-layer MLP, which suggests the CPU path is overhead-dominated too.
A GTX 1660 Ti might do the same work in 1-3 ms.

Both arguments are plausible, so this times the real `SAC.train()` path -- actor,
twin critics, target networks, entropy coefficient, optimiser steps -- on each
device, at the batch size actually used.

It also reports the environment-stepping cost, which stays on the CPU whatever
happens and therefore bounds the achievable speed-up.

Run:  python bench_gpu.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import config
from rl import algorithms as algo_registry
from rl.train import _make_vec_env


def bench_device(device: str, threads: int, batch: int,
                 n_updates: int = 150, warmup: int = 30) -> float:
    torch.set_num_threads(threads)
    envs = _make_vec_env(seed=0, domain_rand=True, n_envs=1)
    model = algo_registry.build("SAC", envs, seed=0, device=device,
                                batch_size=batch, learning_starts=100,
                                buffer_size=20_000)
    model.learn(total_timesteps=1500, progress_bar=False)

    for _ in range(warmup):
        model.train(gradient_steps=1, batch_size=batch)
    if device == "cuda":
        torch.cuda.synchronize()

    ts = []
    for _ in range(n_updates):
        t0 = time.perf_counter()
        model.train(gradient_steps=1, batch_size=batch)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    envs.close()
    return float(np.median(ts)) * 1e3


def bench_env_step(n: int = 3000) -> float:
    """Environment stepping stays on CPU; it bounds any achievable speed-up."""
    from rl.environment import ArmReachEnv
    env = ArmReachEnv(domain_rand=True, seed=0)
    env.reset(seed=0)
    a = np.full(9, 0.1, dtype=np.float32)
    for _ in range(200):
        env.step(a)
    t0 = time.perf_counter()
    for i in range(n):
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset(seed=i)
    return (time.perf_counter() - t0) / n * 1e3


def main():
    print("=" * 74)
    print("GPU vs CPU — real SAC.train() path")
    print("=" * 74)
    print(f"  torch          : {torch.__version__}")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  device         : {torch.cuda.get_device_name(0)} "
              f"({p.total_memory/1e9:.1f} GB, {p.multi_processor_count} SMs)")
    print()

    batch = 128          # the tuned configuration's batch size
    rows = []

    cpu8 = bench_device("cpu", 8, batch)
    rows.append(("CPU, 8 threads", cpu8))
    print(f"  {'CPU, 8 threads':<26}{cpu8:>9.2f} ms/update")

    if torch.cuda.is_available():
        gpu = bench_device("cuda", 8, batch)
        rows.append(("GPU (CUDA)", gpu))
        print(f"  {'GPU (CUDA)':<26}{gpu:>9.2f} ms/update")

    env_ms = bench_env_step()
    print(f"\n  {'env step (CPU-bound)':<26}{env_ms:>9.3f} ms/step")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if not torch.cuda.is_available():
        print("  CUDA unavailable — nothing to compare.")
        return

    speedup = cpu8 / gpu
    # Per environment step the run performs ~1.87 gradient updates.
    ratio = 1.87
    cpu_total = ratio * cpu8 + env_ms
    gpu_total = ratio * gpu + env_ms
    print(f"  update speed-up            : {speedup:.2f}x")
    print(f"  per env step, CPU          : {cpu_total:.2f} ms")
    print(f"  per env step, GPU          : {gpu_total:.2f} ms")
    print(f"  END-TO-END speed-up        : {cpu_total/gpu_total:.2f}x")
    print(f"    (environment stepping is CPU-bound either way and caps this)")
    print()
    h_cpu = 300_000 * cpu_total / 1000 / 3600
    h_gpu = 300_000 * gpu_total / 1000 / 3600
    print(f"  a 300k-step run: {h_cpu:.2f} h on CPU  ->  {h_gpu:.2f} h on GPU")
    print()
    if cpu_total / gpu_total > 1.3:
        print("  GPU WINS decisively. Migrate the remaining stages to cuda.")
    elif cpu_total / gpu_total > 1.05:
        print("  GPU wins modestly. Worth migrating for the long stages only.")
    else:
        print("  NO MEANINGFUL GAIN. Keep the CPU path and revert torch to +cpu;")
        print("  the networks are too small for the GPU to pay for its overhead.")


if __name__ == "__main__":
    main()
