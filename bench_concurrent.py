"""
Is running a CPU job and a GPU job CONCURRENTLY faster than the CPU alone?

The remaining experiments are independent training runs, so they are trivially
parallel. The question is not whether the GPU is faster per run -- it is not,
measured at 15.0 ms/update against the CPU's 9.68 -- but whether adding a GPU
job ALONGSIDE a CPU job increases total throughput.

Two things could defeat it:

  1. CPU CONTENTION. A "GPU" run is not free of the CPU. Every one of the ~570
     kernel launches per update is issued by Python on the host, so a GPU job
     occupies a CPU core continuously. It also steps MuJoCo, which is CPU-only.
     The two jobs therefore compete for the same cores.
  2. THERMAL HEADROOM. This machine hard-locked on 2026-08-14 under sustained
     CPU-only load -- no crash dump, no hardware error, the kernel unable to
     write its own event log. A discrete GPU adds substantial package power on
     top. Combined sustained load is a harsher condition than anything that has
     been tried here, including the condition that already caused a failure.

This measures solo throughput for each device, then both together, and reports
GPU temperature throughout.

Run:  python bench_concurrent.py
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

WORKER = r'''
import os, sys, time, json
sys.path.insert(0, r"{here}")
import torch
torch.set_num_threads({threads})
import config
from rl import algorithms as algo_registry
from rl.train import _make_vec_env

dev = "{device}"
envs = _make_vec_env(seed=0, domain_rand=True, n_envs=1)
model = algo_registry.build("SAC", envs, seed=0, device=dev,
                            batch_size=128, learning_starts=100,
                            buffer_size=20000)
model.learn(total_timesteps=1200, progress_bar=False)
for _ in range(20):
    model.train(gradient_steps=1, batch_size=128)
if dev == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter(); n = 0
while time.perf_counter() - t0 < {secs}:
    model.train(gradient_steps=1, batch_size=128)
    n += 1
if dev == "cuda":
    torch.cuda.synchronize()
el = time.perf_counter() - t0
print(json.dumps({{"device": dev, "updates": n, "sec": el,
                   "ms_per_update": el / n * 1000}}))
'''


def run_worker(device, threads, secs, wait=True):
    src = WORKER.format(here=HERE, device=device, threads=threads, secs=secs)
    f = os.path.join(HERE, f"_w_{device}_{threads}.py")
    open(f, "w").write(src)
    p = subprocess.Popen([sys.executable, f], cwd=HERE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True)
    if not wait:
        return p, f
    out, _ = p.communicate()
    os.remove(f)
    for line in out.splitlines():
        if line.strip().startswith("{"):
            return json.loads(line)
    return None


def gpu_temp():
    for exe in (r"C:\Windows\System32\nvidia-smi.exe", "nvidia-smi"):
        try:
            o = subprocess.run([exe, "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            if o.returncode == 0 and o.stdout.strip():
                t, w, u = [x.strip() for x in o.stdout.strip().split(",")]
                return float(t), float(w), float(u)
        except Exception:
            continue
    return None, None, None


def main():
    SECS = 45
    print("=" * 74)
    print("CONCURRENT CPU + GPU THROUGHPUT")
    print("=" * 74)
    t, w, u = gpu_temp()
    if t is not None:
        print(f"  GPU idle: {t:.0f} C, {w:.0f} W, {u:.0f}% util\n")

    print(f"  [1/3] CPU alone, 8 threads ({SECS}s) ...", flush=True)
    solo_cpu = run_worker("cpu", 8, SECS)
    print(f"        {solo_cpu['ms_per_update']:.2f} ms/update")

    print(f"  [2/3] GPU alone ({SECS}s) ...", flush=True)
    solo_gpu = run_worker("cuda", 1, SECS)
    t1, w1, u1 = gpu_temp()
    print(f"        {solo_gpu['ms_per_update']:.2f} ms/update"
          + (f"   GPU {t1:.0f} C, {w1:.0f} W, {u1:.0f}% util" if t1 else ""))

    print(f"  [3/3] BOTH concurrently ({SECS}s) ...", flush=True)
    pc, fc = run_worker("cpu", 7, SECS, wait=False)
    pg, fg = run_worker("cuda", 1, SECS, wait=False)
    temps = []
    t_end = time.time() + SECS + 25
    while time.time() < t_end and (pc.poll() is None or pg.poll() is None):
        tt, ww, uu = gpu_temp()
        if tt is not None:
            temps.append((tt, ww, uu))
        time.sleep(4)
    oc, _ = pc.communicate(); og, _ = pg.communicate()
    for f in (fc, fg):
        try: os.remove(f)
        except OSError: pass

    def parse(o):
        for line in o.splitlines():
            if line.strip().startswith("{"):
                return json.loads(line)
        return None
    both_cpu, both_gpu = parse(oc), parse(og)

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    if not (both_cpu and both_gpu):
        print("  a concurrent worker failed; cannot conclude")
        return

    solo_rate = 1000.0 / solo_cpu["ms_per_update"]
    both_rate = 1000.0 / both_cpu["ms_per_update"] + 1000.0 / both_gpu["ms_per_update"]
    print(f"  CPU alone            : {solo_cpu['ms_per_update']:6.2f} ms/update"
          f"  -> {solo_rate:6.1f} updates/s")
    print(f"  GPU alone            : {solo_gpu['ms_per_update']:6.2f} ms/update")
    print()
    print(f"  concurrent, CPU job  : {both_cpu['ms_per_update']:6.2f} ms/update"
          f"  ({both_cpu['ms_per_update']/solo_cpu['ms_per_update']:.2f}x slower than solo)")
    print(f"  concurrent, GPU job  : {both_gpu['ms_per_update']:6.2f} ms/update"
          f"  ({both_gpu['ms_per_update']/solo_gpu['ms_per_update']:.2f}x slower than solo)")
    print(f"  COMBINED throughput  : {both_rate:6.1f} updates/s")
    print(f"  vs CPU alone         : {both_rate/solo_rate:.2f}x")

    if temps:
        tm = max(x[0] for x in temps); wm = max(x[1] for x in temps)
        print(f"\n  GPU peak during test : {tm:.0f} C, {wm:.0f} W")

    print("\n" + "=" * 74)
    gain = both_rate / solo_rate
    if gain < 1.15:
        print(f"  NOT WORTH IT. {gain:.2f}x throughput does not justify running a")
        print("  discrete GPU at sustained load on a laptop that has already")
        print("  hard-locked once under CPU-only load. Keep the CPU-only path.")
    else:
        print(f"  {gain:.2f}x throughput. Worth considering IF thermals hold --")
        print("  judge against the peak temperature above and the hard-lock history.")


if __name__ == "__main__":
    main()
