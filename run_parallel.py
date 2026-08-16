"""
Run the remaining training jobs across CPU and GPU concurrently.

Measured rationale (bench_concurrent.py, this machine):

    CPU alone,  8 threads     10.04 ms/update    99.6 updates/s
    GPU alone                 16.89 ms/update
    concurrent CPU (7 thr)    11.80 ms/update    (1.18x slower than solo)
    concurrent GPU            17.41 ms/update    (1.03x slower than solo)
    combined                                    142.2 updates/s  = 1.43x

The GPU is slower per run -- the networks are far too small for it, and kernel
launch overhead dominates (26 us per operation against the CPU's 7 us). But the
remaining jobs are independent, so the GPU's spare capacity is free throughput
rather than a replacement.

Thermal position. The concurrent configuration uses 7 CPU threads plus roughly
one core issuing GPU launches, i.e. about 8 in total -- the same count that has
run safely for days. The GPU draws 30 W and peaks at 61 C, well inside its
envelope, because it sits mostly idle waiting for launches. This machine
hard-locked on 2026-08-14 under sustained CPU-only load at ~12 threads, so the
thread count is held at 8 and the GPU temperature is logged every job. If it
exceeds ABORT_TEMP the GPU worker stops and the CPU worker continues alone.

Jobs are claimed atomically through marker files, so the two workers
self-balance: whichever finishes first takes the next job, and neither can take
the same job twice. Safe to kill and restart -- claimed-but-unfinished jobs are
released on restart.

Run:  python run_parallel.py            (spawns both workers)
      python run_parallel.py --worker cpu   (internal)
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config

HERE      = os.path.dirname(os.path.abspath(__file__))
CLAIM_DIR = os.path.join(config.RUNS_DIR, "_claims")
RESULTS   = os.path.join(config.RUNS_DIR, "parallel_results.json")
ABORT_TEMP = 84.0          # deg C; GPU worker stops above this


# ── job list ───────────────────────────────────────────────────────────────
def build_jobs():
    """Every remaining independent training run, longest first.

    Longest-first matters for load balancing: short jobs left at the end let
    both workers finish together instead of one idling through a long tail.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = os.path.join(config.RUNS_DIR, "optuna_SAC.db")
    tuned = dict(optuna.load_study(study_name="SAC_arm_reach",
                                   storage=f"sqlite:///{db}").best_trial.params)
    tuned["gradient_steps"] = 2 * config.N_ENVS
    base = {"gradient_steps": 2 * config.N_ENVS}

    jobs = []
    # Stage 5 (long): replicate the w2 sweep across seeds.
    for w2 in (5.0, 20.0):
        for seed in (1, 2, 3, 4):
            w = dict(config.REWARD_WEIGHTS); w["w2"] = w2
            jobs.append({"id": f"w2_{w2:g}_seed{seed}", "steps": 300_000,
                         "seed": seed, "hyperparams": tuned, "reward_weights": w,
                         "out": os.path.join(config.RUNS_DIR, "w2_replication",
                                             f"w2_{w2:g}_seed{seed}"),
                         "group": "replicate_w2"})
    # Stage 4 (short): which hyperparameter actually caused the improvement?
    for arm, key in (("lr_only", "learning_rate"), ("tau_only", "tau"),
                     ("gamma_only", "gamma"), ("batch_only", "batch_size")):
        for seed in (0, 1, 2):
            jobs.append({"id": f"{arm}_seed{seed}", "steps": 100_000,
                         "seed": seed, "hyperparams": {**base, key: tuned[key]},
                         "reward_weights": None,
                         "out": os.path.join(config.RUNS_DIR, "ablation2",
                                             f"{arm}_seed{seed}"),
                         "group": "ablation_rest"})
    return jobs


# ── atomic claim ───────────────────────────────────────────────────────────
def claim(job_id):
    os.makedirs(CLAIM_DIR, exist_ok=True)
    p = os.path.join(CLAIM_DIR, job_id + ".claim")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def is_done(job):
    return os.path.exists(os.path.join(job["out"], "model.zip"))


def record(job, metrics, device, minutes):
    res = {}
    if os.path.exists(RESULTS):
        try:
            res = json.load(open(RESULTS))
        except (OSError, json.JSONDecodeError):
            pass
    res[job["id"]] = {"group": job["group"], "device": device,
                      "minutes": minutes, **metrics}
    tmp = RESULTS + ".tmp"
    json.dump(res, open(tmp, "w"), indent=2)
    os.replace(tmp, RESULTS)


def gpu_temp():
    try:
        o = subprocess.run([r"C:\Windows\System32\nvidia-smi.exe",
                            "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        if o.returncode == 0:
            return float(o.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


# ── worker ─────────────────────────────────────────────────────────────────
def worker(device):
    import torch
    threads = 7 if device == "cpu" else 1
    torch.set_num_threads(threads)
    os.environ["RL_DEVICE"] = device
    config.DEVICE = device

    from rl.train import train
    from rl.evaluate import load_model, evaluate

    jobs = build_jobs()
    print(f"[{device}] worker up, {threads} torch threads, {len(jobs)} jobs on the board",
          flush=True)

    for job in jobs:
        if is_done(job) or not claim(job["id"]):
            continue
        if device == "cuda":
            t = gpu_temp()
            if t is not None and t > ABORT_TEMP:
                print(f"[cuda] GPU at {t:.0f} C > {ABORT_TEMP} C — stopping; "
                      f"CPU worker continues alone", flush=True)
                return

        print(f"\n[{device}] === {job['id']} ({job['steps']:,} steps) ===", flush=True)
        t0 = time.perf_counter()
        try:
            train("SAC", out_dir=job["out"], seed=job["seed"],
                  total_steps=job["steps"], hyperparams=job["hyperparams"],
                  reward_weights=job["reward_weights"])
            model, envs = load_model(job["out"], "SAC")
            m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES,
                         seed=job["seed"])
            envs.close()
            mins = (time.perf_counter() - t0) / 60
            metrics = {k: float(m[k]) for k in
                       ("mean_final_error", "mean_min_error", "success_rate",
                        "mean_energy", "reach_5cm", "reach_10cm")}
            # Load-sharing analysis for the long (w2) jobs only.
            if job["group"] == "replicate_w2":
                subprocess.run([sys.executable, os.path.join(HERE, "force_comparison.py"),
                                "--run", job["out"], "--episodes", "20"],
                               cwd=HERE, capture_output=True, text=True)
                p = os.path.join(job["out"], "force_comparison_drl_vs_classical.json")
                if os.path.exists(p):
                    d = json.load(open(p))
                    metrics["effort_ratio"] = d["effort"]["ratio_aggregate"]
                    metrics["pearson_r"] = d["overall"]["pearson_r"]
            record(job, metrics, device, mins)
            t = gpu_temp()
            print(f"[{device}] {job['id']}: err={metrics['mean_final_error']:.4f} "
                  f"({mins:.0f} min)" + (f"  GPU {t:.0f} C" if t else ""), flush=True)
        except Exception as e:
            print(f"[{device}] {job['id']} FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"[{device}] no jobs left", flush=True)


# ── launcher ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["cpu", "cuda"])
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
        return

    # Release stale claims from an interrupted run.
    if os.path.isdir(CLAIM_DIR):
        jobs = {j["id"]: j for j in build_jobs()}
        for f in os.listdir(CLAIM_DIR):
            jid = f[:-6]
            if jid in jobs and not is_done(jobs[jid]):
                os.remove(os.path.join(CLAIM_DIR, f))
                print(f"  released stale claim: {jid}")

    print("=" * 74)
    print("PARALLEL CPU + GPU  —  measured 1.43x throughput over CPU alone")
    print("=" * 74)
    procs = []
    for dev in ("cpu", "cuda"):
        log = open(os.path.join(HERE, f"parallel_{dev}.log"), "a")
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "run_parallel.py"),
                              "--worker", dev], cwd=HERE, stdout=log,
                             stderr=subprocess.STDOUT, text=True)
        procs.append((dev, p, log))
        print(f"  {dev} worker -> parallel_{dev}.log (pid {p.pid})")
        time.sleep(8)

    for dev, p, log in procs:
        p.wait(); log.close()
        print(f"  {dev} worker exited rc={p.returncode}", flush=True)

    if os.path.exists(RESULTS):
        res = json.load(open(RESULTS))
        print(f"\n{len(res)} jobs complete. Summary -> {RESULTS}")


if __name__ == "__main__":
    main()
