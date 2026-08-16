"""
Launch the SAC hyperparameter search with thermal-safe parallelism.

Why this file exists rather than four shell invocations
-------------------------------------------------------
The 2026-08-14 15:16 launch started four workers by hand and produced nothing
usable. Two separate faults, both addressed here:

  * The four workers raced on SQLite schema creation
    ("UNIQUE constraint failed: alembic_version.version_num") and three died
    instantly. The study is therefore CREATED ONCE here, before any worker
    starts, so every worker only ever opens an existing study.
  * Nine minutes in, the machine hard-locked (Kernel-Power 41, no bugcheck, no
    WHEA, no low-memory event) under 4 workers x 3 torch threads. The 4-way
    parallelism was not even buying throughput: bench_threads.py measures
    17.36 ms/update at 4 threads against 14.39 at 8, i.e. thread scaling is
    poor and the workers were starving each other while saturating the CPU.

Thermal budget
--------------
The reference point is pilot 6: ONE process at 8 torch threads ran 78 minutes
to completion with no instability. This launcher holds total thread demand at
that same level — WORKERS * THREADS_PER_WORKER = 8 — leaving 8 of the 16
logical CPUs idle. Two workers at 4 threads deliver ~1.66x the throughput of
one at 8 (2 / (17.36/14.39)), so the parallelism is nearly free at equal load.

Do not raise WORKERS without lowering THREADS_PER_WORKER to match.
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optuna

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)

ALGO               = "SAC"
WORKERS            = 2
THREADS_PER_WORKER = 4      # WORKERS * THREADS_PER_WORKER must stay <= 8
TRIALS_PER_WORKER  = 8      # 16 trials total
TRIAL_STEPS        = 100_000

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    assert WORKERS * THREADS_PER_WORKER <= 8, (
        "Total thread demand exceeds the level pilot 6 ran safely at. "
        "This machine hard-locked under sustained full-CPU load on 2026-08-14.")

    os.makedirs(config.RUNS_DIR, exist_ok=True)
    db = os.path.join(config.RUNS_DIR, f"optuna_{ALGO}.db")
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db}",
        engine_kwargs={"connect_args": {"timeout": 60}},
        heartbeat_interval=60, grace_period=300)

    # Create the study up front — see the schema-race note in the docstring.
    study = optuna.create_study(
        study_name=f"{ALGO}_arm_reach", direction="minimize",
        storage=storage, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))

    stale = [t for t in study.trials
             if t.state == optuna.trial.TrialState.RUNNING]
    if stale:
        sys.exit(f"{len(stale)} trial(s) still marked RUNNING: "
                 f"{[t.number for t in stale]}.\n"
                 "Run `python cleanup_optuna.py --apply` first.")

    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Study {ALGO}_arm_reach: {len(study.trials)} trials on record, "
          f"{len(done)} complete.")
    print(f"Launching {WORKERS} workers x {TRIALS_PER_WORKER} trials "
          f"x {TRIAL_STEPS:,} steps, {THREADS_PER_WORKER} torch threads each.")
    print(f"Thread demand: {WORKERS * THREADS_PER_WORKER} of "
          f"{os.cpu_count()} logical CPUs (pilot 6 ran safely at 8).")
    # Replay ratio is now fixed at 2*N_ENVS/N_ENVS = 1.87 updates per env step,
    # and batch_size is capped at 512, so a trial costs ~187k updates at
    # 12-17 ms each (bench_threads.py, 4 threads) -> ~40-55 min.
    #
    # The previous estimate here assumed ratio 0.93 and one batch size and was
    # wrong by ~3x, because it ignored that the space itself set the cost: the
    # gs=8/bs=1024 corner ran ~5x slower than the cheap corner and one trial was
    # on track for 3.7 h. Cost is now near-uniform across the space.
    est_h = TRIALS_PER_WORKER * TRIAL_STEPS * 1.867 * 0.0145 / 3600
    print(f"Estimated wall-clock: ~{est_h:.1f} h (less with pruning).\n")

    procs = []
    for w in range(WORKERS):
        log = os.path.join(HERE, f"search_w{w}.log")
        cmd = [sys.executable, os.path.join(HERE, "rl", "optuna_search.py"),
               "--algo", ALGO,
               "--trials", str(TRIALS_PER_WORKER),
               "--steps", str(TRIAL_STEPS),
               "--seed", "0",                 # training seed fixed => fair trials
               "--sampler-seed", str(w),      # sampler seed differs => no duplicates
               "--threads", str(THREADS_PER_WORKER)]
        f = open(log, "w")
        procs.append((w, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                          cwd=HERE), f))
        print(f"  worker {w} -> {os.path.basename(log)} (pid {procs[-1][1].pid})")
        time.sleep(5)   # stagger startup so workers do not collide on first write

    print("\nWaiting for workers...\n", flush=True)
    for w, p, f in procs:
        rc = p.wait()
        f.close()
        print(f"  worker {w} exited rc={rc}", flush=True)

    study = optuna.load_study(study_name=f"{ALGO}_arm_reach", storage=storage)
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    print("\n" + "=" * 74)
    print(f"SEARCH DONE — {len(complete)} complete / {len(study.trials)} trials")
    print("=" * 74)
    if not complete:
        print("  NO COMPLETED TRIALS. See search_w*.log.")
        return

    best = study.best_trial
    print(f"  best mean_final_error = {best.value:.4f} m   (trial {best.number})")
    print(f"  pilot 6 reference     = 0.2040 m at 300k steps "
          f"(these trials are 100k steps)")
    print("\n  best params:")
    for k, v in sorted(best.params.items()):
        print(f"    {k:<16} {v}")
    print("\n  best trial's other metrics:")
    for k, v in sorted(best.user_attrs.items()):
        print(f"    {k:<18} {v:.4f}" if isinstance(v, float) else f"    {k:<18} {v}")

    print("\n  all completed trials (by objective):")
    for t in sorted(complete, key=lambda t: t.value):
        print(f"    trial {t.number:>2}  err={t.value:.4f}  "
              f"lr={t.params.get('learning_rate', float('nan')):.2e}  "
              f"gs={t.params.get('gradient_steps')}  "
              f"te={t.params.get('target_entropy', float('nan')):.1f}  "
              f"drift={t.user_attrs.get('drift', float('nan')):.3f}")


if __name__ == "__main__":
    main()
