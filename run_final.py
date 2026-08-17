"""
Final batch: close the two remaining reachable gaps.

1. FAIR BENCHMARK (12 jobs, ~4 h)
   The existing algorithm comparison ran SAC_tuned at gamma = 0.957 against
   SAC_default, TD3, DDPG and PPO all at gamma = 0.99. The ablation has since
   established that gamma is the single decisive parameter -- worth 106 % of the
   default-to-tuned interval while four others are worth nothing. That
   comparison therefore pits one correctly-configured algorithm against four
   running with the parameter now known to cripple this task, and no ranking
   among them is meaningful.

   This re-runs all four algorithms with gamma = 0.957, three seeds each, at the
   same 100k budget, so the comparison is fair on the axis that matters.

2. GAIN SEARCH (1 job, ~1 h)
   The conventional controller's effort ratio of 1.33 is load-bearing: the claim
   that the learned policy achieves comparable load sharing rests on it, and
   w2=20's 1.304 sits just below. It comes from a single gain pair out of a
   coarse twelve-point grid. A finer 30-point search either confirms it or
   lowers it, and if it lowers, the comparison must be restated.

Same claim-file mechanism and thermal budget as run_parallel.py: two workers,
7 CPU threads plus ~1 core for GPU dispatch, self-balancing, resumable.

Run:  python run_final.py
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
CLAIM_DIR = os.path.join(config.RUNS_DIR, "_claims_final")
RESULTS   = os.path.join(config.RUNS_DIR, "final_results.json")
GAMMA     = 0.9568097976880915          # the value the ablation identified


def build_jobs():
    """Gain search first (longest single item), then the benchmark grid."""
    jobs = [{"id": "gain_search", "kind": "gain"}]
    for algo in ("SAC", "TD3", "DDPG", "PPO"):
        for seed in (0, 1, 2):
            jobs.append({
                "id": f"fair_{algo}_seed{seed}", "kind": "train", "algo": algo,
                "seed": seed, "steps": 100_000,
                # gamma is the only change from library defaults. gradient_steps
                # is held at the corrected replay ratio for every off-policy
                # method, as in the original benchmark; PPO is on-policy and
                # takes neither.
                "hyperparams": ({"gamma": GAMMA} if algo == "PPO"
                                else {"gamma": GAMMA,
                                      "gradient_steps": 2 * config.N_ENVS}),
                "out": os.path.join(config.RUNS_DIR, "fair_bench",
                                    f"{algo}_seed{seed}")})
    return jobs


def claim(job_id):
    os.makedirs(CLAIM_DIR, exist_ok=True)
    p = os.path.join(CLAIM_DIR, job_id + ".claim")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def done_already(job):
    if job["kind"] == "gain":
        return os.path.exists(os.path.join(config.RUNS_DIR, "gain_search_fine.json"))
    return os.path.exists(os.path.join(job["out"], "model.zip"))


def record(key, payload):
    res = {}
    if os.path.exists(RESULTS):
        try: res = json.load(open(RESULTS))
        except (OSError, json.JSONDecodeError): pass
    res[key] = payload
    tmp = RESULTS + ".tmp"; json.dump(res, open(tmp, "w"), indent=2)
    os.replace(tmp, RESULTS)


def do_gain_search():
    """Finer grid over the conventional controller's two gains."""
    import run_conventional as RC
    from rl.environment import ArmReachEnv
    from force_comparison import moment_arm_matrix, static_optimisation

    grid = [(kp, kd) for kp in (10., 20., 30., 45., 60., 90.)
                     for kd in (6., 12., 20., 30., 45.)]
    rows, best = [], (None, float("inf"))
    for kp, kd in grid:
        env = ArmReachEnv(domain_rand=False, seed=0)
        errs = []
        for ep in range(6):
            obs, _ = env.reset(seed=ep); done = False
            while not done:
                obs, _, t, tr, info = env.step(RC.conventional_action(env, kp, kd))
                done = t or tr
            errs.append(info["err"])
        m = float(np.mean(errs)); rows.append({"kp": kp, "kd": kd, "final_error": m})
        if m < best[1]: best = ((kp, kd), m)
        print(f"    kp={kp:>5.0f} kd={kd:>4.0f}  err {m:.4f}", flush=True)

    # Effort ratio at the best gains -- the number the thesis actually leans on.
    kp, kd = best[0]
    env = ArmReachEnv(domain_rand=False, seed=0)
    fmax = np.abs(env.model.actuator_gainprm[:, 2].copy())
    Fp, Fc, errs, mins = [], [], [], []
    for ep in range(20):
        obs, _ = env.reset(seed=ep); done = False; e = []
        while not done:
            obs, _, t, tr, info = env.step(RC.conventional_action(env, kp, kd))
            done = t or tr; e.append(info["err"])
            d = env.data
            R = moment_arm_matrix(env.model, d)
            fs = d.actuator_force.copy(); sg = np.sign(fs); sg[sg == 0] = -1.
            fp = np.abs(fs)
            fc, _ = static_optimisation(R * sg[:, None], d.qfrc_actuator.copy(),
                                        fmax, f_init=fp)
            Fp.append(fp); Fc.append(fc)
        errs.append(e[-1]); mins.append(min(e))
    Fp, Fc = np.array(Fp), np.array(Fc)
    ratio = float(np.sum((Fp/fmax)**2) / max(np.sum((Fc/fmax)**2), 1e-12))
    r = float(np.corrcoef(Fp.ravel(), Fc.ravel())[0, 1])
    out = {"grid": rows, "best_kp": kp, "best_kd": kd,
           "best_final_error": float(np.mean(errs)),
           "best_min_error": float(np.mean(mins)),
           "effort_ratio": ratio, "pearson_r": r,
           "previous_coarse": {"kp": 60., "kd": 25., "effort_ratio": 1.33}}
    json.dump(out, open(os.path.join(config.RUNS_DIR, "gain_search_fine.json"), "w"),
              indent=2)
    print(f"\n  best kp={kp:.0f} kd={kd:.0f}: err {np.mean(errs):.4f}, "
          f"effort ratio {ratio:.3f} (coarse grid gave 1.33)", flush=True)
    return out


def worker(device):
    import torch
    torch.set_num_threads(7 if device == "cpu" else 1)
    os.environ["RL_DEVICE"] = device
    config.DEVICE = device
    from rl.train import train
    from rl.evaluate import load_model, evaluate

    for job in build_jobs():
        if done_already(job) or not claim(job["id"]):
            continue
        print(f"\n[{device}] === {job['id']} ===", flush=True)
        t0 = time.perf_counter()
        try:
            if job["kind"] == "gain":
                out = do_gain_search()
                record("gain_search", out)
            else:
                train(job["algo"], out_dir=job["out"], seed=job["seed"],
                      total_steps=job["steps"], hyperparams=job["hyperparams"])
                model, envs = load_model(job["out"], job["algo"])
                m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES,
                             seed=job["seed"])
                envs.close()
                record(job["id"], {"algo": job["algo"], "seed": job["seed"],
                                   "device": device,
                                   **{k: float(m[k]) for k in
                                      ("mean_final_error", "mean_min_error",
                                       "success_rate", "mean_energy",
                                       "reach_5cm", "reach_10cm")}})
                print(f"[{device}] {job['id']}: err={m['mean_final_error']:.4f} "
                      f"({(time.perf_counter()-t0)/60:.0f} min)", flush=True)
        except Exception as e:
            print(f"[{device}] {job['id']} FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"[{device}] board empty", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--worker", choices=["cpu", "cuda"])
    a = ap.parse_args()
    if a.worker:
        worker(a.worker); return

    if os.path.isdir(CLAIM_DIR):
        jobs = {j["id"]: j for j in build_jobs()}
        for f in os.listdir(CLAIM_DIR):
            jid = f[:-6]
            if jid in jobs and not done_already(jobs[jid]):
                os.remove(os.path.join(CLAIM_DIR, f)); print(f"  released {jid}")

    print("=" * 74)
    print("FINAL BATCH — fair benchmark at the identified gamma, plus gain search")
    print("=" * 74)
    procs = []
    for dev in ("cpu", "cuda"):
        log = open(os.path.join(HERE, f"final_{dev}.log"), "a")
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "run_final.py"),
                              "--worker", dev], cwd=HERE, stdout=log,
                             stderr=subprocess.STDOUT, text=True)
        procs.append((dev, p, log)); print(f"  {dev} -> final_{dev}.log (pid {p.pid})")
        time.sleep(8)
    for dev, p, log in procs:
        p.wait(); log.close(); print(f"  {dev} exited rc={p.returncode}")


if __name__ == "__main__":
    main()
