"""
Second unattended batch: validate the newest findings before trusting them.

Ordered shortest-first, deliberately. The cheap diagnostics can invalidate the
expensive experiments: if the low-effort policies turn out to be coasting on
gravity rather than controlling efficiently, the 13-hour replication is aimed at
the wrong claim and should not be run as designed.

  1. TORQUE AUDIT      ~10 min  Are the low-effort policies actually actuating,
                                or letting gravity do the work? Compares
                                actuator torque against the gravity/Coriolis
                                bias term along real trajectories. This is the
                                same passivity confound already flagged for PPO,
                                now applied to the w2 sweep.
  2. FORCE + SYNERGY   ~25 min  Full load-sharing and NMF analysis on the w2
                                policies. Does synergy structure converge on the
                                classical one as effort weight rises? Would be a
                                fourth independent line of evidence.
  3. GAIN SEARCH       ~60 min  The 1.33 "achievable floor" rests on one
                                controller at one gain pair from a 12-point
                                grid. A finer search either confirms it or
                                lowers it -- and if it lowers, the claim that
                                w2=20 reaches the floor weakens.
  4. ABLATION REST     ~9 h     The completed ablation excluded target entropy
                                but identified no cause. Four further arms
                                (learning rate, tau, gamma, batch size), three
                                seeds each, to find which parameter is
                                responsible.
  5. REPLICATE W2      ~13 h    w2 = 5 and w2 = 20 across four further seeds.
                                The newest headline currently rests on a single
                                seed -- the same error just corrected for the
                                main result, one level up.

Sequential, 8 of 16 threads, resumable, failures isolated per stage.

Run:  python run_pending2.py
"""

import json
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
import torch
import optuna

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(8)

HERE   = os.path.dirname(os.path.abspath(__file__))
STATE  = os.path.join(config.RUNS_DIR, "pending2_state.json")
REPORT = os.path.join(config.RUNS_DIR, "pending2_summary.json")

W2_RUNS = {"w2_1": os.path.join(config.RUNS_DIR, "SAC_seed0"),
           "w2_5": os.path.join(config.RUNS_DIR, "effort_sweep", "w2_5"),
           "w2_20": os.path.join(config.RUNS_DIR, "effort_sweep", "w2_20")}


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except (OSError, json.JSONDecodeError):
            pass
    return {"done": [], "results": {}}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE)


def tuned_params():
    db = os.path.join(config.RUNS_DIR, "optuna_SAC.db")
    p = dict(optuna.load_study(study_name="SAC_arm_reach",
                               storage=f"sqlite:///{db}").best_trial.params)
    p["gradient_steps"] = 2 * config.N_ENVS
    return p


def _policy_iter(run_dir, episodes, seed=0):
    """Yield (env, info) after each step of a deterministic rollout."""
    from rl.environment import ArmReachEnv
    from rl.evaluate import load_model
    model, vec = load_model(run_dir, "SAC")
    rms, clip, eps = vec.obs_rms, vec.clip_obs, vec.epsilon
    env = ArmReachEnv(domain_rand=False, seed=seed)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            o = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps),
                        -clip, clip).astype(np.float32)
            a, _ = model.predict(o, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            yield env, a, info


# ── Stage 1 ────────────────────────────────────────────────────────────────
def stage_torque_audit(st):
    """
    Is the policy actuating, or coasting on gravity?

    `qfrc_bias` holds the gravity, Coriolis and centrifugal terms -- the
    generalised force the arm experiences with zero muscle activity.
    `qfrc_actuator` is what the muscles supply. Their ratio says how much of the
    arm's motion the controller is responsible for.

    A policy that has learned to drop its arm and let it swing would show a small
    actuator term against a large bias term, and would score a flattering effort
    ratio for a reason that has nothing to do with good load sharing.
    """
    res = st["results"].setdefault("torque_audit", {})
    for tag, run in W2_RUNS.items():
        if tag in res or not os.path.exists(os.path.join(run, "model.zip")):
            continue
        act, bias, acts, ratio = [], [], [], []
        for env, a, info in _policy_iter(run, 15):
            ta = np.linalg.norm(env.data.qfrc_actuator[:4])
            tb = np.linalg.norm(env.data.qfrc_bias[:4])
            act.append(ta); bias.append(tb); acts.append(float(np.mean(a)))
            ratio.append(ta / max(tb, 1e-9))
        res[tag] = {"mean_actuator_torque": float(np.mean(act)),
                    "mean_bias_torque": float(np.mean(bias)),
                    "actuator_over_bias": float(np.mean(ratio)),
                    "median_actuator_over_bias": float(np.median(ratio)),
                    "mean_activation": float(np.mean(acts))}
        save_state(st)
        r = res[tag]
        print(f"  {tag}: actuator/bias = {r['actuator_over_bias']:.2f} "
              f"(median {r['median_actuator_over_bias']:.2f}), "
              f"activation {r['mean_activation']:.3f}", flush=True)
    print("\n  A ratio well below 1 means gravity dominates and the controller is")
    print("  contributing little -- which would make a low effort ratio misleading.")
    return True


# ── Stage 2 ────────────────────────────────────────────────────────────────
def stage_force_synergy(st):
    res = st["results"].setdefault("force_synergy", {})
    for tag, run in W2_RUNS.items():
        if tag in res:
            continue
        subprocess.run([sys.executable, os.path.join(HERE, "force_comparison.py"),
                        "--run", run, "--episodes", "20"],
                       cwd=HERE, capture_output=True, text=True)
        p = os.path.join(run, "force_comparison_drl_vs_classical.json")
        entry = {}
        if os.path.exists(p):
            d = json.load(open(p))
            entry = {"effort_ratio": d["effort"]["ratio_aggregate"],
                     "pearson_r": d["overall"]["pearson_r"],
                     "cosine": d["overall"]["pattern_cosine_mean"],
                     "cosine_null": d["overall"]["pattern_cosine_null_mean"],
                     "nrmse_pct": d["overall"]["nrmse_pct"]}
        res[tag] = entry
        save_state(st)
        print(f"  {tag}: effort {entry.get('effort_ratio', float('nan')):.2f}  "
              f"r {entry.get('pearson_r', float('nan')):.3f}", flush=True)

    # Synergy structure across the same policies, via the existing driver.
    log = os.path.join(HERE, "synergies_w2.log")
    with open(log, "w") as f:
        subprocess.run([sys.executable, os.path.join(HERE, "run_synergies.py"),
                        "--episodes", "12"], cwd=HERE, stdout=f,
                       stderr=subprocess.STDOUT)
    print(f"  synergy analysis written to {os.path.basename(log)}", flush=True)
    return True


# ── Stage 3 ────────────────────────────────────────────────────────────────
def stage_gain_search(st):
    """Finer conventional-controller gain search: is 1.33 really the floor?"""
    import run_conventional as RC
    from rl.environment import ArmReachEnv
    from force_comparison import moment_arm_matrix, static_optimisation

    res = st["results"].setdefault("gain_search", {})
    if res.get("best"):
        return True
    grid = [(kp, kd) for kp in (20., 40., 60., 90., 130., 200.)
                     for kd in (10., 18., 25., 35., 50.)]
    best = (None, float("inf"))
    rows = []
    for kp, kd in grid:
        env = ArmReachEnv(domain_rand=False, seed=0)
        errs = []
        for ep in range(6):
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                obs, _, t, tr, info = env.step(RC.conventional_action(env, kp, kd))
                done = t or tr
            errs.append(info["err"])
        m = float(np.mean(errs))
        rows.append({"kp": kp, "kd": kd, "final_error": m})
        if m < best[1]:
            best = ((kp, kd), m)
        print(f"    kp={kp:>6.0f} kd={kd:>5.0f}  err {m:.4f}", flush=True)
    res["grid"] = rows
    res["best"] = {"kp": best[0][0], "kd": best[0][1], "final_error": best[1]}
    save_state(st)

    # Effort ratio at the best gains -- this is the number that matters.
    env = ArmReachEnv(domain_rand=False, seed=0)
    fmax = np.abs(env.model.actuator_gainprm[:, 2].copy())
    Fp, Fc = [], []
    for ep in range(15):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            obs, _, t, tr, _ = env.step(RC.conventional_action(env, *best[0]))
            done = t or tr
            d = env.data
            R = moment_arm_matrix(env.model, d)
            fs = d.actuator_force.copy(); sg = np.sign(fs); sg[sg == 0] = -1.
            fp = np.abs(fs)
            fc, _ = static_optimisation(R * sg[:, None], d.qfrc_actuator.copy(),
                                        fmax, f_init=fp)
            Fp.append(fp); Fc.append(fc)
    Fp, Fc = np.array(Fp), np.array(Fc)
    ep_ = np.sum((Fp / fmax) ** 2, 1).sum()
    ec_ = np.sum((Fc / fmax) ** 2, 1).sum()
    res["best_effort_ratio"] = float(ep_ / max(ec_, 1e-12))
    save_state(st)
    print(f"\n  best gains kp={best[0][0]:.0f} kd={best[0][1]:.0f} -> "
          f"err {best[1]:.4f}, effort ratio {res['best_effort_ratio']:.2f}")
    print(f"  (previously reported floor: 1.33)")
    return True


# ── Stage 4 ────────────────────────────────────────────────────────────────
def stage_ablation_rest(st):
    """Which of the remaining four parameters produces the improvement?"""
    from rl.train import train
    from rl.evaluate import load_model, evaluate

    tuned = tuned_params()
    base = {"gradient_steps": 2 * config.N_ENVS}
    arms = {
        "lr_only":    {**base, "learning_rate": tuned["learning_rate"]},
        "tau_only":   {**base, "tau": tuned["tau"]},
        "gamma_only": {**base, "gamma": tuned["gamma"]},
        "batch_only": {**base, "batch_size": tuned["batch_size"]},
    }
    out_root = os.path.join(config.RUNS_DIR, "ablation2")
    os.makedirs(out_root, exist_ok=True)
    res = st["results"].setdefault("ablation_rest", {})
    for arm, hp in arms.items():
        for seed in (0, 1, 2):
            key = f"{arm}_seed{seed}"
            if key in res:
                continue
            out = os.path.join(out_root, key)
            print(f"\n--- {key} ---", flush=True)
            train("SAC", out_dir=out, seed=seed, total_steps=100_000,
                  hyperparams=hp)
            model, envs = load_model(out, "SAC")
            m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=seed)
            envs.close()
            res[key] = {"arm": arm, "seed": seed,
                        "mean_final_error": float(m["mean_final_error"]),
                        "mean_min_error": float(m["mean_min_error"])}
            save_state(st)
            print(f"  {key}: err={m['mean_final_error']:.4f}", flush=True)
    return True


# ── Stage 5 ────────────────────────────────────────────────────────────────
def stage_replicate_w2(st):
    """The newest headline is single-seed. Replicate it."""
    from rl.train import train
    from rl.evaluate import load_model, evaluate

    params = tuned_params()
    out_root = os.path.join(config.RUNS_DIR, "w2_replication")
    os.makedirs(out_root, exist_ok=True)
    res = st["results"].setdefault("replicate_w2", {})
    for w2 in (5.0, 20.0):
        for seed in (1, 2, 3, 4):
            key = f"w2_{w2:g}_seed{seed}"
            if key in res:
                continue
            weights = dict(config.REWARD_WEIGHTS); weights["w2"] = w2
            out = os.path.join(out_root, key)
            print(f"\n--- {key} ---", flush=True)
            train("SAC", out_dir=out, seed=seed, total_steps=300_000,
                  hyperparams=params, reward_weights=weights)
            model, envs = load_model(out, "SAC")
            m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=seed)
            envs.close()
            entry = {k: float(m[k]) for k in
                     ("mean_final_error", "mean_min_error", "success_rate",
                      "mean_energy", "reach_5cm", "reach_10cm")}
            subprocess.run([sys.executable, os.path.join(HERE, "force_comparison.py"),
                            "--run", out, "--episodes", "20"],
                           cwd=HERE, capture_output=True, text=True)
            p = os.path.join(out, "force_comparison_drl_vs_classical.json")
            if os.path.exists(p):
                d = json.load(open(p))
                entry["effort_ratio"] = d["effort"]["ratio_aggregate"]
                entry["pearson_r"] = d["overall"]["pearson_r"]
            res[key] = entry
            save_state(st)
            print(f"  {key}: err={entry['mean_final_error']:.4f} "
                  f"effort={entry.get('effort_ratio', float('nan')):.2f}", flush=True)
    return True


STAGES = [("torque_audit",   stage_torque_audit),
          ("force_synergy",  stage_force_synergy),
          ("gain_search",    stage_gain_search),
          ("ablation_rest",  stage_ablation_rest),
          ("replicate_w2",   stage_replicate_w2)]


def main():
    st = load_state()
    print("=" * 74)
    print("VALIDATION BATCH — shortest first, so cheap checks can veto costly ones")
    print("=" * 74)
    print(f"  complete: {st['done'] or 'none'}\n", flush=True)
    for name, fn in STAGES:
        if name in st["done"]:
            print(f"[skip] {name}", flush=True)
            continue
        print(f"\n{'='*74}\n[START] {name}  ({time.strftime('%H:%M:%S')})\n{'='*74}",
              flush=True)
        t0 = time.perf_counter()
        try:
            if fn(st):
                st["done"].append(name)
                print(f"[DONE] {name} in {(time.perf_counter()-t0)/60:.0f} min",
                      flush=True)
        except Exception:
            print(f"[FAILED] {name}:\n{traceback.format_exc()}", flush=True)
        save_state(st)
    json.dump(st["results"], open(REPORT, "w"), indent=2)
    print(f"\n{'='*74}\nCOMPLETE: {st['done']}\nSummary -> {REPORT}\n{'='*74}")


if __name__ == "__main__":
    main()
