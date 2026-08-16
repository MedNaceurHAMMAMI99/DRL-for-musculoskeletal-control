"""
Did the confirmation run's EVALUATION episodes silently diverge?

Why this matters
----------------
The 50-episode evaluation reported `blow_up_rate = 0.00`, but four MuJoCo
"Nan, Inf or huge value in QACC" warnings appeared in the log. The benign
explanation is stderr buffering placing training-phase warnings after the
results. The malignant one is that divergence detection misses episodes during
evaluation — and that exact class of bug is what voided every behavioural
number this project produced before 2026-08-14: MuJoCo self-resets internally
on divergence, so a diverged episode silently teleports the arm to the home
pose and keeps reporting home-pose-to-target distance as if it were data.

An undetected divergence would inflate exactly the metrics that just improved,
so this is checked rather than assumed.

Method
------
Drives the RAW environment (no VecEnv auto-reset) with the confirmed policy and
reads MuJoCo's warning counters directly, independently of the environment's
own bookkeeping. Then cross-checks:

  * how many episodes MuJoCo actually warned in           (ground truth)
  * how many episodes the env flagged via info["blew_up"] (what evaluate sees)

If ground truth > flagged, detection is leaking and the results are suspect.

Run:  python verify_divergence.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

import config
from rl.environment import ArmReachEnv
from rl.evaluate import load_model

OUT        = os.path.join(config.RUNS_DIR, "SAC_seed0")
N_EPISODES = 50

# Every warning class this MuJoCo build exposes (3.7.0). Deliberately wider
# than the three the environment watches: if divergence is being missed, it may
# be surfacing as a class the env does not look at (e.g. BADCTRL).
WARN_NAMES = {
    mujoco.mjtWarning.mjWARN_BADQACC: "BADQACC",
    mujoco.mjtWarning.mjWARN_BADQPOS: "BADQPOS",
    mujoco.mjtWarning.mjWARN_BADQVEL: "BADQVEL",
    mujoco.mjtWarning.mjWARN_BADCTRL: "BADCTRL",
    mujoco.mjtWarning.mjWARN_INERTIA: "INERTIA",
    mujoco.mjtWarning.mjWARN_CONTACTFULL: "CONTACTFULL",
    mujoco.mjtWarning.mjWARN_CNSTRFULL: "CNSTRFULL",
}


def all_warnings(data) -> dict:
    """Every warning counter, not just the three the env watches."""
    return {name: int(data.warning[w].number) for w, name in WARN_NAMES.items()}


def main():
    model, vec = load_model(OUT, "SAC")
    # Reuse the observation normalisation the policy was trained under.
    obs_rms = vec.obs_rms
    clip    = vec.clip_obs
    eps     = vec.epsilon

    def norm(o):
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + eps),
                       -clip, clip).astype(np.float32)

    env = ArmReachEnv(domain_rand=False, seed=0)

    warned_episodes   = 0
    flagged_episodes  = 0
    warn_kinds        = {}
    final_errors      = []
    min_errors        = []
    lengths           = []
    successes         = 0

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=ep)
        before = all_warnings(env.data)
        done = False
        n = 0
        errs = []
        flagged = False
        info = {}
        while not done:
            a, _ = model.predict(norm(obs), deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            n += 1
            errs.append(info["err"])
            if info.get("blew_up"):
                flagged = True
        after = all_warnings(env.data)

        delta = {k: after[k] - before[k] for k in after if after[k] > before[k]}
        if delta:
            warned_episodes += 1
            for k, v in delta.items():
                warn_kinds[k] = warn_kinds.get(k, 0) + v
        if flagged:
            flagged_episodes += 1

        final_errors.append(errs[-1])
        min_errors.append(min(errs))
        lengths.append(n)
        successes += bool(info.get("success"))

    print("=" * 70)
    print(f"DIVERGENCE AUDIT — {N_EPISODES} evaluation episodes, raw env")
    print("=" * 70)
    print(f"  episodes where MuJoCo warned (ground truth) : {warned_episodes}")
    print(f"  episodes flagged by info['blew_up']         : {flagged_episodes}")
    if warn_kinds:
        print(f"  warning counts by kind                     : {warn_kinds}")
    print()

    if warned_episodes == 0:
        print("  RESULT: no divergence during evaluation at all. The warnings in")
        print("  the run log came from TRAINING and were buffered to the end of")
        print("  stderr. blow_up_rate = 0.00 is correct and the metrics stand.")
    elif warned_episodes == flagged_episodes:
        print("  RESULT: divergence occurred but was fully detected — every")
        print("  warning episode was flagged. Metrics are trustworthy, though")
        print("  blow_up_rate should not have read 0.00; check evaluate.py.")
    else:
        print(f"  RESULT: *** DETECTION LEAK *** {warned_episodes - flagged_episodes} "
              f"episode(s) diverged without being flagged.")
        print("  These episodes report post-self-reset home-pose errors as data.")
        print("  The confirmation-run metrics are NOT trustworthy as reported.")

    print()
    print("  independently recomputed from this audit:")
    print(f"    mean_final_error {np.mean(final_errors):.4f}   "
          f"(run reported 0.1063)")
    print(f"    mean_min_error   {np.mean(min_errors):.4f}   "
          f"(run reported 0.0740)")
    print(f"    success_rate     {successes / N_EPISODES:.4f}   "
          f"(run reported 0.0400)")
    print(f"    mean_episode_len {np.mean(lengths):.2f}")

    with open(os.path.join(OUT, "divergence_audit.json"), "w") as f:
        json.dump({"n_episodes": N_EPISODES,
                   "warned_episodes": warned_episodes,
                   "flagged_episodes": flagged_episodes,
                   "warn_kinds": warn_kinds,
                   "mean_final_error": float(np.mean(final_errors)),
                   "mean_min_error": float(np.mean(min_errors)),
                   "success_rate": successes / N_EPISODES,
                   "mean_episode_len": float(np.mean(lengths))}, f, indent=2)


if __name__ == "__main__":
    main()
