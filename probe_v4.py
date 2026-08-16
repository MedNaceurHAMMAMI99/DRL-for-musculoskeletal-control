"""
Short probe: does the v4 reward break the co-contraction, and is domain
randomisation partly responsible for it?

Two hypotheses for the pilot-4 policy's co-contraction index of 0.725:
  (a) effort was too cheap under v3 (w2=0.1, ~3% of the per-step reward), so
      there was no pressure to relax — addressed by v4's w2=1.0;
  (b) domain randomisation. The agent trains with per-muscle strength varied
      +-20% but is evaluated with none. Stiffening the arm by co-contracting is
      the textbook rational response to unknown actuator gains, because it makes
      the limb's behaviour less sensitive to them. If this is the driver, no
      amount of effort penalty will remove it without also costing robustness.

Runs a short training budget under each condition and reports the behaviour
metrics that matter, so the 75-minute run is spent on the right configuration.

Usage:  python probe_v4.py <dr_on|dr_off> [steps]
"""

import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config

MODE  = sys.argv[1] if len(sys.argv) > 1 else "dr_on"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 120_000
DR    = (MODE == "dr_on")

config.EVAL_FREQ       = 40_000
config.CHECKPOINT_FREQ = 60_000

from rl import train as T
from rl.environment import ArmReachEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl import algorithms as algo_registry

# Patch the training env factory's domain_rand for this probe only.
_orig_make = T._make_vec_env
def _make(seed, domain_rand=True, n_envs=None, vecnormalize_path=None):
    return _orig_make(seed, domain_rand=DR, n_envs=n_envs,
                      vecnormalize_path=vecnormalize_path)
T._make_vec_env = _make

OUT = os.path.join(config.RUNS_DIR, f"PROBE_{MODE}")
shutil.rmtree(OUT, ignore_errors=True)

print(f"PROBE v4 [{MODE}] — SAC seed 0, {STEPS:,} steps, "
      f"domain_rand={DR}, w2={config.REWARD_WEIGHTS['w2']}, "
      f"w3={config.REWARD_WEIGHTS['w3']}, w7={config.REWARD_WEIGHTS['w7']}",
      flush=True)
T.train("SAC", out_dir=OUT, seed=0, total_steps=STEPS)

# ---- behaviour metrics on the raw env (no VecEnv auto-reset) ----
_probe = DummyVecEnv([lambda: ArmReachEnv(domain_rand=False, seed=0)])
vecnorm = VecNormalize.load(os.path.join(OUT, "vecnormalize.pkl"), _probe)
vecnorm.training = False
model = algo_registry.REGISTRY["SAC"]["cls"].load(os.path.join(OUT, "model"))
env = ArmReachEnv(domain_rand=False, seed=0)

curves, acts_all, blew, finals, targets = [], [], [], [], []
for ep in range(20):
    obs, _ = env.reset()
    targets.append(env.target.copy())
    errs = []
    while True:
        a, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        acts_all.append(np.clip(a, 0.0, 1.0))
        obs, r, term, trunc, info = env.step(a)
        errs.append(info["err"])
        if term or trunc:
            break
    curves.append(errs)
    blew.append(info["blew_up"])
    finals.append(env.data.site_xpos[env._site_id].copy())

full = [c for c, b in zip(curves, blew) if not b and len(c) == config.MAX_EPISODE_STEPS]
E = np.asarray(full)
A = np.asarray(acts_all)
ago = A[:, config.CCI_AGONIST_IDX].mean(axis=1)
ant = A[:, config.CCI_ANTAGONIST_IDX].mean(axis=1)
cci = float(2.0 * np.minimum(ago, ant).sum() / max((ago + ant).sum(), 1e-9))
finals, targets = np.asarray(finals), np.asarray(targets)
spread = float(np.mean(np.std(finals, axis=0) / np.maximum(np.std(targets, axis=0), 1e-9)))

res = dict(
    mode=MODE, steps=STEPS,
    final_error=float(E[:, -1].mean()) if len(E) else float("nan"),
    best_error=float(E.min(axis=1).mean()) if len(E) else float("nan"),
    drift=float(E[:, -1].mean() - E.min(axis=1).mean()) if len(E) else float("nan"),
    mean_activation=float(A.mean()),
    cci=cci,
    spread_ratio=spread,
    blow_up_rate=float(np.mean(blew)),
    reach_10cm=float(np.mean(E[:, -1] < 0.10)) if len(E) else float("nan"),
    reach_5cm=float(np.mean(E[:, -1] < 0.05)) if len(E) else float("nan"),
)
print(f"\nPROBE RESULT [{MODE}]")
for k, v in res.items():
    print(f"  {k:>18}: {v}")
json.dump(res, open(os.path.join(config.RUNS_DIR, f"probe_{MODE}.json"), "w"), indent=2)
shutil.rmtree(OUT, ignore_errors=True)
