"""
Verify how many GRADIENT UPDATES SB3 actually performs per environment step
for the off-policy algorithms, as configured by this project.

Why this exists
---------------
`config.N_ENVS = 4` and `rl/algorithms.py` sets `train_freq=1, gradient_steps=1`
for SAC/TD3/DDPG. In Stable-Baselines3 one rollout iteration on an n-env VecEnv
collects n transitions (one per env) and is followed by `gradient_steps` updates.
If that is what happens here, the replay ratio is 1/n_envs = 0.25, not the
standard 1.0, and every run this project has done is ~4x less trained than its
step count implies.

This measures it directly rather than arguing from the documentation: it counts
calls into `model.train()` and the gradient_steps each call was given, against
the environment steps actually consumed.

Run:  python verify_replay_ratio.py
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


def measure(algo: str, n_envs: int, gradient_steps: int, total_steps: int = 3000):
    """Return (env_steps, n_updates) actually performed."""
    envs = _make_vec_env(seed=0, domain_rand=True, n_envs=n_envs)
    model = algo_registry.build(
        algo, envs, seed=0, device="cpu",
        learning_starts=200, buffer_size=20_000,
        gradient_steps=gradient_steps,
    )

    counter = {"updates": 0, "calls": 0}
    original_train = model.train

    def counting_train(gradient_steps, batch_size):
        counter["calls"] += 1
        counter["updates"] += gradient_steps
        return original_train(gradient_steps=gradient_steps, batch_size=batch_size)

    model.train = counting_train
    model.learn(total_timesteps=total_steps, progress_bar=False)

    env_steps = int(model.num_timesteps)
    envs.close()
    return env_steps, counter["updates"], counter["calls"]


def main():
    import stable_baselines3 as sb3
    import mujoco
    import optuna

    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    print(f"  stable_baselines3 : {sb3.__version__}")
    print(f"  torch             : {torch.__version__}")
    print(f"  mujoco            : {mujoco.__version__}")
    print(f"  optuna            : {optuna.__version__}")
    print(f"  torch threads     : {torch.get_num_threads()}  (default, unset by this project)")
    print(f"  config.N_ENVS     : {config.N_ENVS}")
    print()

    print("=" * 72)
    print("REPLAY RATIO  (gradient updates per environment step)")
    print("=" * 72)
    print(f"  {'config':<34} {'env steps':>10} {'updates':>9} {'ratio':>8}")
    print("  " + "-" * 64)

    cases = [
        # (label,                              n_envs, gradient_steps)
        ("AS CONFIGURED: n_envs=4, gs=1",            4,   1),
        ("n_envs=1, gs=1  (standard SAC)",           1,   1),
        ("n_envs=4, gs=4  (candidate fix)",          4,   4),
        ("n_envs=4, gs=-1 (SB3 'match env steps')",  4,  -1),
    ]

    results = {}
    for label, n_envs, gs in cases:
        steps, updates, calls = measure("SAC", n_envs, gs)
        ratio = updates / max(steps, 1)
        results[label] = ratio
        print(f"  {label:<34} {steps:>10,} {updates:>9,} {ratio:>8.3f}")

    print()
    configured = results["AS CONFIGURED: n_envs=4, gs=1"]
    print("=" * 72)
    if configured < 0.5:
        print(f"  CONFIRMED: the project trains at a replay ratio of {configured:.2f}.")
        print(f"  A 1,000,000-step run performs only ~{configured*1e6:,.0f} gradient updates.")
    else:
        print(f"  NOT CONFIRMED: replay ratio is {configured:.2f}. The hypothesis is wrong.")
    print("=" * 72)


if __name__ == "__main__":
    main()
