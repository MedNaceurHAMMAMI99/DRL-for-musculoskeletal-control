"""
Algorithm registry — one place that maps an algorithm name to its
Stable-Baselines3 class and hyperparameters.

This is what makes the four-algorithm benchmark real: SAC, TD3, DDPG and PPO all
train on the SAME `ArmReachEnv` through the SAME `train()` entry point, differing
only by the definitions below. Hyperparameters are the SB3-documented defaults
for continuous-control MuJoCo tasks, with the network architecture ([256, 256])
and discount held common across algorithms so the comparison is fair.

To run an equal-budget Optuna search over any of these, see `optuna_search.py`;
the values here are the current defaults and become the search's starting point.
"""

import numpy as np

from stable_baselines3 import SAC, TD3, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise

# Common to every algorithm — fair-comparison invariants.
NET_ARCH = [256, 256]
GAMMA    = 0.99
LR       = 3e-4
N_ACTIONS = 9

# Replay ratio (off-policy algorithms) — fixed 2026-08-14.
#
# `gradient_steps=1` with `config.N_ENVS = 4` was a silent 4x under-training
# bug. In SB3 one rollout iteration on an n-env VecEnv collects n transitions
# (one per env) and is then followed by `gradient_steps` updates, so the ratio
# of gradient updates to environment steps was 1/4, not the standard 1.
# Measured directly by verify_replay_ratio.py on this exact stack
# (SB3 2.8.0 / torch 2.11.0+cpu):
#
#     n_envs=4, gradient_steps=1   ->  0.233 updates per env step
#     n_envs=1, gradient_steps=1   ->  0.933
#     n_envs=4, gradient_steps=-1  ->  0.933
#
# So every run before this date performed ~233,000 gradient updates for a
# nominal 1,000,000-step budget. That is the arithmetic behind the symptoms
# the pilots kept reproducing: a critic that never converged (the evaluation
# curve swung 2-3x between checkpoints to the end of training) and a policy
# that could approach the target but not refine or hold position.
#
# -1 means "as many gradient steps as environment steps collected in this
# rollout", which keeps the ratio at ~1 for ANY value of config.N_ENVS — the
# fix therefore cannot silently regress if the env count is changed again.
# Cost: a run now does ~4x the gradient work, and since a gradient update is
# ~14 ms against ~7 us for a physics step, wall-clock is ~4x too (bench_threads.py).
GRADIENT_STEPS = -1

# Off-policy exploration noise for the deterministic-policy algorithms.
_ACTION_NOISE_SIGMA = 0.1


def _action_noise():
    return NormalActionNoise(
        mean=np.zeros(N_ACTIONS),
        sigma=_ACTION_NOISE_SIGMA * np.ones(N_ACTIONS),
    )


# name -> (SB3 class, kwargs builder). Builders are functions so per-run objects
# (action noise, seeds) are constructed fresh each call.
REGISTRY = {
    "SAC": dict(
        cls=SAC,
        kwargs=lambda: dict(
            learning_rate=LR, buffer_size=300_000, batch_size=512,
            learning_starts=4_000, tau=0.005, gamma=GAMMA,
            train_freq=1, gradient_steps=GRADIENT_STEPS,
            ent_coef="auto", target_entropy="auto",
            policy_kwargs=dict(net_arch=NET_ARCH),
        ),
        on_policy=False,
    ),
    "TD3": dict(
        cls=TD3,
        kwargs=lambda: dict(
            learning_rate=LR, buffer_size=300_000, batch_size=256,
            learning_starts=4_000, tau=0.005, gamma=GAMMA,
            train_freq=(1, "step"), gradient_steps=GRADIENT_STEPS,
            policy_delay=2, action_noise=_action_noise(),
            policy_kwargs=dict(net_arch=NET_ARCH),
        ),
        on_policy=False,
    ),
    "DDPG": dict(
        cls=DDPG,
        kwargs=lambda: dict(
            learning_rate=LR, buffer_size=300_000, batch_size=256,
            learning_starts=4_000, tau=0.005, gamma=GAMMA,
            train_freq=(1, "step"), gradient_steps=GRADIENT_STEPS,
            action_noise=_action_noise(),
            policy_kwargs=dict(net_arch=NET_ARCH),
        ),
        on_policy=False,
    ),
    "PPO": dict(
        cls=PPO,
        kwargs=lambda: dict(
            learning_rate=LR, n_steps=2048, batch_size=64, n_epochs=10,
            gamma=GAMMA, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            policy_kwargs=dict(net_arch=NET_ARCH),
        ),
        on_policy=True,
    ),
}


def is_registered(name: str) -> bool:
    return name.upper() in REGISTRY


def is_on_policy(name: str) -> bool:
    return REGISTRY[name.upper()]["on_policy"]


def build(name: str, env, seed: int, device: str = "cpu",
          tensorboard_log: str = None, **overrides):
    """
    Construct an SB3 model for `name` on `env`.

    `overrides` replaces individual hyperparameters (used by the Optuna search).
    """
    name = name.upper()
    if name not in REGISTRY:
        raise ValueError(f"Unknown algorithm {name!r}. "
                         f"Known: {list(REGISTRY)}")
    entry  = REGISTRY[name]
    kwargs = entry["kwargs"]()
    kwargs.update(overrides)
    return entry["cls"](
        "MlpPolicy", env,
        seed=seed, device=device, verbose=0,
        tensorboard_log=tensorboard_log,
        **kwargs,
    )
