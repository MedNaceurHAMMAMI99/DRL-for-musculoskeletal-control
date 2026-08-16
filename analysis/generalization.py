"""
Robustness / generalization evaluation of a trained policy.

Loads a policy together with its saved observation-normalisation statistics and
evaluates it under a set of perturbation conditions (clean, observation noise,
external force impulses, enlarged workspace). Because the policy was trained with
VecNormalize, the SAME normalisation stats are reused and only the underlying
environment is perturbed — so the measured degradation reflects the policy, not a
normalisation mismatch.

Conditions mirror the robustness/domain-randomization tables in the thesis; every
number they contain must come from running this against a REAL trained checkpoint.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import config

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl import algorithms as algo_registry


DEFAULT_CONDITIONS = {
    "Clean":                 dict(),
    "Obs noise 0.01":        dict(obs_noise_sigma=0.01),
    "Obs noise 0.03":        dict(obs_noise_sigma=0.03),
    "Obs noise 0.05":        dict(obs_noise_sigma=0.05),
    "Force 5N":              dict(ext_force_N=5.0),
    "Force 10N":             dict(ext_force_N=10.0),
    "Noise+Force":           dict(obs_noise_sigma=0.03, ext_force_N=10.0),
    "Extended workspace":    dict(workspace_scale=1.3),
}


def _eval_condition(run_dir, algo, cond_kwargs, n_episodes, seed):
    from rl.environment import ArmReachEnv

    def _f():
        return ArmReachEnv(domain_rand=False, seed=seed, **cond_kwargs)

    envs = DummyVecEnv([_f])
    envs = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"), envs)
    envs.training = False
    envs.norm_reward = False
    cls   = algo_registry.REGISTRY[algo.upper()]["cls"]
    model = cls.load(os.path.join(run_dir, "model"), env=envs)

    successes, errors = [], []
    for _ in range(n_episodes):
        obs = envs.reset()
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = envs.step(action)
        successes.append(bool(infos[0].get("success", False)))
        errors.append(float(infos[0].get("err", float("nan"))))
    return float(np.mean(successes)), float(np.nanmean(errors))


def evaluate_robustness(run_dir: str, algo: str = "SAC",
                        conditions: dict = None, n_episodes: int = None,
                        seed: int = 0) -> dict:
    conditions = conditions or DEFAULT_CONDITIONS
    n_episodes = n_episodes or config.EVAL_EPISODES
    out = {}
    clean_sr = None
    for name, kw in conditions.items():
        sr, err = _eval_condition(run_dir, algo, kw, n_episodes, seed)
        if name == "Clean":
            clean_sr = sr
        out[name] = {
            "success_rate":   sr,
            "mean_error_cm":  err * 100.0,
            "drop_pp":        None if clean_sr is None else (clean_sr - sr) * 100.0,
        }
        print(f"  [robustness {name:20s}] success={sr*100:5.1f}%")
    return out
