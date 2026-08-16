"""
Reward-term ablation study.

Retrains the policy with each reward term removed (weight set to 0) and measures
the resulting task performance, isolating each term's contribution. This is real
retraining, so it costs one training run per ablated term — budget accordingly.

The ablated terms map to config.REWARD_WEIGHTS:
  w1 error, w2 energy, w3 smoothness, w4 alive-bonus, w5 success-bonus.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import config


def run_ablation(algo: str = "SAC", seed: int = 0, total_steps: int = None,
                 terms=("w1", "w2", "w3"), out_root: str = None) -> dict:
    """
    For 'full' and each ablated term, train + evaluate. Returns a table of
    {condition: {success_rate, mean_energy, mean_final_error}}.
    """
    from rl.train import train
    from rl.evaluate import load_model, evaluate

    total_steps = total_steps or config.TRAIN_STEPS
    out_root    = out_root or os.path.join(config.RUNS_DIR, "ablation")
    os.makedirs(out_root, exist_ok=True)

    conditions = {"full": None}
    for t in terms:
        conditions[f"no_{t}"] = {t: 0.0}

    results = {}
    for name, override in conditions.items():
        run_dir = os.path.join(out_root, f"{algo}_{name}_seed{seed}")
        # Reward override is applied by re-defining the env factory:
        _train_with_reward(algo, run_dir, seed, total_steps, override)
        model, envs = load_model(run_dir, algo=algo)
        m = evaluate(model, envs, n_episodes=config.EVAL_EPISODES, seed=seed)
        results[name] = {
            "success_rate":     m["success_rate"],
            "mean_energy":      m["mean_energy"],
            "mean_final_error": m["mean_final_error"],
        }
        print(f"  [ablation {name}] success={m['success_rate']*100:.1f}%  "
              f"energy={m['mean_energy']:.3f}")
    return results


def _train_with_reward(algo, run_dir, seed, total_steps, reward_override):
    """Train with a reward-weight override by monkey-patching the env factory."""
    import rl.train as T
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rl.environment import ArmReachEnv
    from rl import algorithms as algo_registry

    os.makedirs(run_dir, exist_ok=True)

    def make_env(s):
        def _f():
            e = ArmReachEnv(domain_rand=True, seed=s, reward_weights=reward_override)
            e.reset(seed=s)
            return e
        return _f

    envs = DummyVecEnv([make_env(seed + i) for i in range(config.N_ENVS)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = algo_registry.build(algo, envs, seed=seed, device=config.DEVICE)
    model.learn(total_timesteps=total_steps, progress_bar=False)
    model.save(os.path.join(run_dir, "model"))
    envs.save(os.path.join(run_dir, "vecnormalize.pkl"))
