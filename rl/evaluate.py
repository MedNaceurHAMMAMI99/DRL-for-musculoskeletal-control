"""
Load a trained policy (any algorithm) and evaluate it.

Returns per-episode metric arrays (not just means) so the experiment runner can
aggregate across seeds and feed real numbers to the Wilcoxon/BCa statistics. Also
records the per-step muscle-activation trace a(t) in [0,1]^9 used by the synergy
(NMF/VAF) and co-contraction-index analyses — a(t) is defined as the commanded
activation (the policy action), which is exactly what a deployed controller emits.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
import config

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl import algorithms as algo_registry


def load_model(run_dir: str = None, algo: str = "SAC", domain_rand: bool = False):
    """Load a trained model + normalisation stats from run_dir."""
    from rl.environment import ArmReachEnv

    run_dir = run_dir or config.RUNS_DIR
    envs    = DummyVecEnv([lambda: ArmReachEnv(domain_rand=domain_rand, seed=0)])
    envs    = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"), envs)
    envs.training    = False
    envs.norm_reward = False
    cls   = algo_registry.REGISTRY[algo.upper()]["cls"]
    model = cls.load(os.path.join(run_dir, "model"), env=envs)
    return model, envs


def evaluate(model, envs, n_episodes: int = None, seed: int = 0,
             log_activations: bool = False) -> dict:
    """
    Roll out the deterministic policy for n_episodes.

    Returns per-episode arrays plus aggregate means. If log_activations, also
    returns 'activations' — a list of (T, 9) arrays, one per episode.
    """
    n_episodes = n_episodes or config.EVAL_EPISODES
    successes, rewards, errors, energies, times_us, lengths = [], [], [], [], [], []
    min_errors = []
    act_traces = []
    blew_ups   = []

    for _ in range(n_episodes):
        obs   = envs.reset()
        ep_r  = 0.0
        ep_e  = 0.0
        steps = 0
        done  = [False]
        traj  = []
        min_err = float("inf")

        while not done[0]:
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            times_us.append((time.perf_counter() - t0) * 1e6)
            if log_activations:
                traj.append(np.clip(action[0], 0.0, 1.0))
            obs, r, done, infos = envs.step(action)
            ep_r += float(r[0])
            ep_e += float(infos[0].get("energy", 0.0))
            min_err = min(min_err, float(infos[0].get("err", np.inf)))
            steps += 1

        blew_ups.append(bool(infos[0].get("blew_up", False)))
        successes.append(bool(infos[0].get("success", False)))
        rewards.append(ep_r)
        errors.append(float(infos[0].get("err", float("nan"))))
        min_errors.append(min_err)
        energies.append(ep_e / max(steps, 1))
        lengths.append(steps)
        if log_activations:
            act_traces.append(np.asarray(traj, dtype=np.float32))

    errs_arr, min_arr = np.asarray(errors), np.asarray(min_errors)
    out = {
        "success_rate":      float(np.mean(successes)),
        "mean_reward":       float(np.mean(rewards)),
        "mean_final_error":  float(np.nanmean(errors)),
        "mean_energy":       float(np.mean(energies)),
        "mean_inference_us": float(np.mean(times_us)),
        "n_episodes":        n_episodes,
        # Episode-integrity metrics. These exist because a policy that ends
        # episodes early can post plausible-looking error numbers while never
        # attempting the task: before the v3 reward fix, 100% of episodes ended
        # in a deliberate blow-up at step ~7 of 100 and nothing in this dict
        # would have revealed it. A healthy run has blow_up_rate 0.0 and
        # mean_episode_len == config.MAX_EPISODE_STEPS.
        "blow_up_rate":      float(np.mean(blew_ups)),
        "mean_episode_len":  float(np.mean(lengths)),
        "min_episode_len":   int(np.min(lengths)),
        "max_episode_steps": config.MAX_EPISODE_STEPS,
        # Graded accuracy. The headline `success_rate` is the strict criterion
        # (< 2 cm AND joint speed < 0.1 rad/s, i.e. reached AND settled). A
        # privileged-information oracle (reach_oracle.py) satisfies it in only
        # ~7% of episodes, so reporting it alone hides all between-algorithm
        # differences. These position-only rates, at the final step and at the
        # closest approach, are the graded measures reaching studies report.
        "reach_2cm":         float(np.mean(errs_arr < 0.02)),
        "reach_5cm":         float(np.mean(errs_arr < 0.05)),
        "reach_10cm":        float(np.mean(errs_arr < 0.10)),
        "touch_2cm":         float(np.mean(min_arr < 0.02)),
        "touch_5cm":         float(np.mean(min_arr < 0.05)),
        "mean_min_error":    float(np.nanmean(min_arr)),
        # Per-episode arrays for statistics across seeds:
        "success_per_ep":    [bool(s) for s in successes],
        "reward_per_ep":     [float(x) for x in rewards],
        "error_per_ep":      [float(x) for x in errors],
        "min_error_per_ep":  [float(x) for x in min_errors],
        "energy_per_ep":     [float(x) for x in energies],
    }
    if log_activations:
        out["activations"] = act_traces
    return out
