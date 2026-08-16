"""
Training entry point for the arm-reaching task, generalised to any algorithm
in `rl.algorithms.REGISTRY` (SAC, TD3, DDPG, PPO).

All algorithms train on the same `ArmReachEnv` for the same `total_steps`
budget so the four-way benchmark is fair. A `MetricsCallback` records the loss
history that is available for each algorithm (used for the learning-curve
figure) and the measured wall-clock training time (a real, honest artifact).

Crash resilience (added 2026-08-14). A forced Windows-Update reboot destroyed a
~900k-step SAC run that had no on-disk state, because the only save happened
after `model.learn()` returned. Training now checkpoints every
`config.CHECKPOINT_FREQ` steps into `<out_dir>/checkpoint/` (model, VecNormalize
statistics, replay buffer, and the learning curve so far), written to temporary
files and atomically renamed so an interrupted write cannot corrupt the previous
checkpoint. `train()` resumes from that directory automatically, so an
interruption now costs at most one checkpoint interval instead of the whole run.
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import deque
import numpy as np
import config

from stable_baselines3.common.vec_env import (DummyVecEnv, VecNormalize,
                                              sync_envs_normalization)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from rl import algorithms as algo_registry


class EvalCurveCallback(BaseCallback):
    """
    Periodic deterministic evaluation during training -> a real learning /
    sample-efficiency curve per run (logged to eval_curve in train_meta.json).
    Uses a separate eval env (no domain randomisation, unnormalised rewards)
    whose observation statistics are synced from the training env, so it
    matches the final evaluation protocol exactly. Adds ~5% overhead; does not
    influence training.
    """

    def __init__(self, eval_env, every: int, n_episodes: int = 5,
                 best_dir: str = None):
        super().__init__(verbose=0)
        self.eval_env   = eval_env
        self.every      = every
        self.n_episodes = n_episodes
        self._next      = every
        self.points     = []
        # Best-model selection. The curve swings ~3x between checkpoints, so
        # reporting whatever policy the last gradient step happened to produce
        # throws away the run's own best result. Selection happens on THIS
        # env — seeded seed+10_000, i.e. a validation stream distinct from the
        # seed-0 stream rl/evaluate.py reports on — so the reported number is
        # not the one that was selected on.
        self.best_dir   = best_dir
        self.best_err   = float("inf")
        self.best_step  = None

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        sync_envs_normalization(self.model.get_vec_normalize_env(), self.eval_env)
        succ, errs, lens, blew = [], [], [], []
        for _ in range(self.n_episodes):
            obs  = self.eval_env.reset()
            done = [False]
            info = [{}]
            n    = 0
            while not done[0]:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, info = self.eval_env.step(a)
                n += 1
            succ.append(bool(info[0].get("success", False)))
            errs.append(float(info[0].get("err", np.nan)))
            blew.append(bool(info[0].get("blew_up", False)))
            lens.append(n)
        err_mean  = float(np.nanmean(errs))
        blow_rate = float(np.mean(blew))
        point = {"t": int(self.num_timesteps),
                 "success_rate": float(np.mean(succ)),
                 "mean_final_error": err_mean,
                 # Logged every checkpoint so a policy that stops attempting the
                 # task can never again masquerade as a plateauing one.
                 "blow_up_rate": blow_rate,
                 "mean_episode_len": float(np.mean(lens))}
        self.points.append(point)

        flag = ""
        if self.best_dir is not None and blow_rate == 0.0 and err_mean < self.best_err:
            # Only a run that completes its episodes is eligible: a blown-up
            # episode's error is not a measurement of reaching performance.
            self.best_err, self.best_step = err_mean, int(self.num_timesteps)
            os.makedirs(self.best_dir, exist_ok=True)
            _atomic_save(self.model.save, os.path.join(self.best_dir, "model.zip"))
            _atomic_save(self.model.get_vec_normalize_env().save,
                         os.path.join(self.best_dir, "vecnormalize.pkl"))
            flag = "  <- best"

        print(f"    [curve {self.num_timesteps:>9,}] "
              f"success={100*np.mean(succ):.0f}%  err={err_mean:.3f}  "
              f"blowup={100*blow_rate:.0f}%  len={np.mean(lens):.0f}{flag}",
              flush=True)
        return True


def _atomic_save(save_fn, final_path: str):
    """
    Write via `save_fn(tmp_path)` then atomically rename onto `final_path`.

    A checkpoint is only useful if it survives being interrupted mid-write, which
    is exactly the failure mode this guards against: `os.replace` is atomic on
    NTFS, so a crash during the save leaves the PREVIOUS checkpoint intact rather
    than a half-written file that cannot be loaded.
    """
    tmp = final_path + ".tmp"
    save_fn(tmp)
    # SB3's save() appends .zip if the path has no extension; find what it wrote.
    if not os.path.exists(tmp) and os.path.exists(tmp + ".zip"):
        tmp = tmp + ".zip"
    os.replace(tmp, final_path)


class CheckpointCallback(BaseCallback):
    """
    Periodically persist everything needed to resume this run.

    Writes model weights, the VecNormalize observation statistics, the replay
    buffer (off-policy algorithms only, if `config.CHECKPOINT_REPLAY_BUFFER`),
    and a progress record holding the step count, accumulated wall-clock and the
    learning curve so far. Everything is overwritten in place, so disk use is
    bounded regardless of run length.
    """

    def __init__(self, ckpt_dir: str, every: int, envs, curve_cb,
                 algo: str, seed: int, t0: float, prior_wall_s: float = 0.0):
        super().__init__(verbose=0)
        self.dir      = ckpt_dir
        self.every    = every
        self.envs     = envs
        self.curve_cb = curve_cb
        self.algo     = algo
        self.seed     = seed
        self.t0       = t0
        self.prior_wall_s = prior_wall_s
        self._next    = None
        os.makedirs(self.dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self._next is None:                    # first call: align to now
            self._next = self.num_timesteps + self.every
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        self.save()
        return True

    def save(self):
        _atomic_save(self.model.save, os.path.join(self.dir, "model.zip"))
        _atomic_save(self.envs.save, os.path.join(self.dir, "vecnormalize.pkl"))
        if config.CHECKPOINT_REPLAY_BUFFER and hasattr(self.model, "save_replay_buffer"):
            _atomic_save(self.model.save_replay_buffer,
                         os.path.join(self.dir, "replay_buffer.pkl"))
        progress = {
            "algo": self.algo, "seed": self.seed,
            "num_timesteps": int(self.num_timesteps),
            "wall_clock_s": self.prior_wall_s + (time.perf_counter() - self.t0),
            "eval_curve": self.curve_cb.points,
            "best_err": self.curve_cb.best_err,
            "best_step": self.curve_cb.best_step,
        }
        _atomic_save(
            lambda p: open(p, "w").write(json.dumps(progress, indent=2)),
            os.path.join(self.dir, "progress.json"))
        print(f"    [checkpoint {self.num_timesteps:>9,}] saved", flush=True)


def _load_checkpoint(ckpt_dir: str):
    """Return the progress dict of a resumable checkpoint, or None."""
    prog = os.path.join(ckpt_dir, "progress.json")
    model = os.path.join(ckpt_dir, "model.zip")
    if not (os.path.exists(prog) and os.path.exists(model)):
        return None
    try:
        with open(prog) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


class MetricsCallback(BaseCallback):
    """
    Records whatever training-loss signals an algorithm exposes, for plotting.

    SAC exposes actor/critic loss + entropy coefficient; TD3/DDPG expose
    actor/critic loss; PPO exposes a combined loss. Missing keys are simply not
    recorded, so this callback is safe for every algorithm.
    """

    def __init__(self):
        super().__init__(verbose=0)
        self.timesteps     = []
        self.actor_losses  = []
        self.critic_losses = []
        self.ent_coefs     = []

    def _on_step(self) -> bool:
        log = self.model.logger.name_to_value
        if not any(k.startswith("train/") for k in log):
            return True
        self.timesteps.append(self.num_timesteps)
        self.actor_losses.append(log.get("train/actor_loss",
                                          log.get("train/loss", np.nan)))
        self.critic_losses.append(log.get("train/critic_loss",
                                           log.get("train/value_loss", np.nan)))
        self.ent_coefs.append(log.get("train/ent_coef", np.nan))
        return True


class ConvergenceCallback(MetricsCallback):
    """
    Optional early stop for SAC-style algorithms (kept for backward
    compatibility; NOT used by the fair-comparison benchmark, which trains every
    algorithm to the same fixed budget).
    """

    def __init__(self, patience=100, actor_rel_tol=0.05,
                 critic_abs_tol=1e-3, ent_abs_tol=1e-4, min_steps=50_000):
        super().__init__()
        self.patience       = patience
        self.actor_rel_tol  = actor_rel_tol
        self.critic_abs_tol = critic_abs_tol
        self.ent_abs_tol    = ent_abs_tol
        self.min_steps      = min_steps
        self._w_al = deque(maxlen=patience)
        self._w_cl = deque(maxlen=patience)
        self._w_ec = deque(maxlen=patience)

    def _on_step(self) -> bool:
        super()._on_step()
        log = self.model.logger.name_to_value
        if "train/actor_loss" not in log:
            return True
        self._w_al.append(log["train/actor_loss"])
        self._w_cl.append(log.get("train/critic_loss", np.nan))
        self._w_ec.append(log.get("train/ent_coef", np.nan))
        if len(self._w_al) == self.patience and self.num_timesteps >= self.min_steps:
            a, c, e = map(np.array, (self._w_al, self._w_cl, self._w_ec))
            al_flat = (np.std(a) / (np.abs(np.mean(a)) + 1e-8)) < self.actor_rel_tol
            cl_low  = np.nanmean(c) < self.critic_abs_tol
            ec_low  = np.nanmean(e) < self.ent_abs_tol
            if al_flat and cl_low and ec_low:
                print(f"  [Converged] {self.num_timesteps:,} steps — stopping.")
                return False
        return True


def _make_vec_env(seed: int, domain_rand: bool = True, n_envs: int = None,
                  vecnormalize_path: str = None, reward_weights: dict = None):
    """
    Build the training VecEnv. If `vecnormalize_path` is given, the running
    observation statistics are restored from it instead of started from scratch
    (required for a faithful resume — fresh statistics would silently change the
    observation scaling the policy was trained under).
    """
    from rl.environment import ArmReachEnv
    n_envs = n_envs or config.N_ENVS

    def make_env(s):
        def _f():
            e = ArmReachEnv(domain_rand=domain_rand, seed=s,
                            reward_weights=reward_weights)
            e.reset(seed=s)
            return e
        return _f

    envs = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        envs = VecNormalize.load(vecnormalize_path, envs)
        envs.training = True
    else:
        envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return envs


def train(algo: str = "SAC", out_dir: str = None, seed: int = None,
          total_steps: int = None, early_stop: bool = False,
          resume: bool = True, hyperparams: dict = None,
          reward_weights: dict = None):
    """
    Train `algo` on the arm-reaching task.

    Returns (model, vec_env, callback). Saves model.zip, vecnormalize.pkl and
    train_meta.json (algo, seed, steps, measured wall-clock seconds) into out_dir.

    If `resume` and `<out_dir>/checkpoint/` holds a usable checkpoint, training
    continues from it for the remaining steps instead of starting over. Pass
    `resume=False` to force a clean run.
    """
    algo    = algo.upper()
    seed    = config.SEED if seed is None else seed
    out_dir = out_dir or os.path.join(config.RUNS_DIR, f"{algo}_seed{seed}")
    steps   = total_steps or config.TRAIN_STEPS
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoint")

    ckpt = _load_checkpoint(ckpt_dir) if resume else None

    envs = _make_vec_env(
        seed, domain_rand=True, reward_weights=reward_weights,
        vecnormalize_path=os.path.join(ckpt_dir, "vecnormalize.pkl") if ckpt else None)

    if ckpt:
        # tensorboard_log intentionally disabled — see the fresh-build branch.
        model = algo_registry.REGISTRY[algo]["cls"].load(
            os.path.join(ckpt_dir, "model.zip"), env=envs, device=config.DEVICE)
        rb_path = os.path.join(ckpt_dir, "replay_buffer.pkl")
        if hasattr(model, "load_replay_buffer") and os.path.exists(rb_path):
            model.load_replay_buffer(rb_path)
        done_steps = int(ckpt["num_timesteps"])
        prior_wall = float(ckpt.get("wall_clock_s", 0.0))
        print(f"  [{algo} seed {seed}] RESUMING from {done_steps:,} steps "
              f"({prior_wall/60:.1f} min already spent)")
    else:
        # tensorboard_log intentionally disabled: this repo lives inside a
        # OneDrive-synced folder, and SB3's async TensorBoard event-file writer
        # thread races with OneDrive's sync/placeholder file handling, causing
        # FileNotFoundError crashes mid-training. MetricsCallback reads losses
        # from the in-memory logger (name_to_value) regardless, so no data is
        # lost by not also writing .tfevents files.
        # `hyperparams` overrides the registry defaults — used to train on the
        # parameters an Optuna study selected. Recorded in train_meta.json so a
        # reported run always carries the configuration it was produced with;
        # a tuned result whose parameters are not stored beside it is not
        # reproducible.
        model = algo_registry.build(algo, envs, seed=seed, device=config.DEVICE,
                                    tensorboard_log=None, **(hyperparams or {}))
        done_steps, prior_wall = 0, 0.0

    remaining = max(0, steps - done_steps)

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  [{algo} seed {seed}] policy params: {n_params:,} — "
          f"training {remaining:,} of {steps:,} steps")

    # NOTE: on a resume the loss history (MetricsCallback) restarts empty — it
    # only feeds the diagnostic loss figure, not any reported number.
    cb = (ConvergenceCallback(min_steps=50_000) if early_stop
          else MetricsCallback())

    from rl.environment import ArmReachEnv
    eval_env = DummyVecEnv([lambda: ArmReachEnv(domain_rand=False, seed=seed + 10_000,
                                               reward_weights=reward_weights)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True,
                            norm_reward=False, clip_obs=10.0)
    curve_cb = EvalCurveCallback(eval_env, every=config.EVAL_FREQ, n_episodes=5,
                                 best_dir=os.path.join(out_dir, "best"))
    if ckpt:
        curve_cb.best_err  = float(ckpt.get("best_err", float("inf")))
        curve_cb.best_step = ckpt.get("best_step")
        curve_cb.points = list(ckpt.get("eval_curve", []))
        # Skip the checkpoints already logged so the curve stays on its grid.
        curve_cb._next = (done_steps // config.EVAL_FREQ + 1) * config.EVAL_FREQ

    t0 = time.perf_counter()
    ckpt_cb = CheckpointCallback(ckpt_dir, config.CHECKPOINT_FREQ, envs, curve_cb,
                                 algo, seed, t0, prior_wall_s=prior_wall)
    if remaining > 0:
        model.learn(total_timesteps=remaining,
                    callback=CallbackList([cb, curve_cb, ckpt_cb]),
                    reset_num_timesteps=not bool(ckpt),
                    progress_bar=False)
    wall_s = prior_wall + (time.perf_counter() - t0)

    model.save(os.path.join(out_dir, "model"))
    envs.save(os.path.join(out_dir, "vecnormalize.pkl"))
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump({"algo": algo, "seed": seed, "steps": steps,
                   "wall_clock_s": wall_s, "n_policy_params": n_params,
                   "eval_curve": curve_cb.points,
                   "resumed_from_step": done_steps if ckpt else None,
                   "hyperparams": hyperparams,
                   "reward_weights": reward_weights,
                   "best_err": curve_cb.best_err,
                   "best_step": curve_cb.best_step}, f, indent=2)

    print(f"  [{algo} seed {seed}] done in {wall_s/60:.1f} min")
    return model, envs, cb
