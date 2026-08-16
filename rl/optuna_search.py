"""
Optuna hyperparameter search — real, inspectable, archived.

Runs an Optuna study for one algorithm, each trial training on a reduced budget.
The study is persisted to SQLite (runs/optuna_<algo>.db) so the search space and
every trial are a real artifact, inspectable with
`optuna-dashboard sqlite:///runs/optuna_<algo>.db`.

Only after this study exists may the thesis/paper describe the hyperparameters as
"tuned via Optuna" — until then they are the hand-set defaults in algorithms.py.

REWRITTEN 2026-08-14. The previous version had three defects that would have
made the search worthless:

  1. It maximised `success_rate`. Every policy this project has ever produced
     scores exactly 0.0 on that metric — it requires landing within 2 cm AND
     settling below 0.1 rad/s, which even a privileged oracle manages only ~7%
     of the time. All 25 trials would have returned 0.0, every comparison would
     have been a tie, and the "best" trial would have been whichever ran first.
     The objective is now `mean_final_error` (minimised), which is graded and
     therefore actually carries a gradient for the sampler.
  2. It evaluated on `envs.venv` — the TRAINING environment, with domain
     randomisation still on. Trials were scored under a different protocol from
     the one the thesis reports. Evaluation now builds a clean domain_rand=False
     env and syncs the observation statistics onto it.
  3. No pruning, so hopeless trials ran to full budget.

`net_arch` is deliberately NOT searched. The thesis reports a measured policy
footprint (393,750 parameters, inference latency, FLOPs) for the [256, 256]
architecture, and those are among the few genuinely real artifacts the project
has. Changing the architecture would silently invalidate them.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import optuna
import config


def _suggest(trial, algo: str) -> dict:
    algo = algo.upper()
    common = dict(
        learning_rate=trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
        gamma=trial.suggest_float("gamma", 0.95, 0.999),
    )
    if algo in ("SAC", "TD3", "DDPG"):
        common.update(
            # 1024 dropped 2026-08-14. It is ~2.3x the cost of 256 per update
            # (bench_threads.py) and, combined with the doubled replay ratio
            # below, produced a single trial projected at 3.7 h against ~40 min
            # for the rest of the space — one corner was consuming a worker for
            # the length of the whole search.
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            tau=trial.suggest_float("tau", 0.001, 0.02, log=True),
            # REPLAY RATIO, expressed as gradient_steps = ratio * config.N_ENVS.
            #
            # This must NOT be searched as a raw gradient_steps value. SB3 does
            # `gradient_steps` updates per rollout iteration, and one iteration
            # collects N_ENVS transitions, so raw values [1, 2, 4] at N_ENVS=4
            # mean ratios of 0.23, 0.47 and 0.93 — the first two being exactly
            # the 4x under-training bug that verify_replay_ratio.py confirmed
            # and pilot 6 measured the cost of (final error 0.254 -> 0.204 and
            # drift 0.120 -> 0.072 on fixing it). Searching them again would let
            # the sampler wander back into a known-broken regime and would make
            # the trials incomparable to the pilots.
            #
            # Ratios below 1 are therefore excluded by construction.
            #
            # FIXED, not searched (2026-08-14). The first scan over
            # {N_ENVS, 2*N_ENVS} gave an early but consistent split: the one
            # trial that completed at 2*N_ENVS (ratio 1.87) fell MONOTONICALLY
            # 0.439 -> 0.341 -> 0.278 -> 0.2392, beating pilot 6's 0.255 at the
            # same 100k steps, while the ratio-0.93 trial oscillated
            # 0.316 -> 0.452 — the same non-convergence every run of this
            # project has shown. Pilot 6 already established that 0.23 -> 0.93
            # helps; spending half the trials re-confirming that 0.93 is worse
            # than 1.87 buys nothing. Holding it fixed puts the whole budget on
            # the parameters whose effect is genuinely unknown.
            #
            # This is a hypothesis under test, not an established result: two
            # trials is not evidence. If the search's completed trials do not
            # converge monotonically at this ratio, the assumption is wrong and
            # the scan should be reopened.
            gradient_steps=2 * config.N_ENVS,
        )
    if algo == "SAC":
        # Target entropy is the leading suspect for the inability to SETTLE.
        # SAC's entropy bonus actively pays the policy to stay stochastic; the
        # default "auto" sets the target to -dim(A) = -9, which for a 9-muscle
        # action space keeps a lot of residual noise in the commanded
        # activations. The diagnosed failure is precisely that the hand reaches
        # 0.106 m and then drifts back out by 0.128 m instead of holding.
        common.update(
            target_entropy=trial.suggest_float("target_entropy", -27.0, -4.5),
        )
    if algo == "PPO":
        common.update(
            n_steps=trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
            clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
        )
    return common


def _make_eval_env(train_envs, seed: int):
    """A clean evaluation env matching the REPORTED protocol, not the training one.

    Domain randomisation off, rewards unnormalised, observation statistics copied
    from the training env so the policy sees the feature scaling it was trained
    under.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rl.environment import ArmReachEnv

    env = DummyVecEnv([lambda: ArmReachEnv(domain_rand=False, seed=seed + 10_000)])
    env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False,
                       clip_obs=10.0)
    env.obs_rms = train_envs.obs_rms
    return env


class _PruneCallback:
    """Periodic intermediate scoring so hopeless trials die early."""

    def __init__(self, trial, model, train_envs, seed, every, eval_episodes=8):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def __init__(cb):
                super().__init__(verbose=0)
                cb._next = every

            def _on_step(cb) -> bool:
                if cb.num_timesteps < cb._next:
                    return True
                cb._next += every
                from rl.evaluate import evaluate
                # Closed in a finally: this fires several times per trial and
                # dozens of times per search, and each env holds its own MjModel
                # and MjData. Leaking them was survivable for one worker and is
                # not something to leave in place on a machine that has already
                # hard-locked once under this search.
                env = _make_eval_env(train_envs, seed)
                try:
                    m = evaluate(cb.model, env, n_episodes=eval_episodes, seed=seed)
                finally:
                    env.close()
                trial.report(m["mean_final_error"], cb.num_timesteps)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return True

        self.cb = _CB()


def search(algo: str = "SAC", n_trials: int = 25, trial_steps: int = 200_000,
           eval_episodes: int = 20, seed: int = 0,
           sampler_seed: int = 0) -> dict:
    from rl.train import _make_vec_env
    from rl.evaluate import evaluate
    from rl import algorithms as algo_registry

    algo = algo.upper()
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(config.RUNS_DIR, f'optuna_{algo}.db')}"
    # Several workers write to this one file. SQLite's default lock timeout is
    # short enough that concurrent trial bookkeeping can raise "database is
    # locked" and kill a worker mid-search; 60 s is far longer than any write
    # here takes.
    # heartbeat_interval/grace_period added 2026-08-14. When a worker dies
    # without unwinding, its trial is left in state RUNNING forever: optimize()
    # counts those toward n_trials and the sampler treats them as pending, so a
    # restarted search silently does less work than asked. This has now happened
    # twice — once on a schema race, once when the machine hard-locked under
    # four workers — leaving 6 trials of which 0 were usable. With a heartbeat,
    # a trial whose worker has been silent for grace_period is failed
    # automatically instead of needing cleanup_optuna.py afterwards.
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 60}},
        heartbeat_interval=60,
        grace_period=300,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
    )
    study = optuna.create_study(
        study_name=f"{algo}_arm_reach",
        direction="minimize",              # mean final reaching error, metres
        storage=storage, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        # sampler_seed differs per parallel worker so concurrent workers do not
        # all propose the same points; `seed` (the TRAINING seed) stays fixed at
        # 0 across trials so the comparison between them is fair.
        # 8 -> 5 startup trials: with gradient_steps now fixed rather than
        # searched the space is one dimension smaller, and 8 random draws out of
        # a 16-trial budget would leave only half the search actually guided.
        sampler=optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=5),
    )

    def objective(trial):
        params = _suggest(trial, algo)
        envs = _make_vec_env(seed, domain_rand=True)
        model = algo_registry.build(algo, envs, seed=seed, device=config.DEVICE,
                                    **params)
        pruner = _PruneCallback(trial, model, envs, seed, every=trial_steps // 3)
        model.learn(total_timesteps=trial_steps, callback=pruner.cb,
                    progress_bar=False)

        eval_env = _make_eval_env(envs, seed)
        try:
            m = evaluate(model, eval_env, n_episodes=eval_episodes, seed=seed)
        finally:
            eval_env.close()
            envs.close()

        # Record the whole picture, not just the objective, so the study itself
        # is a reportable artifact and a trial that wins for the wrong reason
        # (e.g. by not moving at all) is visible.
        for k in ("success_rate", "mean_min_error", "reach_5cm", "reach_10cm",
                  "touch_5cm", "blow_up_rate", "mean_episode_len", "mean_energy"):
            trial.set_user_attr(k, float(m[k]))
        trial.set_user_attr("drift", float(m["mean_final_error"] - m["mean_min_error"]))
        return float(m["mean_final_error"])

    study.optimize(objective, n_trials=n_trials)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    # study.best_value raises if nothing completed — which is exactly the state
    # this study was left in after the 2026-08-14 hard lock (6 trials, 0
    # complete). A search that produced no result must say so, not crash.
    if not done:
        print(f"  [Optuna {algo}] {len(study.trials)} trials, NONE COMPLETED — "
              f"no result. Check for orphaned RUNNING trials (cleanup_optuna.py).")
        return {"best_value": None, "best_params": None,
                "n_trials": len(study.trials), "n_complete": 0,
                "storage": storage_url}
    print(f"  [Optuna {algo}] {len(done)} complete / {len(study.trials)} trials; "
          f"best mean_final_error={study.best_value:.4f} m  "
          f"params={study.best_params}")
    return {"best_value": study.best_value, "best_params": study.best_params,
            "n_trials": len(study.trials), "n_complete": len(done),
            "storage": storage_url}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="SAC")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler-seed", type=int, default=0,
                   help="differs per parallel worker; training seed stays --seed")
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    # Limit thread contention: several workers share this machine.
    import torch
    torch.set_num_threads(a.threads)
    search(algo=a.algo, n_trials=a.trials, trial_steps=a.steps, seed=a.seed,
           sampler_seed=a.sampler_seed)
