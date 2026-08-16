# Thesis compute package — run the RL benchmark on a powerful machine

Self-contained copy of the training/evaluation/analysis pipeline. Copy this whole
folder to a strong machine (GPU recommended), run the grid, then copy the `runs/`
folder back for analysis. **No other files from the thesis repo are needed** — the
MuJoCo model (`arm.xml`) and the measured artifacts (`experiments_results.json`) are
bundled here, and the code auto-detects them.

---

## 1. Install (once)

```bash
python -m venv venv
# Windows:  venv\Scripts\activate     Linux/Mac:  source venv/bin/activate

# GPU: install a CUDA PyTorch build FIRST (see requirements.txt), then:
pip install -r requirements.txt
```

## 2. Choose the device

```bash
# GPU machine:
export RL_DEVICE=cuda        # Windows PowerShell:  $env:RL_DEVICE="cuda"
# (default is cpu if unset)
```

## 3. Pilot FIRST — measure real per-seed time (do not skip)

Training time is the whole reason for using a strong machine. Measure it before
committing to a grid:

```bash
python -c "from rl.train import train; train('SAC', seed=0, total_steps=300000)"
cat runs/SAC_seed0/train_meta.json          # -> "wall_clock_s": <seconds for 1 seed>
```

Then size the grid:  `total_grid_time ≈ wall_clock_s × N_algos × N_seeds`.
Also confirm the policy is actually LEARNING — evaluate the pilot:

```bash
python -c "from rl.evaluate import load_model, evaluate; m,e=load_model('runs/SAC_seed0','SAC'); print(evaluate(m,e,n_episodes=20))"
```

If success rate is still ~0 after 3e5 steps, raise the step budget (or debug the
reward) **before** spending compute on a full grid — otherwise you get a grid of zeros.

## 4. Configure the grid — edit `config.py`

```python
SEEDS       = [0, 1, 2, 3, 4]     # ≥5 for a meaningful Wilcoxon test
TRAIN_STEPS = 1_000_000           # set from the pilot; ≥3e5, ideally 1e6
ALGORITHMS  = ["SAC", "TD3", "DDPG", "PPO"]
EVAL_EPISODES    = 50             # you can restore the full protocol here
SYNERGY_EPISODES = 20
```

Report whatever `SEEDS`/`TRAIN_STEPS` you actually run — the numbers are read back
from `results.json`; never assert a larger n than was executed.

## 5. Run the full grid

```bash
python experiment_runner.py --stage all --robustness --ablation
```

This trains every (algorithm, seed), evaluates them, runs the PD/Random baselines,
the robustness sweep, the reward ablation, and the synergy/CCI/statistics analyses,
then writes everything to **`runs/results.json`**. It is resumable: a run whose
`model.zip` already exists is not retrained, so you can stop/restart safely.

Optional extras:
```bash
python -m analysis.model_validation        # force-length + moment-arm characterization
python -c "from rl.optuna_search import search; search('SAC', n_trials=25)"   # HPO study
```

## 6. What to copy back

Copy the entire **`runs/`** folder back to the thesis machine. At minimum it must
contain `runs/results.json`. Including the per-seed subfolders (`SAC_seed0/…` with
`model.zip`, `vecnormalize.pkl`, `train_meta.json`) lets further analysis (extra
robustness conditions, re-evaluation) be done later without retraining.

```
runs/
├── results.json              <- REQUIRED (all numbers)
├── model_validation.json     <- if you ran step 5 extras
├── SAC_seed0/  {model.zip, vecnormalize.pkl, train_meta.json}
├── SAC_seed1/  ...
└── ...                       (one folder per algorithm × seed)
```

Place that `runs/` folder into `2-code/thesis_pipeline/runs/` on the thesis machine;
`python experiment_runner.py --stage report` then renders the real LaTeX tables and
figures, and the thesis auto-includes them.

---

## Notes & tips
- **Determinism:** every run is seeded; re-running the same config reproduces results.
- **Speed:** `N_ENVS=4` uses `DummyVecEnv` (sequential). On a many-core box you can get
  a real data-collection speedup by switching to `SubprocVecEnv` in `rl/train.py`
  (`_make_vec_env`) — optional, not required.
- **GPU note:** the networks are small (38→256→256→9); MuJoCo physics stepping is
  CPU-bound, so a GPU helps the gradient updates more than the rollouts. A fast
  multi-core CPU is often competitive for these off-policy algorithms.
- **Time budget:** at 1e6 steps/seed, expect ~tens of minutes to ~1 h+ per seed
  depending on hardware; plan the grid accordingly.
- **Integrity:** nothing here fabricates results. If a run is undertrained, its
  success rates will be low — report them as measured.
