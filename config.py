"""
Central configuration for the thesis pipeline.

Everything that a reader/reviewer would need to reproduce the reported numbers
is defined here in one place: the anatomical parameters, the reward weights, the
per-algorithm hyperparameters, the seed grid, and the training/evaluation budget.

IMPORTANT — reproducibility contract
------------------------------------
No number reported in the thesis or paper is hand-authored. Every behavioural
result is produced by `experiment_runner.py`, which trains the policies defined
here and writes real per-seed metrics to `runs/results.json`. The documents read
their tables/figures from that file. See `RUNBOOK.md`.
"""

import os

_HERE = os.path.dirname(__file__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# MuJoCo model: prefer a local copy next to this file (portable compute package);
# otherwise fall back to the in-repo location under appendix_run.
_LOCAL_XML = os.path.join(_HERE, "arm.xml")
ARM_XML = _LOCAL_XML if os.path.exists(_LOCAL_XML) else \
    os.path.normpath(os.path.join(_HERE, "..", "appendix_run", "arm.xml"))
FIGURES_DIR = os.path.join(_HERE, "figures")
RUNS_DIR    = os.path.join(_HERE, "runs")
RESULTS_JSON = os.path.join(RUNS_DIR, "results.json")

# ── Muscles ───────────────────────────────────────────────────────────────────
# Actuator order in arm.xml (index 0..8). Used by every analysis that needs a
# per-muscle correspondence (synergies, CCI, force fidelity).
MUSCLE_NAMES = [
    "biceps_long", "biceps_short", "brachialis",
    "triceps_long", "triceps_lat", "triceps_med",
    "deltoid_ant", "deltoid_med", "deltoid_post",
]
N_MUSCLES = len(MUSCLE_NAMES)

# Agonist / antagonist grouping for the co-contraction index (elbow flexion/
# extension is the dominant DOF of the reaching task).
#   Flexors  (elbow):  biceps_long, biceps_short, brachialis
#   Extensors(elbow):  triceps_long, triceps_lat, triceps_med
CCI_AGONIST_IDX    = [0, 1, 2]   # flexors
CCI_ANTAGONIST_IDX = [3, 4, 5]   # extensors

# ── Anatomical parameters (Holzbaur 2005) ─────────────────────────────────────
BICEPS_LONGUS = dict(
    L_M0   = 0.116,   # optimal fibre length (m)
    L_S0   = 0.272,   # slack tendon length (m)
    alpha0 = 0.0,     # pennation angle at L_M0 (rad), zero for biceps
    F_max  = 624.0,   # maximum isometric force (N)
)

# ── Reward weights ────────────────────────────────────────────────────────────
# v2 (2026-08-13) — rebalanced after the env audit (see rl/environment.py
# docstring): the energy term is now normalised per muscle by peak isometric
# force (O(1) instead of O(1e4), which had made "do nothing" the optimal
# policy), the alive bonus is removed (it made hovering beat succeeding), and
# the success bonus is terminal and large. Hand-set values; a planned Optuna
# search (optuna_search.py) may tune them — do not describe them as tuned in
# the text unless runs/optuna_<algo>.db exists.
# v3 (2026-08-14) — restructured after diagnose_termination.py showed the v2
# reward was catastrophically exploitable. Under v2 every per-step reward was
# <= 0 and a numerical blow-up set terminated=True with no penalty, so the
# return from a blow-up was exactly 0 while surviving the episode cost ~-20.
# Blowing the arm up was worth roughly twice the w5=10 success bonus, and the
# policy learned exactly that: 100% of evaluation episodes ended in a
# deliberate blow-up at step ~7 of 100. See STEPS.md.
#
# Two coupled changes remove the exploit:
#   * w4 is a per-step SURVIVAL OFFSET large enough that the per-step reward is
#     strictly positive at any reachable error (worst case ~3.2, see the bound
#     in environment.py). Ending an episode early now forfeits real value. This
#     is a constant shift: it leaves the relative ranking of all non-terminating
#     trajectories untouched, so the v2 distance shaping keeps its tuning.
#   * success no longer TERMINATES the episode; w5 is paid per step while the
#     hand is at the target and settled. In v2, ending on success is what made
#     an alive bonus perverse (hovering beat succeeding). With reach-and-hold
#     there is no such conflict: holding the target IS the task, and the two
#     incentives now point the same way.
# w6 is a one-off penalty on blow-up, belt-and-braces on top of the forfeited
# survival reward (~212 discounted at step 7).
# v4 (2026-08-14, after diagnose_behaviour.py on the pilot-4 policy).
# v3 fixed the exploit — the agent now attempts the task, tracks targets
# (spread ratio 0.80) and travels the full required distance — but it stalls at
# ~0.23 m and cannot refine: closest approach 0.159 m at step 29, then drift
# back out to 0.228 m, 3.03 m of path for a 0.80 m reach, co-contraction index
# 0.725, mean activation 0.463. Precision, not reaching, is what is missing.
#
# Three causes, all of which were nearly free under v3 (each cost ~3% of the
# ~3.3 per-step reward, so the policy had no reason to avoid any of them):
#   * w2 (effort) 0.1 -> 1.0. Co-contracting at CCI 0.72 makes agonist and
#     antagonist torques cancel: the arm goes rigid, and fine positioning then
#     requires precise differences between two large numbers. Making effort cost
#     real forces the reciprocal activation pattern, which is also the
#     biologically correct one and what the thesis's synergy/CCI analysis needs.
#   * w3 (smoothness) 0.05 -> 0.5, and the term is now a MEAN over muscles
#     rather than a sum, so it is bounded by 1 like the effort term instead of
#     by 9. Penalises the jitter behind the 3.8x excess path length.
#   * w7 (precision bonus) NEW. The v3 error cost is quadratic + linear, so its
#     gradient FLATTENS as the hand approaches: 0.96/m of improvement at 23 cm
#     but only 0.60/m at 5 cm. The policy stops improving exactly where the
#     reward stops paying. w7*exp(-err/0.05) restores a steep gradient in the
#     last 20 cm (worth 1.1 at 5 cm, 2.0 at 2 cm) — a dense, shaped version of
#     the sparse success bonus that the policy has never once collected.
#
# w4 rises to 4.5 to keep the strict positivity guarantee that removes the
# termination exploit: the cost terms are now bounded by 2.7 + 1.0 + 0.5 = 4.2.
# These three changes are coupled, so which one mattered is not identifiable
# from one run — analysis/ablation.py can separate them, and the thesis wants a
# reward ablation regardless.
REWARD_WEIGHTS = dict(
    w1 = 1.0,     # reaching error (quadratic + linear, see environment.py)
    w2 = 1.0,     # effort: mean_i (F_i / F_max,i)^2, dimensionless O(1)
    w3 = 0.5,     # action smoothness: mean_i (delta a_i)^2, bounded by 1
    w4 = 4.5,     # per-step survival offset — makes per-step reward > 0
    w5 = 1.0,     # per-step success bonus (reach AND hold), no longer terminal
    w6 = 10.0,    # one-off penalty for a blow-up termination
    w7 = 3.0,     # precision bonus, exp(-err / PRECISION_SCALE)
)
# Length scale of the precision bonus (m). Originally 0.05, which was a design
# error: at the 0.23 m where the policy actually plateaus, exp(-0.23/0.05) =
# 0.010, so the bonus contributed 0.03 of a ~4 reward and did nothing at all.
# The bonus only started paying once the hand was already inside ~15 cm — a
# region the policy could not reach, so it could not be guided into it.
# 0.15 matches the scale to where the policy actually lives (0.05-0.30 m):
# worth 0.65 at 23 cm, 2.16 at 5 cm, 2.6 at 2 cm — a smooth monotone pull
# across the whole flat region rather than a prize behind a wall.
PRECISION_SCALE = 0.15

# ── Experiment grid ───────────────────────────────────────────────────────────
# The four DRL algorithms compared in the thesis. All share the SAME Gymnasium
# environment (rl/environment.py) and the SAME evaluation protocol.
ALGORITHMS = ["SAC", "TD3", "DDPG", "PPO"]

# Non-learning baselines (cheap; no training).
BASELINES = ["PD", "Random"]

# Seed grid — FULL-SCALE protocol (2026-08-13). The reported n and step budget
# are whatever is set here and are read back from results.json — never asserted
# independently. The grid is run in resumable waves (seeds 0-4 first, then 5-9);
# a run whose model.zip exists is never retrained.
SEEDS = list(range(10))

# Per-run training budget (timesteps), same for every algorithm (fair
# comparison). Set after the 2026-08-13 model/env fixes; the SAC pilot at 3e5
# steps gates the grid (see RUNBOOK.md) — do not burn the full grid before the
# pilot shows the task is actually being learned.
TRAIN_STEPS = 1_000_000

# Evaluation.
EVAL_EPISODES  = 50     # deterministic rollouts per trained policy
EVAL_FREQ      = 100_000  # log a learning-curve checkpoint every N steps
SYNERGY_EPISODES = 20   # rollouts whose activation traces feed NMF/CCI
N_SYNERGIES    = 4      # k for NMF (d'Avella 2006 upper-limb reference)

# ── Control rate ──────────────────────────────────────────────────────────────
# Physics runs at 500 Hz (timestep 0.002 in arm.xml); the policy acts every
# FRAME_SKIP physics steps (action repeat), i.e. at 50 Hz. Two reasons, both
# measured on 2026-08-14:
#  * Episode count. At one action per physics step, a 2 s episode is 1000
#    agent steps, so a 1e6-step budget buys only ~1000 reaching attempts and
#    the policy regressed to the workspace centroid (settle spread was 39% of
#    target spread — diagnose_policy.py). At 50 Hz the same budget buys
#    ~10,000 attempts.
#  * It is free. A physics step costs 6.8 us, while a gradient update costs
#    ~17 ms, so wall-clock is set by updates, not simulation: 10x the physics
#    adds ~1 min to a ~72 min run.
# 50 Hz is also the physiologically sensible rate for a motor command, and
# matches standard MuJoCo RL practice (Gym locomotion envs use frame_skip 4-5).
FRAME_SKIP   = 10
EPISODE_SEC  = 2.0
MAX_EPISODE_STEPS = int(EPISODE_SEC / (0.002 * FRAME_SKIP))   # 100 agent steps

# ── Crash resilience ──────────────────────────────────────────────────────────
# A forced Windows-Update reboot on 2026-08-14 destroyed a ~900k-step SAC run
# (75 min of compute, ~5 min short of finishing) because train.py only saved
# after learn() returned. Training now writes a resumable checkpoint every
# CHECKPOINT_FREQ steps, so an interruption costs at most one interval.
# Saving the replay buffer is what makes an off-policy resume faithful rather
# than a warm restart; it costs ~100 MB of disk (overwritten in place, not
# accumulated). Set to False only if disk-constrained.
CHECKPOINT_FREQ           = 50_000
CHECKPOINT_REPLAY_BUFFER  = True

# ── Runtime ───────────────────────────────────────────────────────────────────
N_ENVS = 4
SEED   = 42          # global seed for non-grid utilities (benchmark, figures)
# Training device. Override without editing code:  set RL_DEVICE=cuda on a GPU
# machine (or RL_DEVICE=cpu to force CPU).
DEVICE = os.environ.get("RL_DEVICE", "cpu")

# ── Backward-compatibility aliases ────────────────────────────────────────────
# Older modules import these names directly.
MAX_STEPS = TRAIN_STEPS
SAC = dict(
    learning_rate   = 3e-4,
    buffer_size     = 300_000,
    batch_size      = 512,
    learning_starts = 4_000,
    tau             = 0.005,
    gamma           = 0.99,
    train_freq      = 1,
    gradient_steps  = 1,
    ent_coef        = "auto",
    target_entropy  = "auto",
    net_arch        = [256, 256],
)
