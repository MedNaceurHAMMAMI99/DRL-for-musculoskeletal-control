# Model & environment corrections — 2026-08-13

Everything below was found by direct diagnosis (`diagnose_env.py`,
`validate_muscles.py`) and fixed before the full-scale grid was run. These
changes must be reflected in the thesis model chapter (III) and the paper's
environment/DR description — the previous prose describes the v1 behavior,
which was broken in ways that made the task unlearnable.

## Why every v1 policy scored 0% — root causes, in order of severity

1. **Joint ranges were interpreted in degrees.** `arm.xml` did not set
   `<compiler angle="radian"/>`, and MJCF's default unit is degrees. The
   intended ranges (e.g. elbow `0 2.7` rad ≈ 155°) became 0–2.7 **degrees**:
   every joint was locked to <3° of travel. Measured consequence: the end
   effector could move ~4 cm total, while the target box required ~40–70 cm.

2. **No muscle could actuate the shoulder.** All nine muscles ran
   humerus→forearm. Under shoulder rotation both attachment points move
   rigidly together, so tendon length never changes → zero moment arm about
   all three shoulder DOFs, structurally. The three "deltoids" acted only on
   the elbow. The shoulder was a free, damped, unactuated pendulum.

3. **Elbow muscles had ~zero moment arm at full extension.** Straight-line
   origin→insertion paths passed through the joint axis (all sites at x≈0),
   so flexion could not be initiated from the hanging start pose.

4. **The first observation of every episode was garbage.** `reset()` read
   `site_xpos` after `mj_resetData` without `mj_forward`, so the end effector
   appeared at the world origin. With the (removed) `err > 0.60 m` failure
   cutoff and the true hanging pose 0.87 m below the shoulder, ~78% of
   episodes terminated on step 1 (measured).

5. **Domain randomisation compounded across resets.** `reset()` multiplied
   the live `model.body_mass` (and gain/damping) in place each episode
   without restoring nominals — a multiplicative random walk. Measured: body
   mass at 1.3% of nominal after 1000 resets.

6. **The reward's energy term was 4–5 orders of magnitude too large.**
   `w2·ΣF²` = 0.005 × O(10⁵–10⁶) ≈ 10³–10⁴ per step vs. a reaching term of
   O(0.3). Doing nothing was near-optimal (this exactly matches the v1
   observation of PPO collapsing to zero activation).

7. **The alive bonus made success a losing move.** With success-termination,
   `w4=0.1`/step forever beats a one-time `w5=0.5`: hovering next to the
   target out-earned reaching it.

8. **The target box was partly outside the reachable workspace** and was
   sampled independently of the arm's kinematics.

## Fixes (v2)

**arm.xml** (v1 archived as `arm_v1_broken.xml`):
- `<compiler angle="radian" autolimits="true"/>`.
- Deltoids re-anchored to thorax (worldbody) sites; biceps/triceps long heads
  made biarticular (thorax origins), matching their anatomical scapular
  origins. Elbow flexors/extensors given via points just above the elbow →
  ~2–3 cm moment arms at full extension.
- Flexion axes `(0 -1 0)` so positive q = flexion toward +x, consistent with
  anterior muscle placement. `shoulder_rot` given small passive stiffness
  (no muscle drives it).
- Actuator `lengthrange` measured over the joint workspace
  (`measure_lengthranges.py`), padded ×0.7/×1.3.
- Unchanged from v1: segment lengths, masses, joint ranges, per-muscle peak
  isometric forces (Holzbaur 2005), timestep/solver settings.
- Verified by `validate_muscles.py` (all 7 checks pass; artifact:
  `runs/muscle_function.json`): each muscle alone produces directionally
  correct joint motion; flexors flex, extensors extend, deltoid ant/med/post
  produce shoulder flexion/abduction/extension; co-activation stable.

**rl/environment.py**:
- Nominal model parameters cached at construction and restored every reset;
  DR scales from nominals (drift after 1000 resets: 0.2%, was 98.7%).
- `mj_forward` before the first observation.
- Targets sampled by FK from random joint configurations within limits —
  reachable by construction; `workspace_scale` still supported for the
  robustness study (scales the offset from the shoulder).
- Error-based failure termination removed (success and velocity-guard
  termination kept; time truncation at 1000 steps = 2 s).
- Reward: effort term normalised per muscle by peak isometric force
  (`mean((F/F_max)²)`, O(1)); `w4=0`; terminal success bonus `w5=10`.
  Raw `ΣF²` still logged in `info["energy"]` as the reported metabolic
  metric. New weights in `config.REWARD_WEIGHTS` (w1=1, w2=0.1, w3=0.05,
  w4=0, w5=10).
- Post-fix reward-term magnitudes (random policy, measured): err² ≈ 0.31,
  effort ≈ 0.024, smoothness ≈ 0.075 — same order, error-dominant.

**rl/baselines.py**: densify `actuator_moment` via `mju_sparse2dense`
(MuJoCo ≥ 3.2 stores it sparse; the PD baseline crashed otherwise).

**config.py**: full-scale protocol — SEEDS 0–9, TRAIN_STEPS 1e6,
EVAL_EPISODES 50, SYNERGY_EPISODES 20.

## Round 2 (2026-08-14): the underactuated shoulder

The v2 model above trained (SAC, 1e6 steps) to a plateau of ~0.36 m final
error, 0% success, with a flat learning curve after 100k. Diagnosis, by
static-equilibrium analysis rather than more training (`oracle_feasibility.py`):

- At 200 sampled target postures, solve for the muscle activations that
  cancel gravity + passive torque, and report the residual per joint.
- Result: shoulder flexion, abduction and elbow held **perfectly** (median
  residual 0.00–0.02 N·m against 1.3–3.2 N·m of gravity, using only ~8%
  activation — the muscles are strong enough with large headroom).
- But **shoulder_rot was uncancellable**: 0.34 N·m median residual, holdable
  in only 59% of postures. No muscle in the modeled set spans shoulder
  internal/external rotation — that is the rotator cuff (subscapularis,
  infraspinatus, teres minor), which is not among the 9 modeled muscles.
  The arm was underactuated; the resulting sag displaces the hand by up to
  ~0.19 m, an error floor no policy can beat.

**Fix**: `shoulder_rot` constrained near neutral (range ±0.05 rad, stiffness
200, damping 20) and targets sampled at neutral rotation. The DOF is retained
rather than deleted so the state vector keeps its documented 4-joint / 38-dim
form (which the real, already-measured latency/FLOPs/footprint artifacts
depend on). The task lives in the 3-DOF space the 9 muscles do control —
shoulder flexion, shoulder abduction, elbow flexion — which is exactly enough
to position the hand anywhere in R³. Tendon length-ranges re-measured over
the new workspace. Post-fix feasibility: 94–100% of postures held on all
three controlled DOFs, zero residual; the residual rotation torque is absorbed
by the constraint (≈0.5° deflection ≈ 5 mm at the hand).

### Is the success criterion attainable? (`reach_oracle.py`)

A privileged-information controller (joint-space PD + gravity compensation,
given the target's generating joint configuration, mapped to non-negative
activations by bounded least squares) reaches **within 2 cm at some point in
27% of episodes** (minimum error 2–15 mm), median final error 0.08 m — but
satisfies the strict criterion (< 2 cm AND joint speed < 0.1 rad/s) in only
~7%, because it overshoots and oscillates against muscle activation lag.
The gains are hand-set, not tuned, so this is a loose bound — but it
establishes that the targets are genuinely reachable and that *settling*, not
reaching, is what the strict criterion demands.

**Consequence for reporting**: `rl/evaluate.py` now records graded accuracy —
`reach_2cm/5cm/10cm` (final-step position error) and `touch_2cm/5cm`
(closest approach), alongside the strict `success_rate`. Reporting only a
criterion that a privileged oracle meets 7% of the time would hide every
between-algorithm difference. The strict rate remains the headline; the
graded rates make the benchmark informative.

## Round 3 (2026-08-14): control rate and episode count

With the plant corrected, SAC at 1e6 steps learned — error fell 0.540 → 0.259
by 300k, versus a flat 0.48/0.50/0.38 on the broken model — but then
oscillated (0.27–0.45) and finished at 0.295 m mean final error, 0% strict
success, 0% within 10 cm, 4% ever within 5 cm. Two measurements explain it:

1. **The policy regressed to the workspace centroid** (`diagnose_policy.py`).
   It does use the target — target/settle correlations +0.53, +0.39, +0.25 on
   x/y/z, all positive — but its settle spread is only **39% of the target
   spread**. It has learned a coarse, heavily damped mapping from goal to
   posture: the classic signature of too little experience, not of a
   structural inability to see the goal.

2. **The budget bought only ~1000 episodes.** At one action per physics step,
   a 2 s episode is 1000 agent steps, so 1e6 timesteps is ~1000 reaching
   attempts spread over the whole reachable workspace.

**The fix is nearly free.** Timing (same script): a physics step costs
**6.8 us**, while a SAC gradient update costs ~17 ms — so wall-clock is set
almost entirely by gradient updates, not simulation (the 72.7 min run spent
~0.1 min of it on physics). Introducing action repeat therefore buys episodes
at almost no cost. `FRAME_SKIP = 10`: the policy acts at **50 Hz** while
physics stays at 500 Hz, an episode is 100 agent steps, and the same 1e6-step
budget now buys **~10,000 reaching attempts**. It also shortens the credit
assignment horizon to match the episode (gamma 0.99 -> ~100 steps), and 50 Hz
is both the physiologically sensible rate for a motor command and standard
MuJoCo RL practice (Gym locomotion uses frame_skip 4–5).

Also added: a divergence guard. Training logged
`NaN/Inf in QACC at DOF 0` — MuJoCo can diverge under extreme co-activation.
The episode now terminates on a non-finite state instead of propagating NaN
into the policy.

Note for the documents: the control rate (50 Hz) and episode length (2 s,
100 agent steps) are reportable protocol parameters and must appear in the
experimental setup; effort is averaged over each held interval.

## Consequences for the documents

- Chapter III / paper section describing the model must be updated to the v2
  routing (thorax anchors, biarticular long heads, via points) and must not
  describe v1's humerus→forearm deltoids.
- The DR description stays accurate (same factors), but add: "sampled fresh
  from nominal parameters each episode".
- The reward description must use the v2 form and weights.
- `runs/muscle_function.json` is a real, citable model-validation artifact
  (solo-activation direction tests). The Holzbaur/Murray moment-arm
  comparison (`analysis/model_validation.py`) should be re-run on v2.
- The v1→v2 diagnosis itself is honest, valuable thesis content: it
  mechanistically explains why naive task/model design fails and documents
  the verification methodology.
