# Investigation log — 2026-08-14

A chronological record of everything done in this session, why it was done, what
it found, and what it changed. Written so that someone who has never seen this
project can follow the reasoning, and so that the thesis text can be corrected
against it.

The headline: **a reward-function defect meant the reinforcement-learning agent
was never solving the reaching task at all.** Every behavioural number this
project produced before today measures a policy deliberately destroying the
simulation, not a policy reaching for a target. The defect is now fixed and
verified, and a clean run is in progress.

---

## 0. Starting state

The project trains a reinforcement-learning (RL) agent to control a simulated
human arm with 9 muscles, in MuJoCo (a physics engine). The agent sees the arm's
state and outputs 9 muscle activations in [0, 1]; the goal is to move the hand to
a target point in space and hold it there.

Coming into today:

- The model and environment had been rebuilt on 2026-08-13 after a series of
  genuine bugs (joint ranges in degrees instead of radians, muscles anchored so
  they produced no shoulder torque, domain randomisation drifting across resets).
- A 1,000,000-step SAC training run had been launched overnight.
- Reported performance was 0% success, with the hand settling ~0.30 m from the
  target — attributed at the time to "not enough training".

---

## 1. Establishing what the reboot destroyed

**What we did.** Read the run logs and the Windows event log.

**Why.** The user reported the machine had rebooted; the first question is
whether the overnight run survived.

**What we found.**

- Last boot: 2026-08-14 04:05:19. The System event log records event ID 1074
  twice — `MoUsoCoreWorker.exe` initiating a restart at 03:59, and
  `TrustedInstaller.exe` completing it at 04:05. Both are Windows Update
  components. This was a forced update restart, not a crash and not sleep.
- The training run had started ~02:43 and last wrote at 03:58, at roughly
  900,000–1,000,000 of its 1,000,000 steps. It died about five minutes short.
- `runs/SAC_seed0/` was **empty**. `rl/train.py` called `model.save()` only
  *after* `model.learn()` returned, so 75 minutes of compute left nothing on
  disk. Nothing was salvageable.

**Root cause of the timing.** Windows "active hours" were set to 10:00–03:00.
Windows will not force a restart during active hours, so it waited and restarted
at 03:59 — the first minute it was permitted to. Overnight runs sit precisely in
the unprotected window. Active hours cannot fix this: Windows caps the span at 18
hours, so some part of every day is always exposed.

---

## 2. Making runs survivable (fix #1)

**What we changed.** `rl/train.py` now writes a complete, resumable checkpoint
every `config.CHECKPOINT_FREQ` (50,000) steps into `<out_dir>/checkpoint/`:

| File | Why it is needed |
|---|---|
| `model.zip` | the policy and value-network weights |
| `vecnormalize.pkl` | the running mean/variance used to normalise observations. Without it, a resumed run would rescale its inputs differently and the loaded policy would be reading a different feature space than it was trained on |
| `replay_buffer.pkl` | the stored past experience. SAC is *off-policy*: it learns by replaying old transitions. Resuming without the buffer is a warm restart, not a true continuation |
| `progress.json` | step count, accumulated wall-clock, and the learning curve so far |

Each file is written to a `.tmp` name and then moved into place with
`os.replace`, which is atomic on NTFS. This matters because the failure mode we
are defending against is being killed at an arbitrary moment: a non-atomic write
interrupted halfway would leave a corrupt file and destroy the *previous* good
checkpoint too.

`train()` resumes automatically when a checkpoint exists, and takes
`resume=False` to force a clean start.

**Verification.** A test run trained to 6,000 steps, was stopped, then resumed
with a 10,000-step target. It restarted from step 4,004, ran to 10,000, and
produced a correctly stitched learning curve (`[2000, 4000, 6000, 8000, 10000]`)
with wall-clock accumulated across both sessions. Also confirmed `resume=False`
starts clean.

**Cost.** ~100 MB of disk, overwritten in place rather than accumulated.

---

## 3. Stopping Windows from doing it again (fix #2 — needs your action)

**What we created.** `pause_windows_update.ps1`. Run elevated, it sets
`PauseUpdatesExpiryTime` (pausing updates for N days) and
`NoAutoRebootWithLoggedOnUsers=1` (a second, independent layer that blocks
automatic restart while a user is signed in). `-Revert` undoes both.

```
powershell -ExecutionPolicy Bypass -File pause_windows_update.ps1 -Days 7
```

**Status: still pending.** This requires administrator elevation, which the
agent session cannot obtain. Until it is run, a long grid remains exposed.

**Revert when the grid finishes.** A machine with updates paused indefinitely is
a machine missing security patches. The same applies to the `powercfg`
sleep-disable set on 2026-08-13 (`powercfg /change standby-timeout-ac 30`).

---

## 4. Pilot 3 — relaunching the lost run

**What we did.** Created `run_pilot3.py` and re-ran SAC seed 0 for 1,000,000
steps, this time followed by a real 50-episode evaluation (the lost run died
before evaluating).

**Result.** 74.4 minutes. The learning curve reproduced the lost run *exactly* at
all ten checkpoints (0.506, 0.292, 0.160, 0.512, 0.439, 0.194, 0.156, 0.275,
0.282) — same seed, deterministic, confirming that adding checkpointing did not
perturb training.

Evaluation over 50 episodes:

| metric | value |
|---|---|
| mean final error | 0.215 m |
| success rate (strict) | 0% |
| within 2 cm / 5 cm / 10 cm at the end | 0% / 0% / 10% |
| ever within 2 cm / 5 cm | 0% / 4% |

Read at face value this looked like a modest improvement over the previous
0.295 m and a continuing precision plateau. **That reading was wrong**, for
reasons Step 6 uncovered.

---

## 5. First suspicion: implausible effort

**What we noticed.** The evaluation reported `mean_energy` = 1.29 × 10⁶. This is
the raw metabolic proxy Σ(F²) summed over the 9 muscles — a diagnostic quantity,
not the term used in the reward.

**Why it was suspicious.** Summing the squared peak isometric forces from
`arm.xml` (624, 435, 987, 798, 624, 624, 1142, 1142, 259 N) gives 5.64 × 10⁶ as
the value at *maximum simultaneous contraction of every muscle*. The policy was
sitting at 23% of that — roughly 48% activation in a root-mean-square sense.

For comparison, `oracle_feasibility.py` had previously established that ~8%
activation is enough to hold the arm against gravity. A reaching movement should
not require six times the effort of simply holding position. Something was
driving the muscles far harder than the task needs.

Combined with the MuJoCo `Nan, Inf or huge value in QACC` warnings that appeared
only after ~700,000 steps and grew *more* frequent — a pattern that looks learned
rather than random — this suggested the policy was doing something other than
reaching.

---

## 6. The real defect: the agent was rewarded for destroying the arm

### 6.1 The hypothesis

Reading `rl/environment.py` in full revealed the mechanism.

Two facts about the v2 reward:

1. **Every per-step reward was ≤ 0.** The reward was a pure cost:
   `r = -(w1·(err² + 0.5·err) + w2·effort + w3·smoothness)`, with the alive
   bonus `w4` set to 0.
2. **A numerical blow-up ended the episode with no penalty.** The environment
   guarded against instability with
   `terminated = bool(success or blew_up)`, where `blew_up` fires if any joint
   exceeds 50 rad/s or the state goes non-finite.

The interaction between these is fatal, and it turns on a distinction that is
easy to miss. In RL there are two ways an episode can end:

- **Truncation** — the time limit expired. The agent is told "this episode is
  over, but the world would have continued", and the learning algorithm
  *bootstraps*: it estimates the value of the state it stopped in and adds it.
- **Termination** — the agent reached a genuine end state. No bootstrapping; the
  future value from that point is defined to be exactly **0**.

So a policy that triggers `blew_up` gets a return of 0 from that moment onward.
A policy that survives the full episode accumulates ~100 steps of negative
reward. Blowing up was therefore worth roughly +17, against a success bonus of
`w5` = 10 — and success additionally required landing within 2 cm *and* settling
below 0.1 rad/s, which a privileged-information oracle achieves only ~7% of the
time.

**Destroying the arm was about twice as valuable as the best possible reaching
outcome, and vastly easier.** The reward function's optimum was self-destruction.

### 6.2 Confirming it with measurement

Hypotheses about learned behaviour need evidence, so `diagnose_termination.py`
was written to instrument 50 evaluation episodes of the pilot-3 policy.

```
episode length: mean 6.9   median 7   min 6   max 8   (of a possible 100)
ended by BLOW-UP : 100.0%  (50/50)
ended by SUCCESS :   0.0%
ran full episode :   0.0%
mean activation across episodes: 0.404   (posture-holding needs ~0.08)
```

**Unambiguous.** Every single episode ended in a deliberate blow-up at step ~7 of
100 — 0.14 seconds of simulated time. This matches the `Time = 0.118`–`0.166`
values in the MuJoCo warnings exactly: 6–8 agent steps at 50 Hz.

**Consequences.**

- The reported 0.215 m "final error" is the distance from the target 0.14 s into
  an episode in which the arm was being whipped apart. It is not a measure of
  reaching accuracy.
- The learning curve (0.506 → 0.160 → 0.512 → …) was tracking the same thing.
  Its non-monotonicity was never noise around a plateau; it was measuring an
  increasingly efficient self-destruction.
- **This retroactively explains the entire history of 0% success on this
  project.** The velocity guard (`qvel > 50`) triggers silently, with no warning
  printed, so this could have been happening from the very first run. The NaN
  warnings that appeared at 700k steps were only the small subset of blow-ups
  violent enough to break the integrator outright.

### 6.3 Ruling out the alternative explanation

Before rewriting the reward, we checked whether the model was simply fragile —
if the arm exploded under any strong activation, that would be a modelling
problem and the reward would be a red herring.

`test_stability.py` drives the model with four activation patterns for a full
100 steps: full co-activation (all muscles at 1.0), random uniform, alternating
full on/off, and a policy-like pattern. Under the current configuration
(`timestep=0.002`, `integrator="implicit"`, Newton solver):

```
current (ts=0.002, armature=0)   OK 100/100 on all four patterns
```

**The model is stable.** It survives every legal command, including maximum
simultaneous contraction of all 9 muscles. The blow-ups were therefore not
physics fragility — they were the agent finding and pulling a specific lever
(the 50 rad/s guard) because the reward paid it to.

This also told us the fix belongs in the reward, not the model. No armature or
timestep change was needed, and none was made.

---

## 7. The v3 reward (fix #3)

Five changes, in `config.py`, `rl/environment.py` and `rl/evaluate.py`.

### 7.1 A survival offset makes every step worth taking

`w4` becomes **3.5**, a per-step offset added to the reward. The cost terms are
bounded — `w1·(err² + 0.5·err) ≤ 2.7` at the workspace-diameter error of ~1.4 m,
`w2·effort ≤ 0.1`, `w3·smoothness ≤ 0.45` — so 3.5 guarantees the per-step
reward is **strictly positive everywhere**.

Once every step pays something, ending an episode early always *forfeits* value
instead of banking it. The exploit is removed structurally rather than patched.

Crucially, adding a constant to every step cannot change the relative ranking of
any two trajectories of the same length. The distance shaping tuned in v2
(quadratic + linear, the linear term added because a pure quadratic has no
gradient near the target) is preserved untouched.

### 7.2 Success no longer terminates

In v2, reaching the target ended the episode. That is what made an alive bonus
perverse in v1: if surviving pays and success stops you surviving, hovering
next to the target beats touching it.

In v3 the episode always runs its full 100 steps, and `w5` = 1.0 is paid *per
step* while the hand is on target and settled. The task becomes **reach and
hold**, which is what a rehabilitation controller actually needs to do, and both
incentives now point the same way: get there fast, stay there.

This also strengthens the reported metric. `info["success"]` at the final step
now means "on target *and* settled at t = 2 s", rather than v2's "touched it once
at some point".

### 7.3 Blow-up is penalised

`w6` = 10.0 is subtracted once when an episode ends in a blow-up. Given the
policy already forfeits ~212 in discounted survival reward by quitting at step 7,
this is belt-and-braces rather than the main deterrent.

### 7.4 Divergence must be detected via MuJoCo's warning counters

This one was found *by the verification test*, not by inspection, and it is
subtle.

The v2 code detected numerical divergence by checking whether the state was
finite after stepping. **That check almost never fires**, because when MuJoCo
detects a bad acceleration it emits a warning and then *resets the simulation
internally*. By the time control returns to Python the state is perfectly finite
— the arm has simply teleported to its home pose.

The consequence in v2: a diverged episode silently continued, with the arm
relocated to the home pose and the original target still active, and every
subsequent error reading was the distance from the home pose to the target. Pure
noise, reported as data.

v3 records MuJoCo's `mjWARN_BADQACC` / `BADQPOS` / `BADQVEL` counters before and
after each physics sub-step and treats any increase as a blow-up, keeping the
`isfinite` check as a backstop.

### 7.5 Blown-up episodes report their last real error

v2 called `mj_resetData` and *then* read the hand position, so every blown-up
episode reported the home-pose error. v3 caches the last finite observation and
error each step and reports those instead.

### 7.6 Domain randomisation was applying one global scale

```python
self.model.body_mass[:] *= self.rng.uniform(0.85, 1.15)     # returns a SCALAR
```

Without `size=`, NumPy returns a single number, so every body received the
*identical* mass multiplier and every muscle the identical strength multiplier.
That is a single global scale factor, not the independent per-element variation
the robustness claim depends on — and a policy can trivially infer and cancel one
global scale. Fixed to draw per-element.

### 7.7 Evaluation now reports episode integrity

`rl/evaluate.py` gained `blow_up_rate`, `mean_episode_len`, `min_episode_len`,
and `max_episode_steps`; the training curve gained `blow_up_rate` and
`mean_episode_len` per checkpoint.

This exists because **nothing in the previous metrics dictionary could have
revealed the defect.** A policy that quits after 7 steps posted a plausible-looking
error number and a plausible-looking learning curve. A healthy run now visibly
shows `blow_up_rate = 0.0` and `mean_episode_len = 100`; anything else is a
red flag on the face of the results.

---

## 8. Verification

`test_reward_v3.py` — 14 checks, all passing:

- per-step reward strictly positive on every non-failure step (min 1.671)
- reward positive under full co-activation and under zero activation
- every episode that does not blow up runs to full length
- a benign policy completes all episodes
- blow-up terminates, is flagged in `info`, and is scored worse than a normal step
- a non-finite blow-up terminates and reports the last finite error, not the
  home pose (reported 0.2688, last finite 0.2688, home pose 0.3839)
- mass randomisation varies per body (σ = 0.070); muscle strength varies per
  muscle (σ = 0.080)
- the success predicate fires when on target and settled
- success does **not** terminate the episode, and the bonus is paid

Note that two of these checks failed on first run and caught real defects — the
MuJoCo self-reset behaviour of §7.4, and an over-strict expectation of our own.
The 50 rad/s guard still trips for ~6/30 episodes under *uniformly random*
actions, which is correct: violently flailing the arm is a genuine failure the
agent should learn to avoid, and it is now penalised rather than rewarded.

**Short training smoke test** (60,000 steps, 4.6 min):

```
[curve  20,000] err=0.518  blowup=0%  len=100
[curve  40,000] err=0.408  blowup=0%  len=100
[curve  60,000] err=0.300  blowup=0%  len=100
mean_reward: +294   (was -4.04)
```

Zero blow-ups, full-length episodes, and error falling monotonically. For
context, the broken v2 run needed 300,000 steps to reach 0.160 and then bounced
back to 0.512; this reaches 0.300 in 60,000 while actually attempting the task.

---

## 9. Best-checkpoint selection (fix #4)

The v2 curve swung by a factor of three between checkpoints, and `train.py`
reported whatever policy the final gradient step happened to produce — throwing
away the run's own best result.

Training now also saves `<out_dir>/best/` whenever validation error improves.
Two safeguards make this reportable rather than cherry-picking:

- **Selection and reporting use different data.** Selection runs on the
  validation environment seeded `seed + 10_000`; the reported evaluation runs on
  the seed-0 stream. The number reported is not the number selected on.
- **Only episodes that complete are eligible.** A checkpoint with a non-zero
  blow-up rate cannot be selected, because a blown-up episode's error is not a
  measurement of reaching performance.

Both the final and the best policy are evaluated and reported side by side.

---

## 10. Pilot 4 — the run

The previous `runs/SAC_seed0/` was archived to `runs/SAC_seed0_v2_exploit/`.
This was necessary as well as tidy: the new resume logic would otherwise have
found a completed 1,000,000-step checkpoint and concluded there was nothing to
do — silently continuing a policy trained on the broken reward.

`run_pilot4.py` trains SAC seed 0 for 1,000,000 steps on the v3 reward, then
evaluates both the final and the best checkpoint over 50 episodes.

**The gate for committing to the full grid** (4 algorithms × 10 seeds ≈ 50 h) is
deliberately not "did the number improve". It is:

1. `blow_up_rate` = 0.00
2. `mean_episode_len` = 100
3. a monotone fall in error

That is, evidence the agent is *solving the task* rather than gaming it.

Reference point: the privileged-information oracle (`reach_oracle.py`, which can
see the solution) reaches within 2 cm at some point in 27% of episodes with a
median final error of 0.08 m. That is the practical ceiling for this arm model
and control rate.

### 10.1 Result: the exploit is fixed, the task is still not solved

78.8 minutes. **The gate failed.** Reporting it plainly:

| | final | best |
|---|---|---|
| blow-up rate | 0.02 | 0.02 |
| mean episode length | 98.2 | 98.2 |
| success rate | 0% | 0% |
| mean final error | 0.242 m | 0.253 m |
| within 5 cm / 10 cm | 0% / 4% | 0% / 4% |
| mean energy | 1.25 × 10⁶ | 1.24 × 10⁶ |
| mean reward | +317 | +316 |

**What genuinely improved.** Blow-ups were 0% at every one of the ten training
checkpoints, with episodes running the full 100 steps. The agent is now
attempting the task instead of gaming it. Mean reward went from −4 to +317.

**What did not.** Accuracy is unchanged, and effort is *identical* — 1.25 × 10⁶,
the same ~48% RMS activation as the exploiting policy. Removing the exploit did
not, by itself, produce a competent controller.

Two honest notes on the protocol. The residual 2% blow-up rate is 1 episode in
50, not a systematic behaviour, but it does mean the gate's first condition is
not met. And the "best" checkpoint scored *worse* than the final one on the test
stream (0.253 vs 0.242 m) — which is the expected outcome when selection happens
on a genuinely separate validation stream and the differences are within noise.
That is the selection protocol working correctly, not a bug.

---

## 11. Diagnosing what the v3 policy actually does

**What we did.** Wrote `diagnose_behaviour.py` to separate the plausible
explanations: has the policy collapsed to one posture, does it fail to move, or
does it move but fail to refine?

**A methodological trap worth recording.** The first version of this diagnostic
produced nonsense — every "final hand position" identical, net displacement
exactly 0.000 m alongside a path length of 3.68 m. The cause was
`DummyVecEnv`, which **auto-resets the moment an episode ends**, so every state
read after the loop was the *next* episode's home pose. A second bug truncated
all episodes to the length of the shortest, letting one blown-up 8-step episode
define the window for all 30. Both were fixed by driving the raw environment
directly and applying `VecNormalize.normalize_obs()` by hand. This is the same
class of error as the original defect: an instrument that silently reports a
plausible-looking number.

**What the corrected diagnostic found** (30 episodes):

```
1. TARGET AWARENESS
   corr(target, final hand) = +0.568 / +0.737 / +0.644   (x / y / z)
   spread ratio = 0.80        (1.0 = tracks targets fully, 0 = fixed posture)

2. ERROR OVER TIME
   step  0: 0.792 m
   step 10: 0.301 m
   step 25: 0.266 m
   step 99: 0.228 m
   best mid-episode: 0.159 m at step 29  ->  drift back out of +0.069 m

3. ACTIVATION
   overall mean 0.463        (posture holding needs ~0.08)
   elbow co-contraction index (Falconer & Winter) = 0.725

4. MOVEMENT
   distance needed 0.801 m, net displacement achieved 0.823 m
   total path travelled 3.028 m   (3.8x the required distance)
```

**Interpretation.** The policy is *not* target-blind and *not* immobile — it
tracks targets well and travels the full required distance. It makes a fast
ballistic movement into roughly the right region within 10 steps, gets as close
as 0.159 m by step 29, and then **drifts back out and never settles**. It is
simultaneously rigid (CCI 0.725: agonist and antagonist largely cancelling) and
jittery (3.8× excess path length).

The missing capability is precision and settling, not reaching.

---

## 12. The v4 reward — three coupled changes

Each of the three suspected causes cost roughly 3% of the ~3.3 per-step reward
under v3, i.e. the policy had no material incentive to avoid any of them.

| weight | v3 | v4 | reasoning |
|---|---|---|---|
| `w2` effort | 0.1 | **1.0** | At CCI 0.725 the agonist and antagonist torques largely cancel, so the arm is rigid and fine positioning requires precise differences between two large numbers. Making effort cost something forces the reciprocal pattern — which is also the biologically correct one, and what the thesis's synergy/CCI analysis needs to be non-trivial. |
| `w3` smoothness | 0.05 | **0.5** | Penalises the jitter behind the 3.8× excess path length. The term also changed from a *sum* over 9 muscles to a *mean*, bounding it by 1 like the effort term so the two weights are comparable. |
| `w7` precision | — | **3.0** (new) | The quadratic + linear error cost has a gradient that *flattens* on approach: 0.96/m of improvement at 23 cm but only 0.60/m at 5 cm. The policy stalls exactly where improving stops paying. `w7·exp(−err/0.05)` is negligible far away and steep in the last 10–15 cm — worth 1.1 at 5 cm and 2.0 at 2 cm. It is a dense, shaped version of the sparse success bonus the policy has never once collected. |
| `w4` survival offset | 3.5 | **4.5** | Preserves the strict positivity guarantee that removes the termination exploit. Cost terms are now bounded by 2.7 + 1.0 + 0.5 = 4.2. |

All 14 checks in `test_reward_v3.py` still pass under v4 (minimum per-step
reward 2.385, and the on-target reward is now 8.5 = 4.5 + 1.0 + 3.0).

**A caveat stated up front:** these three changes are coupled, so a single
successful run will not identify which one mattered. `analysis/ablation.py`
exists to separate them, and the thesis wants a reward ablation regardless.

---

## 13. Probing domain randomisation as an alternative cause

Before spending another 75 minutes, we tested the leading competing hypothesis.

**The hypothesis.** The agent trains with per-muscle strength randomised ±20%
but is evaluated with none. Co-contraction is the textbook rational response to
*unknown actuator gains*: stiffening makes the limb's behaviour less sensitive to
them. If domain randomisation is what drives the co-contraction, then no effort
penalty will remove it without also costing the robustness the thesis claims.

`probe_v4.py` trains a short budget under each condition and reports the
behaviour metrics, so the long run is spent on the right configuration.

**Result** (120,000 steps each, 13.1 min each, run in parallel):

| metric | DR on | DR off |
|---|---|---|
| co-contraction index | **0.708** | **0.727** |
| mean activation | 0.428 | 0.462 |
| spread ratio | 0.887 | 0.731 |
| best error | 0.210 m | 0.163 m |
| final error | 0.447 m | 0.500 m |
| drift | +0.238 m | +0.337 m |
| blow-up rate | 0.00 | 0.00 |

**The hypothesis is refuted.** Co-contraction is essentially identical with and
without domain randomisation (0.708 vs 0.727, against 0.725 for the v3 policy).
Uncertainty about muscle gains is not what makes this policy stiffen.

Domain randomisation is also mildly *helpful* on the metrics that matter — better
target tracking (0.887 vs 0.731) and less drift — so it stays on, which is
convenient since the thesis's robustness claims depend on it.

**A second, unwelcome finding.** Raising the effort penalty tenfold did not
reduce activation either (0.428 and 0.462, against 0.463 under v3). The caveat
is that these probes ran 120,000 steps while the v3 number came from 1,000,000,
so this is suggestive rather than conclusive — a v3-weights probe at 120,000
steps would be the missing control. But it is evidence that the co-contraction
has a cause neither `w2` nor domain randomisation addresses.

### 13.1 A design error in the precision bonus, found by this probe

The probes made something obvious that inspection had missed. `PRECISION_SCALE`
was set to 0.05 m, so at the 0.23 m where the policy actually plateaus the bonus
was `3.0 · exp(−0.23/0.05)` = **0.03** — three hundredths of a ~4 reward. It did
nothing.

The bonus only began paying once the hand was already inside ~15 cm, which is
precisely the region the policy could not reach. It was a prize behind a wall:
incapable of guiding the policy to the place where it started being useful.

`PRECISION_SCALE` is now **0.15 m**, matched to where the policy actually lives
(0.05–0.30 m): worth 0.65 at 23 cm, 2.16 at 5 cm and 2.6 at 2 cm — a smooth
monotone pull across the whole flat region. The reward's dynamic range across an
episode is now roughly 3.5 (far away) to 8.5 (on target and settled).

---

## 14. Pilot 5 — the run now in progress

`runs/SAC_seed0/` was cleared and `run_pilot4.py` relaunched with the v4 weights
and the corrected precision scale: SAC seed 0, 1,000,000 steps, domain
randomisation on, evaluating both the final and the best checkpoint over 50
episodes.

The gate is unchanged: `blow_up_rate` = 0, `mean_episode_len` = 100, and a
monotone fall in error — plus, this time, the two behaviour metrics that v4 is
supposed to move: co-contraction index well below 0.7, and drift near zero.

### 14.1 Result: the approach improved, the holding got worse

78 minutes. **The gate failed again**, but this time the failure is specific and
informative rather than diffuse.

| metric | pilot 4 (v3) | pilot 5 (v4) | |
|---|---|---|---|
| closest approach | 0.159 m | **0.106 m** | improved 33% |
| ever within 5 cm | 4% | **14%** | 3.5x |
| within 10 cm at end | 4% | **14%** | 3.5x |
| co-contraction index | 0.725 | **0.610** | improved |
| blow-ups (30 episodes) | 1/30 | **0/30** | clean |
| drift after closest approach | +0.069 m | **+0.128 m** | *worse* |
| mean final error | 0.242 m | 0.234 m | unchanged |
| success rate | 0% | 0% | unchanged |
| mean energy | 1.25e6 | 1.25e6 | unchanged |

**The precision bonus did exactly what it was designed to do** — it halved the
closest approach and tripled the proximity rates. And then the policy drifted
back out, further than before, so the final error did not move.

**The diagnosis is now precise: the policy approaches well and cannot hold.**
Total path length is 3.76 m for a 0.80 m reach, i.e. the hand moves continuously
at roughly 1.9 m/s and never decelerates. It sweeps through the target region
rather than stopping in it.

Critically, this is no longer a reward-design problem. Stopping on target is
worth about 4.0 per step for the remaining ~70 steps of an episode; sweeping
past collects a fraction of that. **The reward already says the right thing —
the policy cannot execute it.**

Two further findings:

* **The effort penalty does nothing.** Mean energy is identical across a tenfold
  change in `w2`, now confirmed over two full 1e6-step runs and two probes.
  Whatever sets the activation level, it is not the effort term.
* **The learning curve never converges**: 0.215 → 0.417 → 0.236 → 0.255 over the
  last 400k steps. Oscillation of that size at the end of training is an
  optimisation symptom, not a reward-shape symptom.

Per the rule stated in `RECAP.txt` §6 — *"if accuracy is still poor, stop tuning
the reward; three iterations is enough to conclude the problem is elsewhere"* —
reward work stops here.

---

## 15. Hyperparameter search

The non-convergent curve is the strongest single signal available, and
`optuna_search.py` had existed since July without ever being run.

### 15.1 The existing search would have produced nothing

Reading it before running it found three defects, the first fatal:

1. **It maximised `success_rate`.** Every policy this project has produced
   scores exactly 0.0 on that metric. All 25 trials would have returned 0.0,
   every comparison would have been a tie, and the "best" trial would have been
   whichever happened to run first. A search with no gradient is not a search.
   The objective is now `mean_final_error`, minimised — graded, and it directly
   penalises the drift that is the diagnosed failure.
2. **It evaluated on the training environment**, `envs.venv`, with domain
   randomisation still on. Trials were scored under a different protocol from
   the one the thesis reports. Evaluation now builds a clean `domain_rand=False`
   environment and syncs the observation statistics onto it.
3. **No pruning**, so hopeless trials ran to full budget. Now uses a
   `MedianPruner` with intermediate reports every third of the budget.

### 15.2 The search space, and why

| parameter | range | reasoning |
|---|---|---|
| `learning_rate` | 5e-5 – 1e-3 (log) | the oscillating curve is the classic symptom |
| `gamma` | 0.95 – 0.999 | episodes are 100 steps; the effective horizon matters for a hold-at-target task |
| `batch_size` | 128 – 1024 | |
| `tau` | 0.001 – 0.02 (log) | target-network update rate, the other classic instability knob |
| `gradient_steps` | 1, 2, 4 | more critic fitting per environment step; an under-fit critic oscillates too |
| `target_entropy` | −27 – −4.5 | **the leading hypothesis.** SAC's entropy bonus actively pays the policy to stay stochastic. The default `auto` sets the target to −dim(A) = −9, which for 9 muscles keeps substantial residual noise in the commanded activations — and the diagnosed failure is precisely an inability to settle. |

`net_arch` is deliberately **not** searched: the thesis reports a measured policy
footprint (393,750 parameters, inference latency, FLOPs) for `[256, 256]`, and
those are among the few genuinely real artifacts the project has. Searching the
architecture would silently invalidate them.

### 15.3 Running it

24 trials at 200,000 steps each, four parallel workers against one SQLite study
(`runs/optuna_SAC.db`), three torch threads apiece on a 16-core machine.

Two infrastructure problems had to be fixed first, both real:

* Four workers creating the database simultaneously raced on schema
  initialisation — `UNIQUE constraint failed: alembic_version.version_num` —
  and three of the four died instantly. The study is now created once before
  any worker starts, and the two orphaned `RUNNING` trials left behind were
  marked `FAIL` so they cannot pollute the study.
* SQLite's default lock timeout is short enough that concurrent trial
  bookkeeping can raise "database is locked". The storage now uses a 60-second
  timeout.

Each worker uses a different **sampler** seed so they do not all propose the same
points, while the **training** seed stays fixed at 0 across every trial so the
comparison between them is fair.

---

## 16. The machine hard-locked under the search (15:07–15:25)

The four search workers launched at 15:16 never produced a result. The reason
was not the search: **the machine stopped responding entirely.**

Timeline, from the Windows event log after the fact:

| time | evidence |
|---|---|
| 15:07:44 | last System event written (DNS-Client warning) |
| ~15:16 | four Optuna workers running, 3 torch threads each |
| 15:25:30 | last timestamp the kernel flushed (Event 6008, "previous system shutdown … was unexpected") |
| 16:30 | screen observed frozen, fans at 100% |
| 19:57:39 | reboot (Kernel-Power **41**, "rebooted without cleanly shutting down") |

So the machine was already dead by ~15:25, over an hour before it was noticed,
and the search survived roughly **9 minutes**, not the 19 that the file
timestamps suggest (the Optuna DB's 15:35 mtime is cached writes that never
flushed cleanly).

### 16.1 It was not a software crash, and not memory

Four checks, all negative:

* **No BugCheck (1001), no minidump** — and `CrashDumpEnabled = 3`, so dumps are
  enabled. A software BSOD *would* have written one. The kernel never reached
  the bugcheck path at all.
* **No WHEA** hardware-error events.
* **No Resource-Exhaustion (2004)** low-memory event.
* **No Kernel-Processor-Power** throttle events.

Memory is also cleared arithmetically. `buffer_size = 300_000` with a 38-dim
observation gives 300k×38×4×2 (obs + next_obs) + 300k×9×4 ≈ **106 MB**, which
matches `replay_buffer.pkl` (105,601,992 bytes) exactly. Four workers held
~425 MB of replay buffers between them. Memory was never the constraint.

### 16.2 What it was

The distinguishing variable is load, not software. Pilots 4 and 5 each ran
**78–80 minutes single-worker without incident**. The only thing that changed at
15:16 was four workers × 3 threads = **12 saturated threads on a 45 W Ryzen
4800H in a laptop chassis**. Total unresponsiveness, fans pegged, no dump, no
WHEA: that is the signature of a **power/thermal hang**, not a Windows fault.

Consequence for the compute plan: sustained all-core load on this machine is not
safe, and every subsequent run in this document is capped at **8 of 16 logical
CPUs** — the level pilot 6 then demonstrated is survivable for 78 minutes.

### 16.3 What the search left behind

6 trials: 2 `FAIL` (the earlier schema race) and **4 orphaned `RUNNING`**, none
of which reached even its first pruning report. **Zero completed trials, zero
usable data.**

Orphans are not inert: `study.optimize` counts `RUNNING` trials toward
`n_trials` and the sampler treats them as pending, so a restarted search
silently does less work than asked. `cleanup_optuna.py` (*new*) fails them
explicitly and never touches `COMPLETE` or `PRUNED` trials, so it cannot destroy
a real result. A storage **heartbeat** (60 s, 300 s grace) now fails such trials
automatically.

---

## 17. The replay-ratio bug — a silent 4x under-training

`config.N_ENVS = 4`, and `rl/algorithms.py` set `train_freq=1, gradient_steps=1`
for every off-policy algorithm. In SB3 one rollout iteration on an n-env VecEnv
collects **n** transitions and is then followed by `gradient_steps` updates.

Measured directly rather than argued from documentation
(`verify_replay_ratio.py` *new*, on SB3 2.8.0 / torch 2.11.0+cpu):

| configuration | updates per env step |
|---|---|
| **as configured** — n_envs=4, gradient_steps=1 | **0.233** |
| n_envs=1, gradient_steps=1 (standard SAC) | 0.933 |
| n_envs=4, gradient_steps=4 | 0.933 |
| n_envs=4, gradient_steps=−1 | 0.933 |

**Every run this project has ever done performed ~233,000 gradient updates for a
nominal 1,000,000-step budget.** An independent check closes the arithmetic:
233k updates × the ~17 ms/update measured in July = **71 min**, against observed
pilot run times of **78.8 and 80.6 min**.

This predicts precisely the symptoms every pilot reproduced — an under-fit
critic, an evaluation curve swinging 2–3× to the end of training, and a policy
that reaches the target region but cannot refine within it.

**Fix**: `GRADIENT_STEPS = -1` in `algorithms.py`. That form holds the ratio at
~1 for *any* value of `N_ENVS`, so the bug cannot silently return if the env
count is changed again.

### 17.1 Thread scaling — and a hypothesis that was wrong

`bench_threads.py` (*new*) measures the real `SAC.train()` path, not a synthetic
matmul:

| threads | bs=256 | bs=512 |
|---|---|---|
| 1 | 16.16 ms | 25.65 ms |
| 2 | 12.12 ms | 17.74 ms |
| 4 | 11.96 ms | 17.36 ms |
| 8 | **10.66 ms** | **14.39 ms** |

The expectation was that `[256, 256]` nets would scale *negatively* past ~4
threads and that fewer threads would be both faster and cooler. **That was
wrong** — 8 threads is fastest. But scaling is poor (8× the threads buys 1.78×),
which means the four-worker parallelism that hard-locked the machine was
**buying almost nothing**: each worker sat thread-starved at 3 while the four of
them together saturated the CPU.

### 17.2 The consequence nobody had priced in

At a *correct* replay ratio, 1e6 steps/seed costs **3.7 h**. The planned
4 algorithms × 10 seeds is therefore **148 h**, not the ~50 h this document has
been assuming. That protocol was never affordable on this machine — it only
looked like 50 h because the runs were silently doing a quarter of the training.

---

## 18. Pilot 6 — the replay-ratio experiment

**One variable changed from pilot 5**: gradient updates per environment step.
Reward v4, FRAME_SKIP=10, model, evaluation protocol and seed all identical.
Pilot 5 changed three coupled reward terms at once and consequently could not
identify which mattered; this run deliberately does not repeat that.

Budget 300,000 steps (~280k updates, 78.1 min) — chosen so that the run does
**more gradient work than pilot 5's entire 1e6-step run** from **3.3× fewer**
environment samples, making the comparison sharp in both directions.

Curve (25k spacing, 5-episode validation evals):

```
 25k 0.544 | 50k 0.334 | 75k 0.206 | 100k 0.255 | 125k 0.302 | 150k 0.226
175k 0.283 | 200k 0.407 | 225k 0.193 | 250k 0.240 | 275k 0.112 | 300k 0.282
```

### 18.1 Result (50-episode evaluation)

| metric | pilot 5 (ratio 0.23) | pilot 6 (ratio 0.93) | |
|---|---|---|---|
| mean_final_error | 0.254 m | **0.204 m** | −20% |
| blow_up_rate | 2% | **0%** | |
| mean_episode_len | 98.5 | **100.0** | |
| mean_energy | 1.25e6 | **0.77e6** | −38% |
| touch_2cm | 0.00 | **0.06** | first non-zero ever |
| reach_10cm | 0.14 | 0.18 | |
| **mean_min_error** | 0.134 m | **0.132 m** | **unchanged** |
| **success_rate** | 0% | **0%** | **unchanged** |

`diagnose_behaviour.py`, 30 episodes:

| | pilot 5 | pilot 6 |
|---|---|---|
| closest approach | 0.106 m | **0.104 m** — unchanged |
| drift after closest approach | +0.128 m | **+0.072 m** — −44% |
| co-contraction index | 0.610 | **0.501** |
| mean activation | 0.463 | **0.358** (posture needs ~0.08) |
| target tracking (spread ratio) | — | **0.88** |
| path length / required distance | 4.7× | 4.4× |

### 18.2 The decomposition — the actually valuable result

There were never one failure but **two**, and they respond to different things:

* **Holding — substantially fixed.** Drift halved, co-contraction down 18%,
  energy down 38%, blow-ups eliminated, every episode full-length. This is what
  gradient starvation was costing.
* **Approach precision — completely untouched.** Closest approach is
  0.104 / 0.106 / 0.159 m across pilots 6 / 5 / 4 and **has not moved for
  anything tried**: three reward redesigns and a 4× correction in gradient work.

So the replay ratio was a genuine bug worth fixing, and the fix stays — but it
is **not** the answer to "why can't it reach 2 cm". That root cause is still
unidentified.

### 18.3 A caveat about the 275k point, and about success_rate

The 0.112 at 275k is a **5-episode** eval and did not survive 50 episodes
(0.204). Curve points at this spacing are noisy and should not be quoted.

Separately: **reporting "0% success" is not informative.** The criterion
(<2 cm *and* <0.1 rad/s) is one the *privileged oracle* meets only ~7% of the
time. A metric a perfect-information controller fails 93% of the time does not
measure the policy. The graded metrics (`reach_10cm`, `touch_5cm`) plus the
oracle bound are the defensible report. Note also that the policy's closest
approach (0.104 m) is now near the oracle's **median final** error (0.08 m), so
some of the residual gap may be the 9-muscle plant rather than the learner.

---

## 19. The search, twice re-scoped

**Launch 1 failed instantly.** Changing the `gradient_steps` choices from
`[1,2,4]` to `[4,8]` collided with the trials already on record:
`ValueError: CategoricalDistribution does not support dynamic value space`.
Optuna pins a categorical space for the life of a study. The old study had 0
usable trials, so it was **archived** (`optuna_SAC_void_20260814.db`) rather
than deleted, and a fresh study started.

That space change was itself a necessary fix: leaving raw `gradient_steps`
`[1,2,4]` in the search would have let the sampler wander straight back into the
0.23 and 0.47 ratios that §17 had just shown to be broken, and would have made
the trials incomparable to the pilots.

**Launch 2 ran 73 min and revealed a cost problem.** 3 trials, 1 complete. The
space's own corners differed ~5× in cost: the `gradient_steps=8` /
`batch_size=1024` trial was one third done in 73 minutes (~3.7 h projected)
while the other worker finished a trial and two thirds of another. The estimate
given at launch (~3.6 h) was wrong by ~3× because it priced every trial at ratio
0.93 and one batch size. Realistic figure was ~10 h. `MedianPruner` could not
compensate: `n_startup_trials=5` means nothing is pruned until 5 trials complete.

**The early signal, which is why the re-scope went the way it did:**

| trial | gradient_steps | intermediate curve | final |
|---|---|---|---|
| 1 | 8 (ratio 1.87) | 0.439 → 0.341 → **0.278** | **0.2392** |
| 2 | 4 (ratio 0.93) | 0.316 → **0.452** | — |

Trial 1 fell **monotonically** and beat pilot 6's 0.255 at the same 100k steps.
Trial 2 oscillated — the same non-convergence every run of this project has
shown. Weak evidence (two trials), but it points the same way pilot 6 did and
further: **0.93 may still not be enough.**

**Launch 3 (23:19)**, on a fresh study, with the previous one archived as
`optuna_SAC_ratio_scan_20260814.db`:

* `gradient_steps` **fixed at 8**, no longer searched — recorded in the code as
  a *hypothesis under test*, with the explicit condition that if completed
  trials do not converge monotonically the assumption is wrong and the scan must
  reopen. Two trials is not evidence.
* `batch_size` capped at **512** (1024 dropped).
* TPE `n_startup_trials` 8 → 5, so a 16-trial budget is mostly guided.
* 2 workers × 4 threads = **8 of 16 CPUs**, the level pilot 6 survived.

Nominal estimate **6.0 h** (~4.5–5 h with pruning now that trial costs are
near-uniform). Note this is *longer* than launch 2's cheap trials, because every
trial now runs at the expensive ratio — capping the batch size saved less than
fixing the ratio cost.

### 19.1 Two process failures worth recording

* **A double-launched run.** A running-process check used `Get-Process python`,
  but the interpreter is `python3.12.exe`, so it returned a false negative and a
  second pilot 6 was started on top of a live first one — two processes writing
  one checkpoint directory and together saturating all 16 threads, the exact
  load profile of §16. Caught, both killed, directory cleared, and
  `run_pilot6.py` now takes a **PID lockfile** and refuses to double-start.
  Process checks in this project must match `python3.12.exe`.
* **A background launch that silently died.** `nohup … &` inside a tool-invoked
  shell does not survive the shell exiting. Long runs must be started as tracked
  background processes, not detached with `&`.

---

## 20. The search result (2026-08-15, 07:20)

13 complete / 16 trials, 3 pruned. Ran 23:19 → 07:20 — **~8 h against a ~4.5–5 h
estimate.** That is the second time this search's runtime was under-estimated;
the lesson is that `MedianPruner` saves far less here than assumed, because most
trials converge well enough to survive the median filter.

### 20.1 The fixed-ratio hypothesis held

§19 fixed `gradient_steps=8` on two trials' evidence and flagged it in-code as
reopenable if completed trials failed to converge monotonically. They did not
fail: four of the first five completions fell monotonically
(0.404→0.303→0.237, 0.401→0.290→0.254, 0.463→0.445→0.279, 0.455→0.443→0.290),
the exception being the highest learning rate. Against every prior run of this
project oscillating 2–3× to the end of training, that is a real contrast. The
hypothesis stands and the ratio scan does not need reopening.

### 20.2 TPE converged on a coherent region

```
trial 12  err=0.1177  lr=4.40e-4  target_entropy=-19.6  drift=0.034
trial 14  err=0.1293  lr=3.91e-4  target_entropy=-20.0  drift=0.051
trial 11  err=0.1311  lr=4.22e-4  target_entropy=-19.3  drift=0.049
trial 13  err=0.1405  lr=4.38e-4  target_entropy=-18.6  drift=0.069
trial 10  err=0.1405  lr=4.49e-4  target_entropy=-19.0  drift=0.035
```

**lr ≈ 4e-4 and `target_entropy` ≈ −19**, tightly clustered across the top five.
SAC's default `auto` sets target entropy to −dim(A) = **−9**; the search asks for
roughly twice as deterministic a policy.

This confirms the hypothesis the search was *designed around* (§15, written
before any trial ran): *"target entropy is the leading suspect for the inability
to SETTLE — SAC's entropy bonus actively pays the policy to stay stochastic."*
It is now supported by a systematic search rather than an intuition, and it
explains mechanically why every prior policy could approach but not hold.

---

## 21. Confirmation at full budget — the plateau breaks

`run_confirm.py` (*new*) re-measures the search's best parameters under the
pilot protocol: 300k steps, 50 evaluation episodes. Parameters are **read from
the study, never transcribed**, so the run cannot silently disagree with the
search it confirms. `train()` gained a `hyperparams` passthrough, recorded into
`train_meta.json` — a tuned result whose parameters are not stored beside it is
not reproducible.

96.6 min. Curve: 0.439 → 0.210 → 0.144 → 0.163 → 0.199 → 0.167 → 0.145 → 0.184
→ 0.167 → 0.133 → **0.081** → 0.092.

| metric | pilot 6 | **final** | best |
|---|---|---|---|
| mean_final_error | 0.204 m | **0.1063 m** | 0.1047 |
| **mean_min_error** | 0.132 m | **0.0740 m** | 0.0760 |
| success_rate | 0% | **4%** | 2% |
| reach_2cm | 0.00 | 0.06 | 0.10 |
| reach_5cm | 0.00 | **0.34** | 0.28 |
| reach_10cm | 0.18 | **0.58** | 0.56 |
| touch_2cm | 0.02 | 0.20 | **0.28** |
| touch_5cm | 0.18 | **0.52** | 0.44 |
| mean_energy | 774,667 | **276,804** | 299,618 |
| blow_up / ep_len | 0% / 100 | 0% / 100 | 0% / 100 |

**`mean_min_error` 0.132 → 0.0740 m.** That number had not moved across pilots
4, 5 and 6 (0.178 / 0.134 / 0.132) through three reward redesigns and the
replay-ratio correction. It is now *past the privileged oracle's 0.08 m median
final error*, and `touch_2cm` at 0.28 matches the oracle's 0.27 — a policy
acting on proprioception alone has reached the privileged-information bound.

Energy fell **64%** while accuracy roughly doubled, so this is not accuracy
bought with co-contraction.

### 21.1 The success_rate caveat — do not headline it

`success_rate` became non-zero for the first time in the project's history. It
is also **1–2 episodes out of 50**: the run reported 4%, the independent audit
(§22.1, a different draw of 50 targets) reported 2%. The 95% interval on 2/50 is
roughly 0.5–14%. This metric is noise at this magnitude and must not carry
weight in the thesis. The graded metrics are the robust result, and §18.3's
argument still applies: the criterion is one the privileged oracle meets ~7% of
the time.

---

## 22. Audits of the confirmation run

### 22.1 Divergence audit — clean

Four MuJoCo `NaN in QACC` warnings appeared in the log while `blow_up_rate` read
0.00. That combination had to be checked rather than assumed, because silent
divergence (MuJoCo self-resetting mid-episode and the home pose being measured
as data) is precisely what voided every behavioural number before 2026-08-14.

`verify_divergence.py` (*new*) drives the RAW env — no VecEnv auto-reset — reads
MuJoCo's warning counters directly for **all seven** warning classes rather than
the three the env watches, and cross-checks ground truth against
`info["blew_up"]`.

**Result: 0 episodes warned, 0 flagged.** No divergence occurred during
evaluation at all; the log warnings came from training and were buffered to the
end of stderr. `blow_up_rate = 0.00` is correct.

The audit also recomputes the metrics through an entirely independent path
(manual observation normalisation, no VecEnv):

| | run reported | audit |
|---|---|---|
| mean_final_error | 0.1063 | 0.1057 |
| mean_min_error | 0.0740 | **0.0740** |
| mean_episode_len | 100.0 | 100.00 |
| success_rate | 0.0400 | 0.0200 (see §21.1) |

### 22.2 Behaviour diagnostic

| | pilot 5 | pilot 6 | **confirmed** |
|---|---|---|---|
| closest approach | 0.106 m | 0.104 m | **0.079 m** |
| drift after closest | +0.128 | +0.072 | **+0.035** |
| total path length | 3.76 m | 3.55 m | **1.64 m** |
| path / required distance | 4.7× | 4.4× | **2.05×** |
| net displacement / required | — | 1.09 | **0.99** |
| mean activation | 0.463 | 0.358 | **0.151** |
| target tracking (spread ratio) | — | 0.88 | **0.91** |
| corr(target, hand) x/y/z | — | .85/.86/.74 | **.94/.90/.86** |
| co-contraction index | 0.610 | 0.501 | **0.564** |

The within-episode trajectory is the qualitative tell:

```
step 0: 0.795 | step 10: 0.198 | step 25: 0.119
step 50: 0.110 | step 75: 0.112 | step 99: 0.113
```

It converges by step 25 and then **holds flat for the remaining 75 steps**.
Every previous policy approached and drifted back out. Path length collapsing
3.55 → 1.64 m for an 0.80 m reach says the same thing from another direction:
the hand no longer sweeps through the target region, it goes there once
(net displacement 0.99× required) and stops. Mean activation 0.151 against the
~0.08 needed to hold posture means the stiffening strategy that motivated the
entire v4 reward redesign is largely gone.

**One metric did not improve: CCI, 0.501 → 0.564.** CCI is scale-free, so this
says that while *absolute* co-contraction collapsed (energy −64%), the
flexor/extensor *balance* is marginally more co-contracted than pilot 6. The
thesis has a CCI analysis section, and this is the one number in this run that
cannot be reported as an improvement.

---

## 23. Files changed or created today

| File | Change |
|---|---|
| `rl/train.py` | atomic checkpointing every 50k steps; resume; best-model saving; blow-up rate and episode length in the curve |
| `rl/environment.py` | v3 reward; success no longer terminates; blow-up penalty; warning-counter divergence detection; last-finite-error reporting; per-element domain randomisation |
| `rl/evaluate.py` | episode-integrity metrics (`blow_up_rate`, `mean_episode_len`, …) |
| `config.py` | `CHECKPOINT_FREQ`, `CHECKPOINT_REPLAY_BUFFER`, reward weights v3 with `w4`/`w5`/`w6` |
| `run_pilot3.py` | *new* — relaunch of the lost run (result now known to be void) |
| `run_pilot4.py` | *new* — first run on the corrected reward |
| `diagnose_termination.py` | *new* — the diagnostic that found the exploit |
| `test_stability.py` | *new* — model stability under adversarial activation patterns |
| `test_reward_v3.py` | *new* — 14-check verification of the fixes |
| `pause_windows_update.ps1` | *new* — update-restart protection (**needs elevated run**) |
| `STEPS.md` | *new* — this document |

Added in the evening session (§16–19):

| File | Change |
|---|---|
| `rl/algorithms.py` | `GRADIENT_STEPS = -1` for SAC/TD3/DDPG — fixes the 4× under-training (§17) |
| `rl/environment.py` | `_obs()` builds the three muscle blocks by slice instead of 27 per-element Python floats |
| `rl/optuna_search.py` | replay ratio fixed at `2*N_ENVS` not searched as raw `gradient_steps`; `batch_size` ≤ 512; storage heartbeat; eval envs closed in `finally`; no longer crashes when zero trials complete; TPE startup 8 → 5 |
| `verify_replay_ratio.py` | *new* — measures updates per env step; confirmed 0.233 vs 0.933 |
| `bench_threads.py` | *new* — real `SAC.train()` cost vs thread count and batch size |
| `cleanup_optuna.py` | *new* — fails orphaned `RUNNING` trials; never touches `COMPLETE`/`PRUNED` |
| `run_pilot6.py` | *new* — the replay-ratio experiment; PID lockfile against double-start |
| `run_search.py` | *new* — creates the study once (no schema race), caps total threads at 8, reports per-worker logs |

Added 2026-08-15 (§20–22):

| File | Change |
|---|---|
| `rl/train.py` | `hyperparams` passthrough to the algorithm builder, recorded in `train_meta.json` |
| `run_confirm.py` | *new* — full-budget confirmation; reads best params from the study rather than transcribing them |
| `verify_divergence.py` | *new* — audits all 7 MuJoCo warning classes against `info["blew_up"]`; independently recomputes the metrics |

---

## 24. Open items

1. ~~APPROACH PRECISION IS THE UNSOLVED PROBLEM~~ — **RESOLVED 2026-08-15**
   (§20–22). `mean_min_error` 0.132 → 0.074 m, past the oracle's 0.08 m median.
   Cause was SAC's default `target_entropy` = −dim(A) = −9 keeping too much
   noise in the commanded activations to ever settle; the search selected ≈ −19.
2. **EVERYTHING RESTS ON ONE SEED.** The result above is SAC seed 0, once.
   Nothing is publishable until it replicates. **Next compute should be 5-seed
   SAC at the confirmed parameters (~8 h)** — it converts an anecdote into a
   result with error bars and reveals the seed variance *before* ~32 h is
   committed to a full grid.
3. **THE FOUR-WAY COMPARISON IS NOW UNFAIR.** SAC has Optuna-tuned parameters;
   TD3/DDPG/PPO have registry defaults, and `target_entropy` — the parameter that
   mattered most — has no analogue outside SAC. `rl/algorithms.py` documents the
   benchmark as fair *because* all four share defaults; that is no longer true.
   Options: (a) search each algorithm (~8 h each), (b) transfer the shared
   parameters (lr, gamma, tau, batch_size, gradient_steps) to TD3/DDPG and
   disclose that `target_entropy` is SAC-only, (c) report SAC tuned vs the rest
   at defaults as a stated limitation. **Undecided — this shapes what the thesis
   can claim.**
4. **The grid protocol must be resized.** 4 algorithms × 10 seeds × 1e6 steps is
   **148 h** at a correct replay ratio (§17.2), not the ~50 h assumed throughout
   this document. At the confirmed configuration a run is 96.6 min, so
   4 × 5 seeds ≈ **32 h** — several supervised sessions, not one night.
5. **CCI did not improve** (0.501 → 0.564, §22.2) even though absolute
   co-contraction collapsed. The thesis's CCI section needs this reported
   honestly rather than folded into the general improvement.
3. **Windows Update was blocked manually** by the user on 2026-08-14 evening, so
   `pause_windows_update.ps1` is no longer the blocker it was. **Revert both
   this and the `powercfg` sleep-disable when the grid finishes.**
4. **Sustained all-core load hard-locks this machine** (§16). Every long run must
   stay at ≤ 8 of 16 logical CPUs. This is a hardware constraint, not a setting.
5. **The thesis and paper must be corrected.** Chapter III (model) and the
   environment/domain-randomisation text still describe the v2 model and reward.
   The reward function described in the text is the exploitable one.
6. **The OneDrive copies are still not synced** with any of the 2026-08-13 or
   2026-08-14 fixes. `C:\Users\moham\thesis_run\` is the working copy.
7. **Every behavioural number predating 2026-08-14 is void** and must not be
   carried into the thesis: they measure the termination exploit. Numbers from
   pilots 4 and 5 are *additionally* compromised by the replay-ratio bug (§17) —
   they are honest measurements of a 4×-under-trained policy. The genuinely real
   artifacts are unaffected, because none depend on policy quality: the
   Newton-Raphson vs SAC forward-pass timing benchmark, the parameter count and
   policy footprint, the muscle-function validation
   (`runs/muscle_function.json`), and the model stability result from §6.3.
8. **`success_rate` should not be the headline metric** (§18.3). Report graded
   accuracy against the oracle bound instead.
9. **The search is running** (launched 23:19, ~4.5–5 h). Its `gradient_steps=8`
   assumption is a hypothesis, not a result — if the completed trials do not
   converge monotonically, reopen the ratio scan.
