# Algorithm comparison — what we can and cannot claim

Drafting notes for the thesis's four-algorithm chapter. Written 2026-08-15.
Kept separate from `STEPS.md` (a debugging log) because this is argument, not
chronology.

---

## 1. The claim the thesis currently makes is unsupported

**This project has no evidence that SAC outperforms TD3, DDPG or PPO.**

The only four-way comparison ever executed was the July smoke test at 12,000
steps per seed. In it:

* all four algorithms scored **0% success**,
* every Wilcoxon comparison **tied at p = 1.0**,
* the sole difference that appeared was PPO collapsing to near-zero activation,

and it ran under a reward later proven exploitable (the agent was paid to
destroy the arm — `STEPS.md` §6) on a model whose joint ranges were locked in
degrees (`MODEL_CHANGES.md`). **That data is void.** The grid has never been run
on working code.

If the thesis or paper states that SAC outperformed the others with statistical
significance, that is placeholder text of the same category as the numbers
audited on 2026-07-18 — not a finding. It is the most exposed claim in the
document, because a reviewer checks the headline comparison first.

---

## 2. The *a priori* case for SAC — theory and literature, not our result

These are reasons to *expect* SAC to do well on this task class. They are not
measurements from this work and must not be written as if they were.

**Off-policy replay.** SAC, TD3 and DDPG reuse stored experience; PPO discards
its batch each iteration. On this task a gradient update costs ~14 ms while a
physics step costs ~7 µs (`bench_threads.py`), so learning is bounded by
gradient work, not simulation — and replay intensity proved decisive here:
raising the replay ratio 0.23 → 0.93 → 1.87 produced monotone gains
(`STEPS.md` §17, §18, §20). PPO cannot exploit that mechanism at all.

**Entropy-regularised exploration in a redundant action space.** Nine muscles
actuate four joints, so many activation patterns produce the same endpoint
force. A stochastic policy explores that redundancy manifold in a way the fixed
additive Gaussian noise of DDPG/TD3 does not.

**Twin critics.** SAC and TD3 both mitigate the value overestimation DDPG is
notoriously brittle to. This predicts DDPG last and TD3 as SAC's closest
competitor — a prediction the benchmark can test.

---

## 3. The interesting result is the opposite of the usual story

**SAC's entropy mechanism was this project's problem, not its advantage.**

SAC sets `target_entropy = -dim(A)` by default, i.e. **-9** for nine muscles.
That default is what prevented the policy from ever settling: it kept enough
residual noise in the commanded activations that the hand approached the target
and then drifted back out. Every run for weeks showed the same signature —
closest approach ~0.10 m, then drift, `success_rate` exactly 0 — and it survived
three complete reward redesigns and a 4x correction to the gradient budget.

The Optuna search selected **target_entropy ≈ -19**, roughly twice as
deterministic as the default, tightly clustered across its top five trials
(`STEPS.md` §20.2). At that setting `mean_min_error` went 0.132 → 0.074 m,
past the privileged oracle's 0.08 m median, and the within-episode error
converged by step 25 and held flat to step 99 instead of drifting.

The defensible, mechanistic claim this supports:

> SAC requires its exploration temperature tuned well below the standard
> `-dim(A)` heuristic for high-dimensional, muscle-redundant control. At the
> default the failure presents as a precision plateau that is easily
> misdiagnosed as insufficient training or as a reward-shaping problem.

That claim is ours, it is falsifiable, and the diagnostic trail supporting it is
recorded. It is worth more than a leaderboard.

**Corollary that must be tested, not assumed:** TD3 and DDPG have no
`target_entropy`; they explore with fixed action noise (σ = 0.1 in
`rl/algorithms.py`). They may not suffer this failure mode at all, and could
therefore look *better* than untuned SAC. Nobody has checked.

---

## 4. The fairness problem

SAC now has Optuna-tuned parameters; TD3, DDPG and PPO have registry defaults.
`rl/algorithms.py` documents the benchmark as fair *because* all four share
defaults — that is no longer true.

A comparison of tuned SAC against default baselines measures **tuning effort**,
not algorithms, and will be read that way.

Options, in ascending order of rigour:

1. Report SAC tuned vs the rest at defaults, disclosed as a stated limitation.
2. Transfer the shared parameters (lr, gamma, tau, batch_size, gradient_steps)
   to TD3/DDPG and disclose that `target_entropy` is SAC-only.
3. **Tune the exploration parameter of every algorithm on equal budget** —
   `target_entropy` (SAC), noise σ (TD3/DDPG), `ent_coef` (PPO). This is the
   cleanest defence, because it gives every algorithm the same treatment on the
   one axis this work has shown to be decisive.

Option 3 is the one that makes §3's claim generalisable rather than SAC-specific.

---

## 5. A confound to state explicitly in any small-budget comparison

PPO is on-policy: at 100,000 steps with `n_steps=2048` and `N_ENVS=4` it
performs roughly **12 policy updates**, against ~187,000 gradient steps for the
off-policy algorithms at replay ratio 1.87.

That is not a defect in the benchmark — it *is* the sample-efficiency
difference, and it is the honest consequence of equal-step-budget comparison.
But it must be reported as "PPO is less sample-efficient at this budget", never
as "PPO is a worse algorithm". A step-matched comparison and a
wall-clock-matched comparison would rank them differently, and saying which
budget was equalised is part of reporting the result.
