"""
Gymnasium environment for the 3-D arm reaching task.

The agent controls 9 muscle activations in [0, 1].
At each timestep MuJoCo advances the physics, and the Hill model embedded
in the actuator XML implicitly solves the muscle dynamics — the RL agent
must learn a policy that produces the correct force pattern without ever
explicitly running the Newton-Raphson solver.

Observation (38-dim):
    [0:4]   joint angles q (rad)
    [4:8]   joint velocities qdot (rad/s)
    [8:17]  muscle lengths L_M (m)
    [17:26] muscle forces F (N)
    [26:35] muscle activations a (dimensionless)
    [35:38] reaching error vector  target - end_effector (m)

Targets are sampled by forward kinematics from random joint configurations
inside the joint limits, so every target is reachable by construction (the
old fixed sampling box lay largely outside the arm's workspace).

Reward (v3, 2026-08-14 — restructured after the termination-exploit diagnosis):
    r = w4 - (w1 * (||err||^2 + 0.5*||err||) + w2 * e_eff + w3 * ||da||^2)
    r += w5   for every step the hand is at the target and settled
    r -= w6   once, if the episode ends in a blow-up
  where e_eff = mean_i (F_i / F_max,i)^2 is the effort normalised per muscle
  by its peak isometric force, so the term is O(1). v1 used w2 * sum(F^2),
  which reached O(10^4) per step and drowned the reaching term by 4-5 orders
  of magnitude — the energy-optimal policy was to do nothing.

  Why v2 had to change. v2 set w4=0, making every per-step reward <= 0, and
  terminated with no penalty on a numerical blow-up. A terminated episode
  bootstraps nothing, so its return is exactly 0, while surviving 100 steps
  cost about -20. Destroying the arm was therefore worth roughly twice the
  w5=10 success bonus — and measurably better than reaching, which even a
  privileged oracle achieves only ~7% of the time. The policy learned this
  precisely: diagnose_termination.py measured 100% of evaluation episodes
  ending in a deliberate blow-up at step ~7 of 100, at ~40% mean activation.
  Every behavioural number produced before 2026-08-14 is a measurement of that
  behaviour, not of reaching.

  v3 makes w4 a survival offset big enough that r > 0 at any reachable error,
  so ending an episode early forfeits value instead of banking it. Because a
  constant shift cannot reorder trajectories of equal length, the v2 distance
  shaping is preserved exactly. Success no longer terminates: w5 is paid per
  step while on target, which turns the task into reach-and-hold and removes
  the v1 conflict where an alive bonus made hovering beat succeeding.

  The raw metabolic proxy sum(F^2) is still reported in info["energy"] as an
  evaluation metric — only the reward uses the normalised term.

Control rate:
    The policy acts every config.FRAME_SKIP physics steps (action repeat), so
    commands are issued at 50 Hz while physics integrates at 500 Hz. See
    config.FRAME_SKIP for why (episode count, and it is nearly free).

Termination:
    failure  : any |joint velocity| > 50 rad/s, or a non-finite state
               (numerical blow-up guard) — now costs w6 and forfeits the rest
               of the episode's survival reward
    truncation: step count >= max_steps (100 agent steps = 2 s)
  Success does NOT terminate (v3); reaching the target early simply means the
  agent collects w5 for the remaining steps, provided it can hold position.
  info["success"] at the final step therefore means "at target AND settled at
  t = 2 s", a stricter and more useful criterion than v2's "touched it once".
  (v1 also failed on err > 0.60 m; with the hanging start pose ~0.87 m below
  the shoulder that killed most episodes on step 1 — removed.)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import config


class ArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, domain_rand: bool = True, max_steps: int = None,
                 render_mode=None, seed=None, reward_weights: dict = None,
                 obs_noise_sigma: float = 0.0, ext_force_N: float = 0.0,
                 workspace_scale: float = 1.0, frame_skip: int = None):
        super().__init__()
        self.model       = mujoco.MjModel.from_xml_path(config.ARM_XML)
        self.data        = mujoco.MjData(self.model)
        self.domain_rand = domain_rand
        self.frame_skip  = config.FRAME_SKIP if frame_skip is None else frame_skip
        self.max_steps   = config.MAX_EPISODE_STEPS if max_steps is None else max_steps
        self.render_mode = render_mode
        self.rng         = np.random.default_rng(seed)

        # Nominal model parameters. Domain randomisation must scale FROM these
        # every episode; v1 scaled the live model arrays in place each reset,
        # a multiplicative random walk that drove body mass to ~1% of nominal
        # within a thousand episodes (measured in diagnose_env.py).
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_gainprm   = self.model.actuator_gainprm.copy()
        self._nominal_damping   = self.model.dof_damping.copy()
        # Per-muscle peak isometric force (gainprm[2] for muscle actuators),
        # used to normalise the reward's effort term.
        self._fmax = self._nominal_gainprm[:, 2].copy()

        # FK scratch buffer + cached ids for reachable-target sampling.
        self._scratch  = mujoco.MjData(self.model)
        self._site_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                           "end_effector")
        self._shoulder = self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "humerus")].copy()

        self.action_space      = spaces.Box(0.0, 1.0, shape=(9,),  dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(38,), dtype=np.float32)

        # Reward weights. Default to the single source of truth in config; an
        # override dict (used by the reward-ablation study) may zero any term.
        w = dict(config.REWARD_WEIGHTS)
        if reward_weights:
            w.update(reward_weights)
        self.w1, self.w2, self.w3 = w["w1"], w["w2"], w["w3"]
        self.w4, self.w5          = w["w4"], w["w5"]
        # w6 defaults to 0 so a reward-ablation override dict that omits it
        # (they are written against the v2 key set) still constructs.
        self.w6                   = w.get("w6", 0.0)
        self.w7                   = w.get("w7", 0.0)

        # Evaluation-time perturbation hooks (0 = disabled). Used by the
        # robustness/generalization analysis, NOT during training.
        self.obs_noise_sigma = obs_noise_sigma
        self.ext_force_N     = ext_force_N
        self.workspace_scale = workspace_scale

    # Warning classes that mean the integrator lost the state. MuJoCo recovers
    # from these by resetting internally, so they must be counted, not inferred
    # from the post-step state (see step()).
    _DIVERGENCE_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQACC,
                            mujoco.mjtWarning.mjWARN_BADQPOS,
                            mujoco.mjtWarning.mjWARN_BADQVEL)

    def _warn_count(self) -> int:
        return int(sum(self.data.warning[w].number
                       for w in self._DIVERGENCE_WARNINGS))

    def _sample_target(self, ee_start: np.ndarray) -> np.ndarray:
        """FK from a random joint configuration -> guaranteed-reachable target.

        workspace_scale != 1 (robustness study) scales the target's offset from
        the shoulder, deliberately pushing outside the trained workspace.
        Resample if the target is trivially close to the start pose.
        """
        lo, hi = self.model.jnt_range[:, 0].copy(), self.model.jnt_range[:, 1].copy()
        # shoulder_rot (index 2) is spanned by no modeled muscle and is
        # constrained near neutral in arm.xml; sample targets at that neutral
        # rotation so every target is reachable by actuation, not merely by
        # kinematics.
        lo[2], hi[2] = -0.02, 0.02
        target = None
        for _ in range(20):
            q = lo + self.rng.uniform(0.05, 0.95, self.model.nq) * (hi - lo)
            self._scratch.qpos[:] = q
            mujoco.mj_forward(self.model, self._scratch)
            p = self._scratch.site_xpos[self._site_id]
            target = self._shoulder + self.workspace_scale * (p - self._shoulder)
            if np.linalg.norm(target - ee_start) >= 0.10:
                break
        return np.asarray(target, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Restore nominal parameters, then apply this episode's randomisation.
        self.model.body_mass[:]        = self._nominal_body_mass
        self.model.actuator_gainprm[:] = self._nominal_gainprm
        self.model.dof_damping[:]      = self._nominal_damping
        if self.domain_rand:
            # size=... is load-bearing. Without it rng.uniform returns a SCALAR,
            # so every body got the identical mass multiplier, every muscle the
            # identical strength multiplier — a single global scale factor
            # rather than the independent per-element variation the robustness
            # claim rests on. A policy can trivially infer and cancel one global
            # scale; surviving independent per-muscle strength variation is the
            # property actually worth reporting.
            self.model.body_mass[:]           *= self.rng.uniform(
                0.85, 1.15, size=self.model.body_mass.shape)
            self.model.actuator_gainprm[:, 2] *= self.rng.uniform(
                0.80, 1.20, size=self.model.actuator_gainprm.shape[0])
            self.model.dof_damping[:]         *= self.rng.uniform(
                0.70, 1.30, size=self.model.dof_damping.shape)

        # Compute kinematics BEFORE the first observation; v1 read site_xpos
        # straight after mj_resetData, when it is stale/zero, so the first
        # observation reported the end effector at the world origin.
        mujoco.mj_forward(self.model, self.data)
        ee_start    = self.data.site_xpos[self._site_id].copy()
        self.target = self._sample_target(ee_start)

        self.steps       = 0
        self.prev_action = np.zeros(9, dtype=np.float32)
        # Last finite observation/error. If the state goes non-finite mid-step
        # we must report these rather than the post-reset home pose (see step).
        obs = self._obs()
        self._last_obs = obs.copy()
        self._last_err = float(np.linalg.norm(
            self.target - self.data.site_xpos[self._site_id]))
        return obs, {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        self.data.ctrl[:] = action
        if self.domain_rand:
            self.data.ctrl[:] = np.clip(
                self.data.ctrl + self.rng.normal(0, 0.02, 9), 0.0, 1.0)

        # Evaluation-time external force perturbation (robustness study only).
        if self.ext_force_N > 0.0:
            body_id = self.model.nbody - 1   # distal (hand) body
            f = self.rng.normal(0, 1, 3)
            f = self.ext_force_N * f / (np.linalg.norm(f) + 1e-9)
            self.data.xfrc_applied[body_id, :3] = f

        # Action repeat: the command is held for frame_skip physics steps, so
        # the policy acts at 50 Hz while physics integrates at 500 Hz. Effort
        # is averaged over the held interval, not sampled at its end.
        e_met_acc, e_eff_acc = 0.0, 0.0
        blew_up, nonfinite, n_sub = False, False, 0
        warn_before = self._warn_count()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            n_sub += 1
            # Divergence detection. Checking isfinite() alone is NOT sufficient:
            # when MuJoCo detects a bad QACC/QPOS/QVEL it emits a warning and
            # RESETS the simulation internally, so by the time control returns
            # the state is finite again — sitting at the home pose. v2 therefore
            # saw no blow-up, silently teleported the arm mid-episode, and went
            # on measuring the error of the home pose against the old target.
            # The warning counters are the only reliable signal that this
            # happened; isfinite() is kept as a backstop.
            if (self._warn_count() > warn_before
                    or not (np.all(np.isfinite(self.data.qpos))
                            and np.all(np.isfinite(self.data.qvel)))):
                blew_up, nonfinite = True, True
                break
            e_met_acc += float(np.sum(self.data.actuator_force ** 2))
            e_eff_acc += float(np.mean((self.data.actuator_force / self._fmax) ** 2))
            if np.any(np.abs(self.data.qvel[:4]) > 50):
                # ~8 revolutions/second: not a reachable human arm state, so
                # treat it as a failed episode rather than continue integrating.
                blew_up = True
                break

        if nonfinite:
            # Report the last finite state. v2 called mj_resetData here and THEN
            # read site_xpos, so every blown-up episode reported the error of
            # the home pose instead of the state it actually failed in — which
            # silently contaminated mean_final_error for every such episode.
            obs, err = self._last_obs.copy(), self._last_err
            v = 0.0
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        else:
            obs = self._obs()
            err = float(np.linalg.norm(
                self.target - self.data.site_xpos[self._site_id]))
            v   = float(np.linalg.norm(self.data.qvel[:4]))
            self._last_obs, self._last_err = obs.copy(), err

        # Effort is averaged over the sub-steps that actually ran.
        denom  = max(n_sub - (1 if nonfinite else 0), 1)
        e_met  = e_met_acc / denom                            # reported metric
        e_eff  = e_eff_acc / denom
        # Mean, not sum: bounds the term by 1 like the effort term, so the two
        # penalties are on the same scale and w3 is interpretable against w2.
        d_act  = float(np.mean((action - self.prev_action) ** 2))
        self.last_energy_penalty = self.w2 * e_eff

        # Error cost: quadratic + linear. Quadratic alone has no gradient
        # teeth near the target (err^2 = 0.01 at 10 cm, below the effort cost
        # of pressing home against gravity), so pilot policies hovered at
        # 10-15 cm; the linear term keeps the approach gradient alive at all
        # ranges. Both share w1 so the reward ablation removes them together.
        #
        # w4 is the survival offset. The cost terms are bounded by
        # w1*(err^2+0.5*err) <= 2.7 at the workspace-diameter error of ~1.4 m,
        # w2*e_eff <= 1.0 and w3*mean(da^2) <= 0.5, so w4=4.5 keeps r > 0
        # everywhere and ending an episode early is never a gain.
        #
        # w7 is the precision bonus. The quadratic+linear cost above has a
        # gradient that flattens on approach (0.96/m at 23 cm, 0.60/m at 5 cm),
        # so the policy stalls where improving stops paying. The exponential is
        # negligible far away and steep in the last 10-15 cm, which is exactly
        # the region the v3 policy could not refine in.
        reward  = self.w4 - (self.w1 * (err**2 + 0.5 * err)
                             + self.w2 * e_eff + self.w3 * d_act)
        reward += self.w7 * np.exp(-err / config.PRECISION_SCALE)
        # A diverged episode can never be a success. On divergence `v` is
        # forced to 0.0 (there is no meaningful velocity to report), which would
        # otherwise satisfy the "settled" half of the criterion automatically --
        # so an episode that blew up within 2 cm of the target would be scored a
        # success AND collect w5. Latent while blow_up_rate is 0, but it is a
        # reward exploit of exactly the kind documented in the module docstring.
        success = bool(err < 0.02 and v < 0.1 and not blew_up)
        if success:
            reward += self.w5          # per step held on target, not terminal
        if blew_up:
            reward -= self.w6

        self.steps      += 1
        # Success no longer terminates: the task is reach AND hold, so an agent
        # that arrives early should be paid for staying, not cut off. The only
        # terminal state is a failure.
        terminated       = bool(blew_up)
        truncated        = self.steps >= self.max_steps
        self.prev_action = action

        return obs, reward, terminated, truncated, {
            "err": err, "success": success, "energy": e_met, "effort": e_eff,
            "blew_up": blew_up, "steps": self.steps}

    def _obs(self) -> np.ndarray:
        ee  = self.data.site_xpos[self._site_id]
        # Slices, not list comprehensions. The three muscle blocks used to be
        # built as `[arr[i] for i in range(9)]`, which materialises 27 Python
        # float objects on every observation; MuJoCo already exposes these as
        # (9,) float64 arrays (nu = na = 9, verified against arm.xml), so the
        # slices are the same values with no per-element boxing.
        obs = np.concatenate([
            self.data.qpos[:4],
            self.data.qvel[:4],
            self.data.actuator_length[:9],
            self.data.actuator_force[:9],
            self.data.act[:9],
            self.target - ee,
        ]).astype(np.float32)
        # Evaluation-time observation noise (robustness study only).
        if self.obs_noise_sigma > 0.0:
            obs = obs + self.rng.normal(0, self.obs_noise_sigma, obs.shape).astype(np.float32)
        return obs
