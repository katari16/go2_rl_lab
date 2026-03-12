"""Environment wrapper for HAC-LOCO stage 2 compliance training.

Wraps the underlying IsaacLab env with a frozen low-level policy and
a TRAINABLE force estimator, exposing a new VecEnv interface for the
high-level compliance policy.

The high-level policy sees:
    obs:     o'_t = [s_t, c_t, l_{t-1}, a'_{t-1}, a_{t-1}]
    action:  a' = [Dvx, Dvy, Dwz]  (3 dims)

On step(a'):
    1. Clip a' to [-0.5, 0.5]
    2. Get raw env obs, save original velocity cmd c
    3. Modify velocity command: c* = c + a'
    4. Update history buffer, run estimator -> latent (TRAINABLE)
    5. Build augmented low-level obs: [raw_obs(57), force_est(3)] = 60
    6. Run frozen low-level policy -> 12 joint actions
    7. Step underlying env with joint actions
    8. Correct velocity tracking reward (HAC-LOCO eqs. 5-6)
    9. Compute compliance reward (HAC-LOCO eq. 7) from GT force
   10. total_reward = base + tracking_correction + w_comp * r_comp
   11. Return high-level obs, total reward, dones, extras

Obs layout — no foot contacts, XYZ force:
  Raw policy obs (60 dims): ang_vel(3), gravity(3), vel_cmd(3), jpos(12),
      jvel(12), last_action(12), torque*0.1(12), force_est(3)

High-level actor obs (124 dims):
    [0:3]    base_ang_vel
    [3:6]    projected_gravity
    [6:18]   joint_pos_rel
    [18:30]  joint_vel_rel
    [30:42]  applied_torque (scale=0.1)
    --- s_t: 42 dims ---
    [42:45]  velocity_commands (original c, NOT c*)
    --- c_t: 3 dims ---
    [45:112] previous estimator latent l_{t-1} (67 = 3 force + 64 enc)
    --- l_{t-1}: 67 dims ---
    [112:115] previous high-level action a'_{t-1}
    --- a'_{t-1}: 3 dims ---
    [115:127] previous low-level action a_{t-1}
    --- a_{t-1}: 12 dims ---

High-level critic obs (130 dims) = actor obs (127) + GT force XYZ (3)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


class ComplianceEnvWrapper:
    """Wraps env with frozen low-level policy + trainable estimator for stage 2.

    The high-level compliance policy outputs residual velocity commands
    a' = [Dvx, Dvy, Dwz] that modify the velocity command tracked by the
    frozen low-level policy: c* = c + a'.

    IMPORTANT: The force estimator is TRAINABLE (not frozen). The runner
    handles its gradient updates separately.
    """

    # Dims for this env (no foot contacts, XYZ force, enc_latent=64)
    S_DIM = 42           # ang_vel(3) + gravity(3) + jpos(12) + jvel(12) + torque(12)
    C_DIM = 3            # velocity commands
    LATENT_DIM = 67      # force_dim(3) + enc_latent(64)
    HIGH_ACTION_DIM = 3  # [Dvx, Dvy, Dwz]
    LOW_ACTION_DIM = 12  # joint positions

    HIGH_LEVEL_OBS_DIM = S_DIM + C_DIM + LATENT_DIM + HIGH_ACTION_DIM + LOW_ACTION_DIM  # 127
    HIGH_LEVEL_CRITIC_OBS_DIM = HIGH_LEVEL_OBS_DIM + 3  # + GT force XYZ = 130
    HIGH_LEVEL_ACTION_DIM = HIGH_ACTION_DIM  # 3

    # Raw env obs layout
    RAW_OBS_DIM = 57     # without force estimate
    FORCE_DIM = 3        # XYZ

    def __init__(
        self,
        env,
        frozen_policy,
        estimator,  # TRAINABLE, not frozen
        history_buffer,
        alpha: float = 10.0,
        beta: float = 10.0,
        compliance_reward_weight: float = 0.5,
        gt_force_start: int = 64,
        gt_force_dim: int = 3,
        device: str | torch.device = "cpu",
        reward_type: str = "penalty",
        gravity_correction: bool = False,
        use_tracking_reward: bool = True,
        robot_mass: float = 15.0,
    ) -> None:
        self._env = env
        self.frozen_policy = frozen_policy
        self.estimator = estimator  # TRAINABLE
        self.history_buffer = history_buffer
        self.alpha = alpha
        self.beta = beta
        self.compliance_reward_weight = compliance_reward_weight
        self.device = device
        self._reward_type = reward_type
        self._gravity_correction = gravity_correction
        self._use_tracking_reward = use_tracking_reward
        self._robot_mass = robot_mass

        num_envs = env.num_envs

        self._underlying = env.unwrapped if hasattr(env, "unwrapped") else env

        self._action_clip = 3.0

        # Tracking reward weights (must match env cfg)
        self._track_lin_weight = 1.5
        self._track_ang_weight = 0.75
        self._track_std_sq = 0.25

        # GT force location in critic obs
        self._gt_force_start = gt_force_start
        self._gt_force_dim = gt_force_dim

        # State buffers
        self._last_latent = torch.zeros(num_envs, self.LATENT_DIM, device=device)
        self._last_high_action = torch.zeros(num_envs, 3, device=device)
        self._last_low_action = torch.zeros(num_envs, 12, device=device)

        # VecEnv interface
        self.num_actions = self.HIGH_LEVEL_ACTION_DIM

        # Episode stats
        self._ep_compliance_rew_sum = torch.zeros(num_envs, device=device)
        self._ep_base_rew_sum = torch.zeros(num_envs, device=device)
        self._ep_tracking_corr_sum = torch.zeros(num_envs, device=device)
        self._ep_steps = torch.zeros(num_envs, device=device)

        # Per-step stats for runner logging
        self.last_compliance_reward = torch.zeros(num_envs, device=device)
        self.last_gt_force_mag = torch.zeros(num_envs, device=device)
        self.last_action_norm = torch.zeros(num_envs, device=device)
        self.last_tracking_correction = torch.zeros(num_envs, device=device)

    def _extract_s_t(self, raw_policy: torch.Tensor) -> torch.Tensor:
        """Extract proprioceptive state s_t (42 dims) from raw 60-dim obs.

        Excludes: vel_cmd [6:9], last_action [33:45], force_estimate [57:60]
        Includes: ang_vel(3) + gravity(3) + jpos(12) + jvel(12) + torque(12) = 42
        """
        return torch.cat([
            raw_policy[:, 0:6],    # ang_vel + gravity
            raw_policy[:, 9:33],   # joint_pos + joint_vel
            raw_policy[:, 45:57],  # applied_torque
        ], dim=-1)

    def _build_high_level_obs(
        self,
        raw_policy: torch.Tensor,
        raw_critic: torch.Tensor | None = None,
    ) -> TensorDict:
        """Build high-level obs: o'_t = [s_t, c_t, l_{t-1}, a'_{t-1}, a_{t-1}]."""
        s_t = self._extract_s_t(raw_policy)  # 42
        c_t = raw_policy[:, 6:9]             # 3

        high_obs = torch.cat([
            s_t,                        # 42
            c_t,                        # 3
            self._last_latent,          # 67
            self._last_high_action,     # 3
            self._last_low_action,      # 12
        ], dim=-1)  # 127

        # Critic: append GT force XYZ (privileged)
        if raw_critic is not None:
            gt_force = raw_critic[:, self._gt_force_start:self._gt_force_start + self._gt_force_dim]
            critic_obs = torch.cat([high_obs, gt_force], dim=-1)  # 130
        else:
            critic_obs = torch.cat([
                high_obs,
                torch.zeros(high_obs.shape[0], self._gt_force_dim, device=self.device),
            ], dim=-1)

        return TensorDict(
            {"policy": high_obs, "critic": critic_obs},
            batch_size=[raw_policy.shape[0]],
            device=self.device,
        )

    def _compute_compliance_reward(
        self,
        a_prime: torch.Tensor,
        gt_force: torch.Tensor,
    ) -> torch.Tensor:
        """HAC-LOCO compliance reward (eq. 7).

        Penalty mode:
            When |f| <= alpha: r = -||a'||^2 (minimize adjustments)
            When |f| > alpha:  r = -||a'_xy - target||^2 - ||a'_z||^2

        Positive mode:
            When |f| <= alpha: r = exp(-||a'||^2)
            When |f| > alpha:  r = exp(-||a'_xy - target||^2 - ||a'_z||^2)

        Args:
            a_prime:  [num_envs, 3] -- high-level action [Dvx, Dvy, Dwz]
            gt_force: [num_envs, 3] -- GT XYZ force (Newtons)
        """
        f_norm = gt_force.norm(dim=-1)  # [num_envs]

        # Large force: comply in XY, minimize yaw adjustment
        f_xy = gt_force[:, :2]
        f_xy_norm = f_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        target_a_xy = (f_xy / self.beta) * (1.0 + self.alpha / f_xy_norm)
        target_a_xy = target_a_xy.clamp(-self._action_clip, self._action_clip)

        # Squared errors
        err_small = (a_prime ** 2).sum(dim=-1)
        err_large = ((a_prime[:, :2] - target_a_xy) ** 2).sum(dim=-1) + a_prime[:, 2] ** 2

        err = torch.where(f_norm <= self.alpha, err_small, err_large)

        if self._reward_type == "positive":
            return torch.exp(-err)
        else:
            return -err

    # ── VecEnv interface ──────────────────────────────────────────────────

    def get_observations(self) -> dict:
        raw_obs = self._env.get_observations()
        raw_policy = raw_obs["policy"].to(self.device)
        raw_critic = raw_obs.get("critic", None)
        if raw_critic is not None:
            raw_critic = raw_critic.to(self.device)
        return self._build_high_level_obs(raw_policy, raw_critic)

    def step(self, high_level_action: torch.Tensor):
        """Execute one high-level step.

        Returns: (obs_dict, rewards, dones, extras)
        """
        high_level_action = high_level_action.to(self.device)
        high_level_action = torch.clamp(high_level_action, -self._action_clip, self._action_clip)

        # 1. Get current raw obs
        raw_obs = self._env.get_observations()
        raw_policy = raw_obs["policy"].to(self.device)  # [num_envs, 60]

        # 2. Extract raw obs without force estimate (first 57 dims)
        raw_57 = raw_policy[:, :self.RAW_OBS_DIM]

        # 3. Modify velocity commands: c* = c + a'
        modified_57 = raw_57.clone()
        modified_57[:, 6] += high_level_action[:, 0]  # Dvx
        modified_57[:, 7] += high_level_action[:, 1]  # Dvy
        modified_57[:, 8] += high_level_action[:, 2]  # Dwz

        # 4. Update history buffer and run estimator (TRAINABLE — no inference_mode)
        self.history_buffer.insert(modified_57)
        force_hat, latent = self.estimator.get_latent(
            self.history_buffer.get_flattened()
        )

        # 5. Set force estimate on env (for ForceEstimateObsTerm passthrough)
        self._underlying._force_estimate_xy = force_hat

        # 6. Build low-level obs: [raw_57_modified(57), force_hat(3)] = 60
        low_obs_tensor = torch.cat([modified_57, force_hat.detach()], dim=-1)
        low_level_obs = TensorDict(
            {"policy": low_obs_tensor},
            batch_size=[modified_57.shape[0]],
            device=self.device,
        )

        # 7. Run frozen low-level policy
        with torch.inference_mode():
            low_level_action = self.frozen_policy.act_inference(low_level_obs)

        # 8. Save original velocity command for tracking correction
        vel_cmd = raw_57[:, 6:9]  # original c

        # 9. Step underlying env with low-level actions
        obs_next, rewards, dones, extras = self._env.step(low_level_action)

        # 10. Correct velocity tracking reward (HAC-LOCO eqs. 5-6)
        robot = self._underlying.scene["robot"]
        v_xy = robot.data.root_lin_vel_b[:, :2].to(self.device)
        w_z = robot.data.root_ang_vel_b[:, 2:3].to(self.device)

        c_xy = vel_cmd[:, :2]
        c_z = vel_cmd[:, 2:3]
        a_xy = high_level_action[:, :2]
        a_z = high_level_action[:, 2:3]

        r_lin_orig = torch.exp(-((c_xy - v_xy) ** 2).sum(dim=-1) / self._track_std_sq)
        r_ang_orig = torch.exp(-((c_z - w_z) ** 2).sum(dim=-1) / self._track_std_sq)
        r_lin_mod = torch.exp(-(((c_xy + a_xy) - v_xy) ** 2).sum(dim=-1) / self._track_std_sq)
        r_ang_mod = torch.exp(-(((c_z + a_z) - w_z) ** 2).sum(dim=-1) / self._track_std_sq)

        tracking_correction = (
            self._track_lin_weight * (r_lin_mod - r_lin_orig)
            + self._track_ang_weight * (r_ang_mod - r_ang_orig)
        )

        # 11. GT force from critic obs for compliance reward
        raw_obs_next = obs_next
        if "critic" in raw_obs_next.keys():
            critic_obs = raw_obs_next["critic"].to(self.device)
            gt_force = critic_obs[:, self._gt_force_start:self._gt_force_start + self._gt_force_dim]
        else:
            gt_force = torch.zeros(raw_policy.shape[0], self._gt_force_dim, device=self.device)

        # 12. Total reward: modified tracking (eqs. 5-6) + compliance (eq. 7)
        # NO base env reward — it's constant w.r.t. a' and drowns the signal
        compliance_rew = self._compute_compliance_reward(high_level_action, gt_force)

        # Modified tracking: how well the low-level tracks c* = c + a'
        r_tracking = (
            self._track_lin_weight * r_lin_mod
            + self._track_ang_weight * r_ang_mod
        )

        if self._use_tracking_reward:
            total_reward = r_tracking + self.compliance_reward_weight * compliance_rew
        else:
            total_reward = self.compliance_reward_weight * compliance_rew

        # 13. Stats
        self.last_compliance_reward = compliance_rew.detach()
        self.last_gt_force_mag = gt_force.norm(dim=-1).detach()
        self.last_action_norm = high_level_action.norm(dim=-1).detach()
        self.last_tracking_correction = tracking_correction.detach()

        self._ep_compliance_rew_sum += compliance_rew.detach()
        self._ep_base_rew_sum += r_tracking.detach()
        self._ep_tracking_corr_sum += tracking_correction.detach()
        self._ep_steps += 1

        # 14. Update state buffers
        latent_for_obs = latent.detach()
        if self._gravity_correction:
            # Subtract gravity component from force estimate (first 3 dims of latent)
            # Projected gravity in body frame is raw_obs[3:6] = [g_Bx, g_By, g_Bz]
            # F_gravity_B = m * [g_Bx, g_By, 0]
            gravity_body = raw_57[:, 3:6].detach()
            f_gravity = self._robot_mass * gravity_body
            f_gravity[:, 2] = 0.0  # only correct XY
            latent_for_obs = latent_for_obs.clone()
            latent_for_obs[:, :self.FORCE_DIM] -= f_gravity
        self._last_latent = latent_for_obs
        self._last_high_action = high_level_action.detach()
        self._last_low_action = low_level_action.detach()

        # 15. Reset for terminated envs
        dones_device = dones.to(self.device)
        done_ids = dones_device.nonzero(as_tuple=False).flatten()
        self.history_buffer.reset(done_ids)
        if len(done_ids) > 0:
            self._last_latent[done_ids] = 0.0
            self._last_high_action[done_ids] = 0.0
            self._last_low_action[done_ids] = 0.0

            # Episode stats for logging
            ep_len = self._ep_steps[done_ids].clamp(min=1)
            if "episode" in extras:
                log_dict = extras["episode"]
            elif "log" in extras:
                log_dict = extras["log"]
            else:
                log_dict = {}
                extras["log"] = log_dict

            log_dict["Compliance/r_comp_mean"] = (
                self._ep_compliance_rew_sum[done_ids] / ep_len
            ).mean().item()
            log_dict["Compliance/base_reward_mean"] = (
                self._ep_base_rew_sum[done_ids] / ep_len
            ).mean().item()
            log_dict["Compliance/tracking_correction_mean"] = (
                self._ep_tracking_corr_sum[done_ids] / ep_len
            ).mean().item()

            self._ep_compliance_rew_sum[done_ids] = 0
            self._ep_base_rew_sum[done_ids] = 0
            self._ep_tracking_corr_sum[done_ids] = 0
            self._ep_steps[done_ids] = 0

        # 16. Build next high-level obs
        raw_policy_next = obs_next["policy"].to(self.device)
        raw_critic_next = obs_next.get("critic", None)
        if raw_critic_next is not None:
            raw_critic_next = raw_critic_next.to(self.device)
        high_obs = self._build_high_level_obs(raw_policy_next, raw_critic_next)

        return high_obs, total_reward, dones, extras

    def __getattr__(self, name: str):
        return getattr(self._env, name)
