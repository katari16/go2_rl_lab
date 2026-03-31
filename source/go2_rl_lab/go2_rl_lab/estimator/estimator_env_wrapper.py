"""Environment wrapper that augments policy observations with the estimator latent.

The wrapper intercepts get_observations() and step() calls to:
1. Maintain a rolling obs-history buffer.
2. Run the VelocityEstimator on the history to produce latent l_t.
3. Augment the "policy" obs group with l_t:
       augmented_policy_obs = concat(raw_policy_obs, l_t)
4. Leave the "critic" obs group untouched.

This means the standard OnPolicyRunner sees a (num_obs + latent_dim)-dimensional
"policy" observation and builds the actor network with the correct input size
automatically — no changes to rsl_rl internals required.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .obs_history_buffer import ObsHistoryBuffer
from .velocity_estimator import VelocityEstimator


class EstimatorEnvWrapper:
    """Wraps an rsl_rl VecEnv to augment policy observations with estimator latent.

    Optionally applies SAC-Loco-style velocity command modulation:
        v*_x = v'_x + k * F̂_x,  v*_y = v'_y + k * F̂_y
    where F̂ is an EMA-filtered force estimate from the estimator.

    Args:
        env:                Underlying rsl_rl VecEnv (RslRlVecEnvWrapper).
        estimator:          VelocityEstimator instance.
        history_buffer:     Shared ObsHistoryBuffer (same object used by runner
                            for estimator training data collection).
        device:             Torch device string.
    """

    def __init__(
        self,
        env,
        estimator: VelocityEstimator,
        history_buffer: ObsHistoryBuffer,
        device: str | torch.device,
    ) -> None:
        self._env = env
        self.estimator = estimator
        self.history_buffer = history_buffer
        self.device = device

        # Cache last raw policy obs so the runner can retrieve it for
        # the reconstruction target (next_obs in estimator.update()).
        self._last_raw_policy_obs: torch.Tensor | None = None

        # ── Compliance modulation (SAC-Loco) ─────────────────────────────
        # Set compliance_k > 0 to enable velocity command modulation.
        self.compliance_k: float = 0.0
        self.ema_alpha: float = 0.1
        # Velocity command XY indices in the raw observation
        self._vel_cmd_start: int = 6
        self._vel_cmd_end: int = 8
        # EMA-filtered force estimate (lazily initialized)
        self._force_filtered: torch.Tensor | None = None

        # ── Tracking reward correction ────────────────────────────────────
        # When enabled AND compliance_k > 0, corrects the velocity tracking
        # reward so it uses modulated command c* = c + k*f̂ instead of c.
        # This ensures the reward is consistent with the modulated observation.
        self.tracking_correction_enabled: bool = False
        # Must match env cfg RewardsCfg weights and std
        self._track_lin_weight: float = 1.5
        self._track_ang_weight: float = 0.75
        self._track_std_sq: float = 0.25  # std=sqrt(0.25), std²=0.25

    # ── Observation augmentation ──────────────────────────────────────────

    def _augment_obs(self, raw_obs: TensorDict) -> TensorDict:
        """Augment the 'policy' key with estimator latent.

        When compliance_k > 0, also modulates the velocity command with
        EMA-filtered force estimate: v*_xy = v'_xy + k * F̂_filtered_xy.
        """
        raw_policy = raw_obs["policy"].to(self.device)
        # Insert current obs into history, then compute latent
        self.history_buffer.insert(raw_policy)
        force_hat, latent = self.estimator.get_latent(self.history_buffer.get_flattened())

        # ── Compliance modulation ─────────────────────────────────────
        if self.compliance_k > 0.0:
            # Lazy-init EMA state
            if self._force_filtered is None:
                self._force_filtered = torch.zeros(
                    raw_policy.shape[0], force_hat.shape[-1], device=self.device
                )
            # EMA update
            self._force_filtered = (
                self.ema_alpha * force_hat + (1.0 - self.ema_alpha) * self._force_filtered
            )
            # Modulate velocity command XY in the raw obs
            raw_policy = raw_policy.clone()
            s, e = self._vel_cmd_start, self._vel_cmd_end
            raw_policy[:, s:e] = raw_policy[:, s:e] + self.compliance_k * self._force_filtered[:, :2]

        augmented_policy = torch.cat([raw_policy, latent], dim=-1)

        # Build a new TensorDict with the augmented policy obs
        augmented = {k: v for k, v in raw_obs.items()}
        augmented["policy"] = augmented_policy
        return TensorDict(augmented, batch_size=raw_obs.batch_size, device=self.device)

    # ── VecEnv interface ──────────────────────────────────────────────────

    def get_observations(self) -> TensorDict:
        raw_obs = self._env.get_observations()
        return self._augment_obs(raw_obs)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        raw_obs, rewards, dones, extras = self._env.step(actions)

        # Reset history for environments that just terminated
        done_ids = dones.nonzero(as_tuple=False).flatten()
        self.history_buffer.reset(done_ids)

        # Reset EMA-filtered force for terminated episodes
        if self._force_filtered is not None and len(done_ids) > 0:
            self._force_filtered[done_ids] = 0.0

        # Store raw policy obs before augmentation (used as next_obs target)
        self._last_raw_policy_obs = raw_obs["policy"].to(self.device).clone()

        augmented_obs = self._augment_obs(raw_obs)

        # ── Tracking reward correction ────────────────────────────────
        # When compliance is active, the observation's velocity command is
        # modulated to c* = c + k*f̂, but the env computed tracking reward
        # against the original c. We add a correction so the effective
        # tracking reward uses c*.
        if (
            self.tracking_correction_enabled
            and self.compliance_k > 0.0
            and self._force_filtered is not None
        ):
            correction = self._compute_tracking_correction()
            rewards = rewards.to(self.device) + correction

        return augmented_obs, rewards, dones, extras

    def _compute_tracking_correction(self) -> torch.Tensor:
        """Compute tracking reward correction: r_track(c*) - r_track(c).

        The env computed tracking reward using original command c.
        We need tracking reward using modulated command c* = c + k*f̂.
        The correction is the difference.
        """
        underlying = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        robot = underlying.scene["robot"]
        v_xy = robot.data.root_lin_vel_b[:, :2].to(self.device)

        # Original velocity command from raw obs (before modulation)
        c_xy = self._last_raw_policy_obs[:, self._vel_cmd_start:self._vel_cmd_end]

        # Compliance modulation delta
        delta_xy = self.compliance_k * self._force_filtered[:, :2]

        # Original tracking (what env computed)
        r_lin_orig = torch.exp(
            -((c_xy - v_xy) ** 2).sum(dim=-1) / self._track_std_sq
        )
        # Modified tracking (using c* = c + k*f̂)
        r_lin_mod = torch.exp(
            -(((c_xy + delta_xy) - v_xy) ** 2).sum(dim=-1) / self._track_std_sq
        )

        # Only correct linear XY tracking (force is XY only, yaw unchanged)
        return self._track_lin_weight * (r_lin_mod - r_lin_orig)

    def get_last_raw_policy_obs(self) -> torch.Tensor | None:
        """Return the raw (un-augmented) policy obs from the last step.

        Used by the runner as the reconstruction target next_obs for the
        estimator's decoder loss.
        """
        return self._last_raw_policy_obs

    # ── Delegate everything else to the underlying env ────────────────────

    def __getattr__(self, name: str):
        return getattr(self._env, name)
