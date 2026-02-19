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

    # ── Observation augmentation ──────────────────────────────────────────

    def _augment_obs(self, raw_obs: TensorDict) -> TensorDict:
        """Augment the 'policy' key with estimator latent."""
        raw_policy = raw_obs["policy"].to(self.device)
        # Insert current obs into history, then compute latent
        self.history_buffer.insert(raw_policy)
        _, latent = self.estimator.get_latent(self.history_buffer.get_flattened())
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

        # Store raw policy obs before augmentation (used as next_obs target)
        self._last_raw_policy_obs = raw_obs["policy"].to(self.device).clone()

        augmented_obs = self._augment_obs(raw_obs)
        return augmented_obs, rewards, dones, extras

    def get_last_raw_policy_obs(self) -> torch.Tensor | None:
        """Return the raw (un-augmented) policy obs from the last step.

        Used by the runner as the reconstruction target next_obs for the
        estimator's decoder loss.
        """
        return self._last_raw_policy_obs

    # ── Delegate everything else to the underlying env ────────────────────

    def __getattr__(self, name: str):
        return getattr(self._env, name)
