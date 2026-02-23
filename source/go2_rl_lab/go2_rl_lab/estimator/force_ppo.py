"""ForceEstimatorPPO — PPO with force estimator updated per mini-batch.

Follows the HAC-LOCO pattern (HIM-PPO): the estimator has its own optimizer
and is updated inside the PPO mini-batch loop *before* the PPO gradient step.

Training loop (per mini-batch):
    1. estimator.update(obs_history_batch, gt_force_batch, next_obs_batch, lr)
    2. Standard PPO: surrogate_loss + value_loss - entropy  → PPO optimizer
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO

from .force_estimator import ForceEstimator


class ForceEstimatorPPO(PPO):
    """PPO extended with per-mini-batch force estimator updates.

    The estimator is trained on data stored by the runner in external buffers
    (obs_history, next_raw_obs). These are passed via :meth:`set_estimator_data`
    before each :meth:`update` call.

    Args:
        force_estimator:  ForceEstimator instance (owns its own optimizer).
        gt_force_start:   Start index of GT force in critic obs.
        force_dim:        Dimension of the force vector (default: 2).
        **ppo_kwargs:     All standard PPO constructor arguments.
    """

    def __init__(
        self,
        force_estimator: ForceEstimator,
        gt_force_start: int,
        force_dim: int = 2,
        **ppo_kwargs,
    ) -> None:
        super().__init__(**ppo_kwargs)
        self.force_estimator = force_estimator
        self._gt_force_start = gt_force_start
        self._force_dim = force_dim

        # Estimator rollout data — set by the runner before update()
        self._est_obs_history: torch.Tensor | None = None
        self._est_next_raw_obs: torch.Tensor | None = None

    def set_estimator_data(
        self,
        obs_history: torch.Tensor,
        next_raw_obs: torch.Tensor,
    ) -> None:
        """Provide estimator training data collected during the rollout.

        Args:
            obs_history:  [num_steps, num_envs, temporal_steps * obs_dim]
            next_raw_obs: [num_steps, num_envs, obs_dim]
        """
        self._est_obs_history = obs_history
        self._est_next_raw_obs = next_raw_obs

    def update(self) -> dict[str, float]:
        """PPO update with estimator updated per mini-batch (HAC-LOCO pattern).

        Returns:
            Loss dict with standard PPO losses + force_loss, rec_loss.
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_force_loss = 0
        mean_rec_loss = 0

        # ── Flatten estimator data to match PPO's flattened batches ──────
        n = self.storage.num_transitions_per_env * self.storage.num_envs
        est_obs_h = self._est_obs_history.reshape(n, -1)
        est_next_o = self._est_next_raw_obs.reshape(n, -1)

        # Extract GT force from critic obs stored in PPO rollout storage
        critic_obs = self.storage.observations["critic"]  # [num_steps, num_envs, critic_dim]
        gt_force = critic_obs[:, :, self._gt_force_start : self._gt_force_start + self._force_dim]
        gt_force = gt_force.reshape(n, self._force_dim).detach()

        # ── Mini-batch generator ─────────────────────────────────────────
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # We need the same indices as the PPO generator to slice estimator data.
        # The standard generator uses sequential slicing of a randperm, so we
        # replicate that indexing scheme.
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(
            self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        mini_batch_counter = 0

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            # ── Estimator update FIRST (own optimizer) ───────────────────
            # Compute the batch indices for this mini-batch
            epoch_idx = mini_batch_counter // self.num_mini_batches
            mb_idx = mini_batch_counter % self.num_mini_batches
            start = mb_idx * mini_batch_size
            stop = (mb_idx + 1) * mini_batch_size
            batch_idx = indices[start:stop]

            self.force_estimator.train()
            est_losses = self.force_estimator.update(
                obs_history=est_obs_h[batch_idx],
                gt_force=gt_force[batch_idx],
                next_obs=est_next_o[batch_idx],
                lr=self.learning_rate,
            )
            mean_force_loss += est_losses["force_loss"]
            mean_rec_loss += est_losses["rec_loss"]

            mini_batch_counter += 1

            # ── Standard PPO update ──────────────────────────────────────
            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Recompute actions log prob and entropy
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL divergence and adaptive LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # PPO gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        # ── Average losses ───────────────────────────────────────────────
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_force_loss /= num_updates
        mean_rec_loss /= num_updates

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "force_loss": mean_force_loss,
            "rec_loss": mean_rec_loss,
        }
