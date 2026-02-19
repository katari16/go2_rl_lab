"""EstimatorOnPolicyRunner — PPO + concurrent velocity estimator training.

Extends rsl_rl's OnPolicyRunner with:
1. A VelocityEstimator trained concurrently via supervised learning.
2. An EstimatorEnvWrapper that augments policy observations with the latent.
3. Estimator-specific terminal logging alongside standard PPO stats.
4. Estimator weights saved/loaded with the policy checkpoint.

Training loop (per iteration):
    ┌── Rollout collection (num_steps_per_env steps) ──────────────────────┐
    │  for each step t:                                                      │
    │    1. augmented_obs = wrapper.get_observations()                       │
    │       = concat(raw_obs[t], latent[t])   (policy sees 45+67=112 dims)  │
    │    2. actions = alg.act(augmented_obs)                                 │
    │    3. raw_obs[t+1], ... = env.step(actions)                            │
    │    4. Store (obs_history_flat[t], next_raw_obs[t]) in est. buffers     │
    └───────────────────────────────────────────────────────────────────────┘
    ┌── PPO update (standard) ─────────────────────────────────────────────┐
    └───────────────────────────────────────────────────────────────────────┘
    ┌── Estimator update (supervised, num_estimator_mini_batches passes) ──┐
    │  gt_velocity extracted from critic obs stored in PPO rollout buffer   │
    │  Losses: L_vel = MSE(v̂, v_gt)  +  L_rec = MSE(ô_{t+1}, o_{t+1})    │
    └───────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from rsl_rl.runners import OnPolicyRunner

from .estimator_env_wrapper import EstimatorEnvWrapper
from .obs_history_buffer import ObsHistoryBuffer
from .velocity_estimator import VelocityEstimator


class EstimatorOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner extended with concurrent state-estimator training.

    Extra keys expected in train_cfg (under "estimator"):
        temporal_steps          int     History window length (default: 50)
        enc_hidden_dims         list    Encoder hidden dims (default: [256,128,64])
        v_head_dims             list    Velocity-head dims (default: [32,16])
        f_head_dims             list    Force-head dims (default: None — disabled)
        force_dim               int     Force output dim (default: 2 for XY)
        dec_hidden_dims         list    Decoder dims (default: [512,256,128])
        activation              str     Activation (default: "elu")
        learning_rate           float   Estimator LR (default: 1e-3)
        vel_loss_weight         float   Weight for velocity loss (default: 1.0)
        force_loss_weight       float   Weight for force loss (default: 1.0)
        rec_loss_weight         float   Weight for reconstruction loss (default: 1.0)
        max_grad_norm           float   Gradient clip (default: 10.0)
        gt_vel_obs_start_idx    int     Slice start in critic obs for gt_velocity
                                        (default: 0 — base_lin_vel is first in critic)
        gt_force_obs_start_idx  int     Slice start in critic obs for gt_force
                                        (default: -1 — disabled)
        num_estimator_mini_batches  int  Mini-batches for estimator update (default: 4)
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # ── Parse estimator config ────────────────────────────────────────
        est_cfg: dict = train_cfg.get("estimator", {})
        self._temporal_steps: int = est_cfg.get("temporal_steps", 50)
        self._gt_vel_start: int = est_cfg.get("gt_vel_obs_start_idx", 0)
        self._gt_force_start: int = est_cfg.get("gt_force_obs_start_idx", -1)
        self._force_dim: int = est_cfg.get("force_dim", 2)
        self._has_force: bool = self._gt_force_start >= 0
        self._num_est_mini_batches: int = est_cfg.get("num_estimator_mini_batches", 4)

        # Infer single-step obs dim from env (before augmentation)
        raw_obs = env.get_observations()
        self._num_one_step_obs: int = raw_obs["policy"].shape[-1]

        # ── Create estimator network ──────────────────────────────────────
        self.estimator = VelocityEstimator(
            temporal_steps=self._temporal_steps,
            num_one_step_obs=self._num_one_step_obs,
            enc_hidden_dims=est_cfg.get("enc_hidden_dims", [256, 128, 64]),
            v_head_dims=est_cfg.get("v_head_dims", [32, 16]),
            f_head_dims=est_cfg.get("f_head_dims", None),
            force_dim=self._force_dim,
            dec_hidden_dims=est_cfg.get("dec_hidden_dims", [512, 256, 128]),
            activation=est_cfg.get("activation", "elu"),
            learning_rate=est_cfg.get("learning_rate", 1e-3),
            vel_loss_weight=est_cfg.get("vel_loss_weight", 1.0),
            force_loss_weight=est_cfg.get("force_loss_weight", 1.0),
            rec_loss_weight=est_cfg.get("rec_loss_weight", 1.0),
            max_grad_norm=est_cfg.get("max_grad_norm", 10.0),
        ).to(device)

        heads = "v_head"
        if self.estimator.has_force_head:
            heads += " + f_head"
        print(
            f"[EstimatorRunner] Estimator ({heads}): "
            f"input={self._temporal_steps}×{self._num_one_step_obs}="
            f"{self._temporal_steps * self._num_one_step_obs}  "
            f"latent_dim={self.estimator.latent_dim}  "
            f"policy_input={self._num_one_step_obs + self.estimator.latent_dim}"
        )

        # ── Create obs history buffer ─────────────────────────────────────
        self._history_buffer = ObsHistoryBuffer(
            num_envs=env.num_envs,
            temporal_steps=self._temporal_steps,
            obs_dim=self._num_one_step_obs,
            device=device,
        )

        # ── Wrap env — augments "policy" obs with estimator latent ────────
        self._wrapped_env = EstimatorEnvWrapper(
            env=env,
            estimator=self.estimator,
            history_buffer=self._history_buffer,
            device=device,
        )

        # ── Call parent __init__ with the WRAPPED env ─────────────────────
        # Parent's _construct_algorithm calls self.env.get_observations()
        # which returns augmented obs → actor is built with correct input dim.
        super().__init__(self._wrapped_env, train_cfg, log_dir=log_dir, device=device)

        # ── Allocate estimator rollout buffers ────────────────────────────
        # Populated in learn() and consumed by _update_estimator().
        num_envs = env.num_envs
        history_flat_dim = self._temporal_steps * self._num_one_step_obs
        self._est_obs_history = torch.zeros(
            self.num_steps_per_env, num_envs, history_flat_dim, device=device
        )
        self._est_next_raw_obs = torch.zeros(
            self.num_steps_per_env, num_envs, self._num_one_step_obs, device=device
        )

        # Running stats for logging
        self._est_vel_loss_buf: deque = deque(maxlen=20)
        self._est_rec_loss_buf: deque = deque(maxlen=20)
        self._est_force_loss_buf: deque = deque(maxlen=20)

    # ── Main training loop ────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Full training loop with concurrent estimator updates."""
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # ── Rollout collection ─────────────────────────────────────────
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    # Store current obs_history BEFORE the step
                    self._est_obs_history[step] = self._history_buffer.get_flattened()

                    # Act
                    actions = self.alg.act(obs)

                    # Step environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # Store raw next obs (reconstruction target)
                    raw_next = self._wrapped_env.get_last_raw_policy_obs()
                    if raw_next is not None:
                        self._est_next_raw_obs[step] = raw_next

                    self.alg.process_env_step(obs, rewards, dones, extras)

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            # ── PPO update ─────────────────────────────────────────────────
            loss_dict = self.alg.update()

            # ── Estimator update ───────────────────────────────────────────
            est_losses = self._update_estimator()

            learn_time = time.time() - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self._log_with_estimator(locals(), est_losses)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ── Estimator update ──────────────────────────────────────────────────

    def _update_estimator(self) -> dict[str, float]:
        """Train the estimator on the just-collected rollout data."""
        # Flatten [num_steps, num_envs, ...] → [num_steps*num_envs, ...]
        n = self.num_steps_per_env * self.env.num_envs
        obs_h = self._est_obs_history.reshape(n, -1)
        next_o = self._est_next_raw_obs.reshape(n, -1)

        # Extract gt_velocity from critic obs stored in PPO buffer
        # critic obs layout (test bench): [base_lin_vel(3), base_ang_vel(3), ...]
        critic_obs = self.alg.storage.observations["critic"]  # [num_steps, num_envs, critic_dim]
        gt_v = critic_obs[:, :, self._gt_vel_start : self._gt_vel_start + 3].reshape(n, 3).detach()

        # Extract gt_force from critic obs (if force estimation enabled)
        gt_f = None
        if self._has_force and self.estimator.has_force_head:
            gt_f = critic_obs[
                :, :, self._gt_force_start : self._gt_force_start + self._force_dim
            ].reshape(n, self._force_dim).detach()

        # Mini-batch estimator updates
        batch_size = max(n // self._num_est_mini_batches, 1)
        indices = torch.randperm(n, device=self.device)
        vel_losses, rec_losses, force_losses = [], [], []

        self.estimator.train()
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            gt_f_batch = gt_f[idx] if gt_f is not None else None
            losses = self.estimator.update(obs_h[idx], gt_v[idx], next_o[idx], gt_force=gt_f_batch)
            vel_losses.append(losses["vel_loss"])
            rec_losses.append(losses["rec_loss"])
            if "force_loss" in losses:
                force_losses.append(losses["force_loss"])
        self.estimator.eval()

        mean_vel = statistics.mean(vel_losses)
        mean_rec = statistics.mean(rec_losses)
        self._est_vel_loss_buf.append(mean_vel)
        self._est_rec_loss_buf.append(mean_rec)

        result = {"vel_loss": mean_vel, "rec_loss": mean_rec}
        if force_losses:
            mean_force = statistics.mean(force_losses)
            self._est_force_loss_buf.append(mean_force)
            result["force_loss"] = mean_force

            # Bridge: write smoothed force loss to env extras for adaptive curriculum
            smooth_force_loss = (
                statistics.mean(self._est_force_loss_buf)
                if len(self._est_force_loss_buf) > 5
                else mean_force
            )
            self._wrapped_env._env.unwrapped.extras["estimator_force_loss_smooth"] = smooth_force_loss

        return result

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_with_estimator(self, locs: dict, est_losses: dict[str, float]) -> None:
        """Log PPO stats + estimator stats to tensorboard and terminal."""
        # Standard OnPolicyRunner log (handles reward, losses, FPS, …)
        self.log(locs)

        it = locs["it"]
        # Tensorboard
        self.writer.add_scalar("Estimator/vel_loss", est_losses["vel_loss"], it)
        self.writer.add_scalar("Estimator/rec_loss", est_losses["rec_loss"], it)
        if "force_loss" in est_losses:
            self.writer.add_scalar("Estimator/force_loss", est_losses["force_loss"], it)
        if len(self._est_vel_loss_buf) > 0:
            self.writer.add_scalar(
                "Estimator/vel_loss_smooth", statistics.mean(self._est_vel_loss_buf), it
            )
        if len(self._est_force_loss_buf) > 0:
            self.writer.add_scalar(
                "Estimator/force_loss_smooth", statistics.mean(self._est_force_loss_buf), it
            )

        # Terminal — append estimator block after normal log
        pad = 35
        est_str = (
            f"\n{'─' * 80}\n"
            f"{'[Estimator]':>{pad}} vel_loss={est_losses['vel_loss']:.5f}  "
            f"rec_loss={est_losses['rec_loss']:.5f}"
        )
        if "force_loss" in est_losses:
            est_str += f"  force_loss={est_losses['force_loss']:.5f}"
        if len(self._est_vel_loss_buf) >= 5:
            est_str += (
                f"  |  smooth_vel={statistics.mean(self._est_vel_loss_buf):.5f}"
            )
        if len(self._est_force_loss_buf) >= 5:
            est_str += (
                f"  smooth_frc={statistics.mean(self._est_force_loss_buf):.5f}"
            )
        print(est_str)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save policy + estimator weights to the same checkpoint file."""
        super().save(path, infos)
        # Append estimator state to the same file
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        ckpt["estimator_state_dict"] = self.estimator.state_dict()
        ckpt["estimator_optimizer_state_dict"] = self.estimator.optimizer.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load policy + estimator weights from checkpoint."""
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        ckpt = torch.load(path, weights_only=False, map_location=map_location)
        if "estimator_state_dict" in ckpt:
            self.estimator.load_state_dict(ckpt["estimator_state_dict"])
            if load_optimizer and "estimator_optimizer_state_dict" in ckpt:
                self.estimator.optimizer.load_state_dict(ckpt["estimator_optimizer_state_dict"])
            print("[EstimatorRunner] Estimator weights loaded from checkpoint.")
        else:
            print("[EstimatorRunner] No estimator weights found in checkpoint — starting fresh.")
        return infos

    # ── Mode helpers ──────────────────────────────────────────────────────

    def train_mode(self) -> None:
        super().train_mode()
        self.estimator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.estimator.eval()
