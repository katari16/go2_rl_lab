"""FrozenPolicyEstimatorRunner — train force estimator on a frozen walking policy.

Stage 2 estimator-only training. The base policy has 57-dim obs (no force
estimate slot), so the estimator is a pure side-channel — its output never
feeds back into the policy. Training loop:

  1. Load a pre-trained 57-dim policy checkpoint via --locomotion_checkpoint.
  2. Freeze all actor params; actor.eval(); skip PPO updates.
  3. Verify the actor checksum every 100 iters (catches any accidental drift).
  4. Activate forces immediately at max_force (no reward gate).
  5. A fraction of envs (force_free_fraction) always get zero force so the
     estimator learns to output zeros when nothing is pulling.
  6. Train only the force estimator via supervised loss against GT wrench.
"""

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from rsl_rl.runners import OnPolicyRunner

from go2_rl_lab.estimator.force_estimator import ForceEstimator
from go2_rl_lab.estimator.obs_history_buffer import ObsHistoryBuffer


class FrozenPolicyEstimatorRunner(OnPolicyRunner):

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        est_cfg: dict = train_cfg.get("estimator", {})
        self._temporal_steps: int = est_cfg.get("temporal_steps", 30)
        self._force_dim: int = est_cfg.get("force_dim", 3)
        self._force_free_fraction: float = train_cfg.get("force_free_fraction", 0.1)
        self._force_event_term: str = train_cfg.get("force_event_term_name", "persistent_xyz_force")
        self._max_force: float = train_cfg.get("max_force", 50.0)
        self._max_torque: float = train_cfg.get("max_torque", 5.0)

        # 57-dim policy obs has no force estimate slot — full obs is raw history.
        raw_obs = env.get_observations()
        self._num_one_step_obs: int = raw_obs["policy"].shape[-1]

        self.estimator = ForceEstimator(
            temporal_steps=self._temporal_steps,
            num_one_step_obs=self._num_one_step_obs,
            enc_hidden_dims=est_cfg.get("enc_hidden_dims", [128, 64]),
            f_head_dims=est_cfg.get("f_head_dims", [32, 16]),
            force_dim=self._force_dim,
            dec_hidden_dims=est_cfg.get("dec_hidden_dims", [256, 128]),
            activation=est_cfg.get("activation", "elu"),
            learning_rate=est_cfg.get("learning_rate", 1e-3),
            force_loss_weight=est_cfg.get("force_loss_weight", 1.0),
            angle_loss_weight=est_cfg.get("angle_loss_weight", 1.0),
            rec_loss_weight=est_cfg.get("rec_loss_weight", 1.0),
            angle_min_force=est_cfg.get("angle_min_force", 1.0),
            max_grad_norm=est_cfg.get("max_grad_norm", 10.0),
            torque_angle_loss_weight=est_cfg.get("torque_angle_loss_weight", 0.0),
            torque_angle_min=est_cfg.get("torque_angle_min", 0.3),
            yaw_loss_weight=est_cfg.get("yaw_loss_weight", 0.0),
            tcn_mode=est_cfg.get("tcn_mode", "none"),
            tcn_channels=est_cfg.get("tcn_channels", None),
            tcn_kernel_size=est_cfg.get("tcn_kernel_size", 3),
            tcn_dilations=est_cfg.get("tcn_dilations", None),
            temporal_decay=est_cfg.get("temporal_decay", "none"),
            force_layout=est_cfg.get("force_layout", "auto"),
        ).to(device)

        self._history_buffer = ObsHistoryBuffer(
            num_envs=env.num_envs,
            temporal_steps=self._temporal_steps,
            obs_dim=self._num_one_step_obs,
            device=device,
        )

        # Parent init creates the PPO algorithm + actor-critic
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # Rollout buffers for estimator training
        num_envs = env.num_envs
        history_flat_dim = self._temporal_steps * self._num_one_step_obs
        self._est_obs_history = torch.zeros(
            self.num_steps_per_env, num_envs, history_flat_dim, device=device
        )
        self._est_next_raw_obs = torch.zeros(
            self.num_steps_per_env, num_envs, self._num_one_step_obs, device=device
        )

        self._est_loss_buf: deque = deque(maxlen=20)
        self._last_est_stats: dict = {}
        self._actor_checksum: torch.Tensor | None = None
        self._checksum_interval: int = 100

        # Pre-compute force-free env mask (fixed set, resampled each force interval)
        self._num_force_free = int(num_envs * self._force_free_fraction)

        print(
            f"[FrozenEstimatorRunner] Estimator-only training (policy frozen)\n"
            f"  Forces: {self._max_force:.0f}N, force_free_fraction={self._force_free_fraction:.0%} "
            f"({self._num_force_free}/{num_envs} envs)\n"
            f"  Estimator: {self._temporal_steps}x{self._num_one_step_obs}="
            f"{self._temporal_steps * self._num_one_step_obs} → force_dim={self._force_dim}"
        )

    def _activate_forces(self) -> None:
        """Activate forces on the environment immediately."""
        isaac_env = self.env.unwrapped
        event_cfg = isaac_env.event_manager.get_term_cfg(self._force_event_term)
        event_cfg.params["force_range"] = (0.0, self._max_force)
        if "torque_range" in event_cfg.params:
            event_cfg.params["torque_range"] = (0.0, self._max_torque)
        event_cfg.params["force_free_fraction"] = self._force_free_fraction
        print(f"  Forces activated: {self._max_force:.0f}N, event={self._force_event_term}")

    def _get_policy_nn(self):
        try:
            return self.alg.policy
        except AttributeError:
            return self.alg.actor_critic

    def _freeze_actor(self) -> None:
        """Freeze all actor (policy) parameters so PPO doesn't update them."""
        policy_nn = self._get_policy_nn()
        frozen_count = 0
        for _, param in policy_nn.named_parameters():
            param.requires_grad = False
            frozen_count += 1
        print(f"  Actor frozen: {frozen_count} parameters")
        self._actor_checksum = self._compute_actor_checksum()

    def _compute_actor_checksum(self) -> torch.Tensor:
        policy_nn = self._get_policy_nn()
        with torch.no_grad():
            parts = [p.detach().float().sum() for p in policy_nn.parameters()]
            return torch.stack(parts).sum().cpu()

    def _verify_actor_frozen(self, it: int) -> None:
        if self._actor_checksum is None:
            return
        current = self._compute_actor_checksum()
        if not torch.allclose(current, self._actor_checksum, atol=0.0):
            raise RuntimeError(
                f"[FrozenEstimatorRunner] Actor weights changed at iter {it}! "
                f"checksum drift: {(current - self._actor_checksum).item():.6e}"
            )

    # ── Main training loop ────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Activate forces and freeze actor before first iteration
        self._activate_forces()
        self._freeze_actor()

        obs = self.env.get_observations().to(self.device)
        self.estimator.train()
        # Actor in eval mode (frozen, no dropout/batchnorm effects)
        self.alg.policy.eval() if hasattr(self.alg, 'policy') else self.alg.actor_critic.eval()

        ep_infos = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # ── Rollout (frozen policy, collect estimator data) ──────────
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    raw_obs = obs["policy"][:, :self._num_one_step_obs]

                    self._history_buffer.insert(raw_obs)
                    self._est_obs_history[step] = self._history_buffer.get_flattened()

                    actions = self.alg.act(obs)

                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device), rewards.to(self.device), dones.to(self.device),
                    )

                    self._est_next_raw_obs[step] = obs["policy"][:, :self._num_one_step_obs]

                    done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids) > 0:
                        self._history_buffer.reset(done_ids)

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

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns (needed for alg.act bookkeeping even though we skip PPO)
                self.alg.compute_returns(obs)
                # We skip alg.update() (which would clear the rollout storage),
                # so clear it manually to prevent overflow on the next rollout.
                self.alg.storage.clear()

            # ── Skip PPO update — only train estimator ──────────────────
            if it % self._checksum_interval == 0:
                self._verify_actor_frozen(it)

            est_stats = self._train_estimator()
            self._last_est_stats = est_stats
            if "force_loss" in est_stats:
                self._est_loss_buf.append(est_stats["force_loss"])

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Parent log() expects PPO's loss_dict — provide an empty one since we skip PPO.
            loss_dict: dict = {}

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

            if it == start_iter and not self.disable_logs:
                from rsl_rl.utils import store_code_state
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ── Estimator training ────────────────────────────────────────────────

    def _train_estimator(self) -> dict:
        isaac_env = self.env.unwrapped
        asset = isaac_env.scene["robot"]
        force_layout = self.estimator.force_layout

        if force_layout == "xy_yaw":
            gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :2]
            gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:, 0, 2:3]
            gt_force = torch.cat([gt_f, gt_t], dim=-1)
        elif self._force_dim <= 3:
            gt_force = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :self._force_dim]
        elif self._force_dim == 4:
            gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :3]
            gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:, 0, 2:3]
            gt_force = torch.cat([gt_f, gt_t], dim=-1)
        else:
            gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :3]
            gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:, 0, :3]
            gt_force = torch.cat([gt_f, gt_t], dim=-1)

        num_steps = self.num_steps_per_env
        num_envs = self.env.num_envs

        obs_hist_flat = self._est_obs_history.reshape(num_steps * num_envs, -1)
        next_obs_flat = self._est_next_raw_obs.reshape(num_steps * num_envs, -1)
        gt_force_flat = gt_force.unsqueeze(0).expand(num_steps, -1, -1).reshape(num_steps * num_envs, -1)

        batch_size = obs_hist_flat.shape[0]
        mini_batch_size = batch_size // 4
        indices = torch.randperm(batch_size, device=self.device)

        total_stats = {}
        for i in range(4):
            idx = indices[i * mini_batch_size: (i + 1) * mini_batch_size]
            stats = self.estimator.update(
                obs_hist_flat[idx], gt_force_flat[idx], next_obs_flat[idx],
            )
            if not total_stats:
                total_stats = {k: v for k, v in stats.items()}
            else:
                for k, v in stats.items():
                    total_stats[k] = (total_stats[k] + v) / 2

        return total_stats

    # ── Logging ───────────────────────────────────────────────────────────

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)

        it = locs["it"]
        isaac_env = self.env.unwrapped
        est = self._last_est_stats

        if est:
            for key in ["force_loss", "angle_loss", "rec_loss", "mae_total",
                        "mae_x", "mae_y", "mae_z",
                        "mae_tau_roll", "mae_tau_pitch", "mae_tau_yaw",
                        "torque_angle_loss", "yaw_loss",
                        "angle_err_mean_deg", "angle_err_median_deg",
                        "gt_force_mean_mag", "pred_force_mean_mag",
                        "grad_norm_encoder", "grad_norm_f_head",
                        "grad_norm_decoder", "grad_norm_tcn"]:
                if key in est:
                    self.writer.add_scalar(f"Estimator/{key}", est[key], it)
            if len(self._est_loss_buf) > 0:
                self.writer.add_scalar(
                    "Estimator/force_loss_smooth", statistics.mean(self._est_loss_buf), it
                )

        asset = isaac_env.scene["robot"]
        f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :3]
        f_mags = f.norm(dim=1)
        self.writer.add_scalar("Compliant/force_magnitude_mean", f_mags.mean().item(), it)

        ang = est.get("angle_err_median_deg", 0)
        mae = est.get("mae_total", 0)
        floss = est.get("force_loss", 0)
        print(f"  [FROZEN] est: loss={floss:.4f} mae={mae:.3f} ang={ang:.1f}° |f|={f_mags.mean():.1f}N")

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        ckpt["force_estimator_state_dict"] = self.estimator.state_dict()
        ckpt["force_estimator_optimizer_state_dict"] = self.estimator.optimizer.state_dict()
        ckpt["frozen_estimator_state"] = {
            "force_free_fraction": self._force_free_fraction,
            "max_force": self._max_force,
        }
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        ckpt = torch.load(path, weights_only=False, map_location=map_location)
        if "force_estimator_state_dict" in ckpt:
            self.estimator.load_state_dict(ckpt["force_estimator_state_dict"])
            if load_optimizer and "force_estimator_optimizer_state_dict" in ckpt:
                self.estimator.optimizer.load_state_dict(ckpt["force_estimator_optimizer_state_dict"])
            print("[FrozenEstimatorRunner] Estimator weights loaded from checkpoint.")
        return infos

    def train_mode(self) -> None:
        self.estimator.train()

    def eval_mode(self) -> None:
        self.estimator.eval()
        self.alg.policy.eval() if hasattr(self.alg, 'policy') else self.alg.actor_critic.eval()
