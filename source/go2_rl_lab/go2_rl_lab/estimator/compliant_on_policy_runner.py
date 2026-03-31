"""CompliantOnPolicyRunner — PPO + trainable force estimator + linear compliance mapping.

Extends OnPolicyRunner with three training phases:

Phase 1 (reward < threshold): Learn walking. Estimator trains from checkpoint.
    Forces = 0. Policy obs include force estimate (zeros initially).

Phase 2 (reward >= threshold): Forces activate. Estimator keeps training.
    Policy experiences forces, estimator learns the new gait's dynamics.

Phase 3 (angular error < 7°): Linear mapping activates.
    v* = v_cmd + k(f)*f_hat where k=0 if |f|<alpha else 1/beta.
    Standard tracking reward drives compliance — no separate r_comp needed.
"""

from __future__ import annotations

import math
import os
import statistics
import time
import torch
from collections import deque
from rsl_rl.runners import OnPolicyRunner

from go2_rl_lab.estimator.force_estimator import ForceEstimator
from go2_rl_lab.estimator.obs_history_buffer import ObsHistoryBuffer


class CompliantOnPolicyRunner(OnPolicyRunner):

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # ── Parse config ─────────────────────────────────────────────────
        est_cfg: dict = train_cfg.get("estimator", {})
        self._temporal_steps: int = est_cfg.get("temporal_steps", 20)
        self._force_dim: int = est_cfg.get("force_dim", 2)

        # Gate thresholds
        self._force_threshold: float = train_cfg.get("force_activation_reward_threshold", 30.0)
        self._angular_threshold: float = train_cfg.get("estimator_angular_threshold", 7.0)
        self._force_event_term: str = train_cfg.get("force_event_term_name", "persistent_xy_force")
        self._max_force: float = train_cfg.get("max_force", 20.0)

        # Compliance parameters
        self._compliance_alpha: float = train_cfg.get("compliance_alpha", 5.0)
        self._compliance_beta: float = train_cfg.get("compliance_beta", 50.0)

        # GT force location in critic obs (from force-only layout: indices 64:66)
        self._gt_force_start: int = est_cfg.get("gt_force_obs_start_idx", -1)

        # ── Initialize force estimate on env before first get_observations() ──
        # This ensures ForceEstimateObsTerm returns the correct dimension
        env.unwrapped._force_estimate_xy = torch.zeros(env.num_envs, self._force_dim, device=device)

        # ── Infer obs dim from env (before calling super) ────────────────
        raw_obs = env.get_observations()
        policy_obs_dim = raw_obs["policy"].shape[-1]
        # The last force_dim dims are the force estimate passthrough (zeros initially)
        self._num_one_step_obs: int = policy_obs_dim - self._force_dim

        # ── Create force estimator ───────────────────────────────────────
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
        ).to(device)

        # ── Load estimator from checkpoint if provided ───────────────────
        ckpt_path = train_cfg.get("estimator_checkpoint", "")
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "force_estimator_state_dict" in ckpt:
                self.estimator.load_state_dict(ckpt["force_estimator_state_dict"])
                print(f"[CompliantRunner] Loaded estimator from: {ckpt_path}")
            else:
                print(f"[CompliantRunner] WARNING: No estimator weights in {ckpt_path}")

        # ── History buffer ───────────────────────────────────────────────
        self._history_buffer = ObsHistoryBuffer(
            num_envs=env.num_envs,
            temporal_steps=self._temporal_steps,
            obs_dim=self._num_one_step_obs,
            device=device,
        )

        # ── Call parent __init__ ─────────────────────────────────────────
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # ── Estimator rollout buffers ────────────────────────────────────
        num_envs = env.num_envs
        history_flat_dim = self._temporal_steps * self._num_one_step_obs
        self._est_obs_history = torch.zeros(
            self.num_steps_per_env, num_envs, history_flat_dim, device=device
        )
        self._est_next_raw_obs = torch.zeros(
            self.num_steps_per_env, num_envs, self._num_one_step_obs, device=device
        )

        # ── State ────────────────────────────────────────────────────────
        self._force_active: bool = False
        self._mapping_active: bool = False
        self._est_loss_buf: deque = deque(maxlen=20)
        self._last_est_stats: dict = {}

        print(
            f"[CompliantRunner] Phase gates: "
            f"forces @ reward>={self._force_threshold:.0f}, "
            f"mapping @ angular_err<{self._angular_threshold:.0f}°\n"
            f"  Compliance: alpha={self._compliance_alpha:.1f}N, "
            f"beta={self._compliance_beta:.1f} (k=1/beta={1.0/self._compliance_beta:.4f})\n"
            f"  Estimator: {self._temporal_steps}x{self._num_one_step_obs}="
            f"{self._temporal_steps * self._num_one_step_obs} → force_dim={self._force_dim}"
        )

    # ── Main training loop ────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
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

            # ── Rollout ──────────────────────────────────────────────────
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    # Extract raw 61-dim obs (without force estimate at end)
                    raw_obs = obs["policy"][:, :self._num_one_step_obs]

                    # Update history and run estimator inference
                    self._history_buffer.insert(raw_obs)
                    self._est_obs_history[step] = self._history_buffer.get_flattened()
                    force_hat, _ = self.estimator.get_latent(self._history_buffer.get_flattened())

                    # Zero out force estimate before estimator is trained (phase 1)
                    # to avoid random noise confusing the policy
                    if not self._force_active:
                        force_hat = force_hat * 0.0

                    # Set force estimate on env (obs term reads it, reward reads it)
                    self.env.unwrapped._force_estimate_xy = force_hat

                    # Act
                    actions = self.alg.act(obs)

                    # Step environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # Store next raw obs for reconstruction target
                    self._est_next_raw_obs[step] = obs["policy"][:, :self._num_one_step_obs]

                    # Reset history for terminated envs
                    done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids) > 0:
                        self._history_buffer.reset(done_ids)

                    self.alg.process_env_step(obs, rewards, dones, extras)

                    # Book keeping
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

                self.alg.compute_returns(obs)

            # ── PPO update ───────────────────────────────────────────────
            loss_dict = self.alg.update()

            # ── Estimator update (train on rollout data) ─────────────────
            if self._force_active:
                est_stats = self._train_estimator()
                self._last_est_stats = est_stats
                if "force_loss" in est_stats:
                    self._est_loss_buf.append(est_stats["force_loss"])

            # ── Phase gates ──────────────────────────────────────────────
            self._check_force_gate(rewbuffer, it)
            self._check_mapping_gate(it)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

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
        """Train the estimator on collected rollout data (same as ForceOnPolicyRunner)."""
        isaac_env = self.env.unwrapped

        # Get GT force from wrench composer (supports both 2D and 3D)
        asset = isaac_env.scene["robot"]
        gt_force = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :self._force_dim]

        # Use stored rollout data
        num_steps = self.num_steps_per_env
        num_envs = self.env.num_envs

        # Flatten rollout: [steps * envs, ...]
        obs_hist_flat = self._est_obs_history.reshape(num_steps * num_envs, -1)
        next_obs_flat = self._est_next_raw_obs.reshape(num_steps * num_envs, -1)

        # GT force: use current force for all steps (approximation — force is persistent)
        gt_force_flat = gt_force.unsqueeze(0).expand(num_steps, -1, -1).reshape(num_steps * num_envs, -1)

        # Mini-batch training (4 mini-batches, same as PPO)
        batch_size = obs_hist_flat.shape[0]
        mini_batch_size = batch_size // 4
        indices = torch.randperm(batch_size, device=self.device)

        total_stats = {}
        for i in range(4):
            idx = indices[i * mini_batch_size: (i + 1) * mini_batch_size]
            stats = self.estimator.update(
                obs_hist_flat[idx],
                gt_force_flat[idx],
                next_obs_flat[idx],
            )
            if not total_stats:
                total_stats = {k: v for k, v in stats.items()}
            else:
                for k, v in stats.items():
                    total_stats[k] = (total_stats[k] + v) / 2  # running average

        return total_stats

    # ── Phase gates ───────────────────────────────────────────────────────

    def _check_force_gate(self, rewbuffer: deque, it: int) -> None:
        """Phase 1 → 2: Activate forces when mean episode reward >= threshold."""
        if self._force_active or len(rewbuffer) < 10:
            return
        mean_rew = statistics.mean(rewbuffer)
        if mean_rew >= self._force_threshold:
            self._force_active = True
            isaac_env = self.env.unwrapped
            event_cfg = isaac_env.event_manager.get_term_cfg(self._force_event_term)
            event_cfg.params["force_range"] = (0.0, self._max_force)
            print(
                f"\n{'=' * 80}\n"
                f"  [CompliantRunner] PHASE 2: Mean reward {mean_rew:.1f} >= {self._force_threshold:.1f}\n"
                f"  Forces activated at {self._max_force:.0f}N. Estimator training begins.\n"
                f"{'=' * 80}"
            )

    def _check_mapping_gate(self, it: int) -> None:
        """Phase 2 → 3: Activate linear mapping when estimator angular error < threshold."""
        if self._mapping_active or not self._force_active:
            return
        angle_err = self._last_est_stats.get("angle_err_median_deg", 999.0)
        if angle_err < self._angular_threshold:
            self._mapping_active = True
            isaac_env = self.env.unwrapped
            isaac_env._mapping_active = True
            isaac_env._compliance_alpha = self._compliance_alpha
            isaac_env._compliance_beta = self._compliance_beta
            print(
                f"\n{'=' * 80}\n"
                f"  [CompliantRunner] PHASE 3: Angular error {angle_err:.1f}° < {self._angular_threshold:.1f}°\n"
                f"  Linear mapping activated: k(f) = 0 if |f|<{self._compliance_alpha:.1f} "
                f"else 1/{self._compliance_beta:.1f}\n"
                f"{'=' * 80}"
            )

    # ── Logging ───────────────────────────────────────────────────────────

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)

        it = locs["it"]
        rewbuffer = locs["rewbuffer"]
        isaac_env = self.env.unwrapped

        # ── Phase status ─────────────────────────────────────────────────
        if self._mapping_active:
            phase = "PHASE 3 (mapping)"
        elif self._force_active:
            phase = "PHASE 2 (forces+estimator)"
        else:
            phase = "PHASE 1 (walking)"

        self.writer.add_scalar("Compliant/phase", 3 if self._mapping_active else (2 if self._force_active else 1), it)
        self.writer.add_scalar("Compliant/force_active", float(self._force_active), it)
        self.writer.add_scalar("Compliant/mapping_active", float(self._mapping_active), it)

        # ── Progress toward force gate ───────────────────────────────────
        if len(rewbuffer) > 0:
            mean_rew = statistics.mean(rewbuffer)
            pct = min(100.0, mean_rew / self._force_threshold * 100.0)
            self.writer.add_scalar("Compliant/reward_progress_pct", pct, it)

        # ── Estimator stats (when training) ──────────────────────────────
        est = self._last_est_stats
        if est:
            for key in ["force_loss", "angle_loss", "rec_loss", "mae_total",
                        "mae_x", "mae_y", "mae_z", "angle_err_mean_deg", "angle_err_median_deg",
                        "gt_force_mean_mag", "pred_force_mean_mag"]:
                if key in est:
                    self.writer.add_scalar(f"Estimator/{key}", est[key], it)
            if len(self._est_loss_buf) > 0:
                self.writer.add_scalar(
                    "Estimator/force_loss_smooth", statistics.mean(self._est_loss_buf), it
                )

        # ── Force magnitude ──────────────────────────────────────────────
        if self._force_active:
            asset = isaac_env.scene["robot"]
            f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :self._force_dim]
            f_mags = f.norm(dim=1)
            self.writer.add_scalar("Compliant/force_magnitude_mean", f_mags.mean().item(), it)

        # ── Terminal output ──────────────────────────────────────────────
        term_str = f"  [{phase}]"

        if not self._force_active and len(rewbuffer) > 0:
            mean_rew = statistics.mean(rewbuffer)
            pct = min(100.0, mean_rew / self._force_threshold * 100.0)
            term_str += f"  reward={mean_rew:.1f}/{self._force_threshold:.1f} ({pct:.0f}%)"

        if self._force_active and est:
            ang = est.get("angle_err_median_deg", 0)
            mae = est.get("mae_total", 0)
            floss = est.get("force_loss", 0)
            term_str += f"  est: loss={floss:.4f} mae={mae:.3f} ang={ang:.1f}°"
            if not self._mapping_active:
                ang_pct = max(0, 100.0 * (1.0 - ang / self._angular_threshold)) if ang < 90 else 0
                term_str += f" ({ang_pct:.0f}% to mapping)"

        if self._mapping_active:
            asset = isaac_env.scene["robot"]
            f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :self._force_dim]
            f_mean = f.norm(dim=1).mean().item()
            term_str += f"  |f|={f_mean:.1f}N  alpha={self._compliance_alpha:.1f} beta={self._compliance_beta:.1f}"

        print(term_str)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        ckpt["force_estimator_state_dict"] = self.estimator.state_dict()
        ckpt["force_estimator_optimizer_state_dict"] = self.estimator.optimizer.state_dict()
        ckpt["compliant_state"] = {
            "force_active": self._force_active,
            "mapping_active": self._mapping_active,
        }
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        ckpt = torch.load(path, weights_only=False, map_location=map_location)
        if "force_estimator_state_dict" in ckpt:
            self.estimator.load_state_dict(ckpt["force_estimator_state_dict"])
            if load_optimizer and "force_estimator_optimizer_state_dict" in ckpt:
                self.estimator.optimizer.load_state_dict(ckpt["force_estimator_optimizer_state_dict"])
            print("[CompliantRunner] Estimator weights loaded from checkpoint.")
        state = ckpt.get("compliant_state", {})
        if state.get("force_active"):
            self._force_active = True
            isaac_env = self.env.unwrapped
            event_cfg = isaac_env.event_manager.get_term_cfg(self._force_event_term)
            event_cfg.params["force_range"] = (0.0, self._max_force)
        if state.get("mapping_active"):
            self._mapping_active = True
            isaac_env = self.env.unwrapped
            isaac_env._mapping_active = True
            isaac_env._compliance_alpha = self._compliance_alpha
            isaac_env._compliance_beta = self._compliance_beta
        return infos

    # ── Mode helpers ──────────────────────────────────────────────────────

    def train_mode(self) -> None:
        super().train_mode()
        self.estimator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.estimator.eval()
