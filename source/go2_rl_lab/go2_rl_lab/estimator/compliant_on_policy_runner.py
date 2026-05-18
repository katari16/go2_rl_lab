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
        self._max_torque: float = train_cfg.get("max_torque", 5.0)

        # Force-gate mode: "total" (default, current), "excluded", "tracking"
        self._force_gate_mode: str = train_cfg.get("force_gate_mode", "total")
        self._force_gate_excluded: tuple = tuple(train_cfg.get("force_gate_excluded_terms", ()))
        self._force_gate_tracking: dict = dict(train_cfg.get("force_gate_tracking_thresholds", {}))
        self._force_gate_dwell_iters: int = train_cfg.get("force_gate_dwell_iters", 50)
        self._force_gate_min_long_envs: int = train_cfg.get("force_gate_min_long_envs", 100)
        self._force_gate_min_episode_steps: int = train_cfg.get("force_gate_min_episode_steps", 200)
        self._force_gate_dwell_count: int = 0

        # Post-gate curriculum shape: "none" (step jump to max, default = R1),
        # "linear" (linear ramp from ramp_start to max over ramp_iters),
        # "bucketed" (step through bucket_maxes, holding each for bucket_iters).
        # Torque max scales proportionally to force max.
        self._force_ramp_mode: str = train_cfg.get("force_ramp_mode", "none")
        self._force_ramp_start: float = float(train_cfg.get("force_ramp_start", 10.0))
        self._force_ramp_iters: int = int(train_cfg.get("force_ramp_iters", 2500))
        self._force_bucket_maxes: tuple = tuple(train_cfg.get("force_bucket_maxes", (10.0, 20.0, 30.0)))
        self._force_bucket_iters: int = int(train_cfg.get("force_bucket_iters", 1000))
        self._activation_iter: int = -1

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
        # The policy obs ends with a force_dim estimate passthrough; strip it for estimator input
        self._policy_raw_dim: int = policy_obs_dim - self._force_dim

        # Privileged-info extension (observability ablations R9/R10/R11/R12).
        # Each name maps to a dim contributed to the estimator's per-step input.
        # Supported: "mass" (1), "lin_vel" (3), "contacts" (4).
        priv_features = tuple(train_cfg.get("privileged_features", ()))
        priv_dims = {"mass": 1, "lin_vel": 3, "contacts": 4}
        for name in priv_features:
            if name not in priv_dims:
                raise ValueError(f"Unknown privileged feature: {name!r}. Options: {list(priv_dims)}")
        self._priv_features: tuple = priv_features
        self._priv_dim: int = sum(priv_dims[n] for n in priv_features)

        # Final per-step input dim seen by the estimator: policy obs + privileged
        self._num_one_step_obs: int = self._policy_raw_dim + self._priv_dim

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

        priv_desc = (
            f"  Privileged estimator inputs: {self._priv_features} (+{self._priv_dim} dims)\n"
            if self._priv_dim > 0 else ""
        )
        print(
            f"[CompliantRunner] Phase gates: "
            f"forces @ reward>={self._force_threshold:.0f}, "
            f"mapping @ angular_err<{self._angular_threshold:.0f}°\n"
            f"  Compliance: alpha={self._compliance_alpha:.1f}N, "
            f"beta={self._compliance_beta:.1f} (k=1/beta={1.0/self._compliance_beta:.4f})\n"
            f"{priv_desc}"
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
                    # Policy obs without the trailing force estimate; optionally
                    # concatenate privileged features for the estimator only.
                    policy_raw = obs["policy"][:, :self._policy_raw_dim]
                    if self._priv_dim > 0:
                        priv = self._get_privileged_obs()
                        raw_obs = torch.cat([policy_raw, priv], dim=-1)
                    else:
                        raw_obs = policy_raw

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

                    # Store next raw obs for reconstruction target (matches estimator input dim)
                    next_policy_raw = obs["policy"][:, :self._policy_raw_dim]
                    if self._priv_dim > 0:
                        next_priv = self._get_privileged_obs()
                        self._est_next_raw_obs[step] = torch.cat([next_policy_raw, next_priv], dim=-1)
                    else:
                        self._est_next_raw_obs[step] = next_policy_raw

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
            self._update_force_ramp(it)
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

    # ── Privileged feature extraction ────────────────────────────────────

    def _get_privileged_obs(self) -> torch.Tensor:
        """Concatenate privileged features (mass, lin_vel, contacts) for the estimator.

        Used by the observability ablations (R9/R10/R11/R12). Order matches
        ``self._priv_features`` so the estimator input layout is deterministic.
        Returns shape [num_envs, self._priv_dim].
        """
        isaac_env = self.env.unwrapped
        asset = isaac_env.scene["robot"]
        parts = []
        for name in self._priv_features:
            if name == "mass":
                # Total robot mass (sum over bodies), per env. Shape [num_envs, 1].
                masses = asset.root_physx_view.get_masses().to(self.device)
                parts.append(masses.sum(dim=-1, keepdim=True))
            elif name == "lin_vel":
                # GT base linear velocity in body frame. Shape [num_envs, 3].
                parts.append(asset.data.root_lin_vel_b.to(self.device))
            elif name == "contacts":
                # Foot contact force norms (FL/FR/RL/RR), scaled. Shape [num_envs, 4].
                sensor = isaac_env.scene.sensors["contact_forces"]
                # Match the critic's scaling (0.01) to keep magnitudes comparable
                foot_bodies = [i for i, name_ in enumerate(sensor.body_names) if name_.endswith("_foot")]
                forces = sensor.data.net_forces_w[:, foot_bodies, :]
                parts.append(torch.norm(forces, dim=-1).to(self.device) * 0.01)
            else:
                raise ValueError(f"Unhandled privileged feature: {name}")
        return torch.cat(parts, dim=-1)

    # ── Estimator training ────────────────────────────────────────────────

    def _train_estimator(self) -> dict:
        """Train the estimator on collected rollout data (same as ForceOnPolicyRunner)."""
        isaac_env = self.env.unwrapped

        # Get GT force/wrench from wrench composer
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
        """Phase 1 → 2: dispatch to the configured gating strategy."""
        if self._force_active:
            return
        mode = self._force_gate_mode
        if mode == "total":
            triggered, summary = self._force_gate_total(rewbuffer)
        elif mode == "excluded":
            triggered, summary = self._force_gate_excluded_sum()
        elif mode == "tracking":
            triggered, summary = self._force_gate_tracking_pct()
        else:
            raise ValueError(f"Unknown force_gate_mode: {mode}")
        if not triggered:
            return
        self._activate_forces(summary, it)

    def _activate_forces(self, summary: str, it: int) -> None:
        self._force_active = True
        self._activation_iter = it
        isaac_env = self.env.unwrapped
        event_cfg = isaac_env.event_manager.get_term_cfg(self._force_event_term)

        # Pick initial force max based on curriculum mode
        if self._force_ramp_mode == "linear":
            initial_force = min(self._force_ramp_start, self._max_force)
        elif self._force_ramp_mode == "bucketed":
            initial_force = min(self._force_bucket_maxes[0], self._max_force)
        else:
            initial_force = self._max_force

        initial_torque = (initial_force / max(self._max_force, 1e-6)) * self._max_torque
        event_cfg.params["force_range"] = (0.0, initial_force)
        if "torque_range" in event_cfg.params:
            event_cfg.params["torque_range"] = (0.0, initial_torque)

        sched = self._force_ramp_mode
        if sched == "linear":
            sched_desc = (f"linear ramp {self._force_ramp_start:.0f}→{self._max_force:.0f}N "
                          f"over {self._force_ramp_iters} iters")
        elif sched == "bucketed":
            sched_desc = (f"bucketed {self._force_bucket_maxes} N × {self._force_bucket_iters} iters/bucket")
        else:
            sched_desc = f"step jump to {self._max_force:.0f}N"

        print(
            f"\n{'=' * 80}\n"
            f"  [CompliantRunner] PHASE 2 ({self._force_gate_mode}, iter {it}): {summary}\n"
            f"  Force schedule: {sched_desc}\n"
            f"  Initial: forces (0, {initial_force:.1f})N"
            + (f", torques (0, {initial_torque:.2f})Nm" if "torque_range" in event_cfg.params else "")
            + f"\n  Estimator training begins.\n"
            f"{'=' * 80}"
        )

    def _update_force_ramp(self, it: int) -> None:
        """Per-iteration update of force/torque ranges according to curriculum schedule.

        No-op for ``force_ramp_mode == "none"`` (default R1 step-jump behavior).
        """
        if not self._force_active or self._activation_iter < 0:
            return
        if self._force_ramp_mode == "none":
            return

        iters_since_gate = it - self._activation_iter

        if self._force_ramp_mode == "linear":
            ramp_iters = max(self._force_ramp_iters, 1)
            progress = min(max(iters_since_gate / ramp_iters, 0.0), 1.0)
            new_force = self._force_ramp_start + progress * (self._max_force - self._force_ramp_start)
        elif self._force_ramp_mode == "bucketed":
            bucket_idx = min(iters_since_gate // max(self._force_bucket_iters, 1),
                             len(self._force_bucket_maxes) - 1)
            new_force = float(self._force_bucket_maxes[int(bucket_idx)])
            new_force = min(new_force, self._max_force)
        else:
            raise ValueError(f"Unknown force_ramp_mode: {self._force_ramp_mode}")

        new_torque = (new_force / max(self._max_force, 1e-6)) * self._max_torque

        isaac_env = self.env.unwrapped
        event_cfg = isaac_env.event_manager.get_term_cfg(self._force_event_term)
        event_cfg.params["force_range"] = (0.0, new_force)
        if "torque_range" in event_cfg.params:
            event_cfg.params["torque_range"] = (0.0, new_torque)

        # Expose for tensorboard logging
        isaac_env.extras["Curriculum/force_max"] = new_force
        isaac_env.extras["Curriculum/torque_max"] = new_torque

    # -- Mode "total" -----------------------------------------------------
    def _force_gate_total(self, rewbuffer: deque):
        if len(rewbuffer) < 10:
            return False, ""
        mean_rew = statistics.mean(rewbuffer)
        if mean_rew >= self._force_threshold:
            return True, f"Mean reward {mean_rew:.1f} >= {self._force_threshold:.1f}"
        return False, ""

    # -- Mode "excluded" --------------------------------------------------
    def _force_gate_excluded_sum(self):
        isaac_env = self.env.unwrapped
        long_running = isaac_env.episode_length_buf >= self._force_gate_min_episode_steps
        n_long = int(long_running.sum().item())
        if n_long < self._force_gate_min_long_envs:
            return False, ""
        total = torch.zeros(isaac_env.num_envs, device=isaac_env.device)
        for name, sums in isaac_env.reward_manager._episode_sums.items():
            if name in self._force_gate_excluded:
                continue
            total += sums
        ep_lens = isaac_env.episode_length_buf[long_running].float()
        per_step = total[long_running] / ep_lens
        mean_ep = (per_step.mean() * isaac_env.max_episode_length).item()
        if mean_ep >= self._force_threshold:
            return True, (f"Mean reward (excl. {list(self._force_gate_excluded)}) "
                          f"{mean_ep:.1f} >= {self._force_threshold:.1f}")
        return False, ""

    # -- Mode "tracking" --------------------------------------------------
    def _force_gate_tracking_pct(self):
        isaac_env = self.env.unwrapped
        long_running = isaac_env.episode_length_buf >= self._force_gate_min_episode_steps
        n_long = int(long_running.sum().item())
        if n_long < self._force_gate_min_long_envs:
            self._force_gate_dwell_count = 0
            return False, ""
        rm = isaac_env.reward_manager
        weights = dict(zip(rm._term_names, [c.weight for c in rm._term_cfgs]))
        ep_lens = isaac_env.episode_length_buf[long_running].float()
        all_above = True
        details = []
        for term_name, frac in self._force_gate_tracking.items():
            sums = rm._episode_sums.get(term_name)
            w = weights.get(term_name, 0.0)
            if sums is None or w == 0.0:
                all_above = False
                details.append(f"{term_name}=missing")
                break
            per_step_pct = (sums[long_running] / ep_lens).mean().item() / w
            details.append(f"{term_name}={per_step_pct:.2f}/{frac:.2f}")
            if per_step_pct < frac:
                all_above = False
        if all_above:
            self._force_gate_dwell_count += 1
        else:
            self._force_gate_dwell_count = 0
        if self._force_gate_dwell_count >= self._force_gate_dwell_iters:
            return True, (f"Tracking thresholds met for "
                          f"{self._force_gate_dwell_count} iters | " + ", ".join(details))
        return False, ""

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

        # ── Force magnitude (always use force components, not torque) ────
        if self._force_active:
            asset = isaac_env.scene["robot"]
            f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :3]
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
            f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :3]
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
            if "torque_range" in event_cfg.params:
                event_cfg.params["torque_range"] = (0.0, self._max_torque)
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
