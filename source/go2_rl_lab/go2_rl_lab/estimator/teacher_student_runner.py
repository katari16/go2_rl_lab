"""TeacherStudentRunner — PPO + teacher-student force estimation + compliance.

Extends CompliantOnPolicyRunner with PAINT-style teacher-student distillation.

Training phases:
    Phase 1 (reward < threshold): Walking only. Teacher trains with GT force.
    Phase 2 (reward >= threshold): Forces activate. Teacher keeps training.
    Phase 3 (teacher angular error < threshold): Student distillation begins.
        Teacher frozen, student encoder learns from teacher latent via MSE.
    Phase 4 (student angular error < threshold): Linear mapping activates.
        Policy uses student's force estimate for compliance.
"""

from __future__ import annotations

import math
import os
import statistics
import time
import torch
from collections import deque
from rsl_rl.runners import OnPolicyRunner

from go2_rl_lab.estimator.teacher_student_estimator import TeacherStudentEstimator
from go2_rl_lab.estimator.obs_history_buffer import ObsHistoryBuffer


class TeacherStudentRunner(OnPolicyRunner):

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # ── Parse config ─────────────────────────────────────────────────
        est_cfg: dict = train_cfg.get("estimator", {})
        self._temporal_steps: int = est_cfg.get("temporal_steps", 20)
        self._force_dim: int = est_cfg.get("force_dim", 3)

        # Gate thresholds
        self._force_threshold: float = train_cfg.get("force_activation_reward_threshold", 30.0)
        self._teacher_angular_threshold: float = train_cfg.get("teacher_angular_threshold", 7.0)
        self._student_angular_threshold: float = train_cfg.get("student_angular_threshold", 7.0)
        self._force_event_term: str = train_cfg.get("force_event_term_name", "persistent_xyz_force")
        self._max_force: float = train_cfg.get("max_force", 20.0)

        # Compliance
        self._compliance_alpha: float = train_cfg.get("compliance_alpha", 5.0)
        self._compliance_beta: float = train_cfg.get("compliance_beta", 50.0)

        # ── Initialize force estimate on env ─────────────────────────────
        env.unwrapped._force_estimate_xy = torch.zeros(env.num_envs, self._force_dim, device=device)

        # ── Infer obs dim ────────────────────────────────────────────────
        raw_obs = env.get_observations()
        policy_obs_dim = raw_obs["policy"].shape[-1]
        self._num_one_step_obs: int = policy_obs_dim - self._force_dim

        # ── Create teacher-student estimator ─────────────────────────────
        self.estimator = TeacherStudentEstimator(
            temporal_steps=self._temporal_steps,
            num_one_step_obs=self._num_one_step_obs,
            force_dim=self._force_dim,
            enc_hidden_dims=est_cfg.get("enc_hidden_dims", [128, 64]),
            f_head_dims=est_cfg.get("f_head_dims", [32, 16]),
            dec_hidden_dims=est_cfg.get("dec_hidden_dims", [256, 128]),
            activation=est_cfg.get("activation", "elu"),
            teacher_lr=est_cfg.get("teacher_lr", 1e-3),
            student_lr=est_cfg.get("student_lr", 1e-3),
            force_loss_weight=est_cfg.get("force_loss_weight", 1.0),
            angle_loss_weight=est_cfg.get("angle_loss_weight", 3.0),
            rec_loss_weight=est_cfg.get("rec_loss_weight", 1.0),
            kl_loss_weight=est_cfg.get("kl_loss_weight", 1.0),
            angle_min_force=est_cfg.get("angle_min_force", 1.0),
            max_grad_norm=est_cfg.get("max_grad_norm", 10.0),
        ).to(device)

        # ── History buffer (for student) ─────────────────────────────────
        self._history_buffer = ObsHistoryBuffer(
            num_envs=env.num_envs,
            temporal_steps=self._temporal_steps,
            obs_dim=self._num_one_step_obs,
            device=device,
        )

        # ── Call parent __init__ ─────────────────────────────────────────
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # ── Rollout buffers ──────────────────────────────────────────────
        num_envs = env.num_envs
        history_flat_dim = self._temporal_steps * self._num_one_step_obs

        # For student: obs history
        self._est_obs_history = torch.zeros(
            self.num_steps_per_env, num_envs, history_flat_dim, device=device
        )
        # For teacher: current raw obs
        self._est_current_obs = torch.zeros(
            self.num_steps_per_env, num_envs, self._num_one_step_obs, device=device
        )
        # Next obs (reconstruction target for teacher)
        self._est_next_raw_obs = torch.zeros(
            self.num_steps_per_env, num_envs, self._num_one_step_obs, device=device
        )

        # ── State ────────────────────────────────────────────────────────
        self._force_active: bool = False
        self._student_active: bool = False
        self._mapping_active: bool = False
        self._est_loss_buf: deque = deque(maxlen=20)
        self._last_est_stats: dict = {}

        print(
            f"[TeacherStudentRunner] Phase gates:\n"
            f"  Forces @ reward>={self._force_threshold:.0f}\n"
            f"  Student distillation @ teacher_angular_err<{self._teacher_angular_threshold:.0f}°\n"
            f"  Mapping @ student_angular_err<{self._student_angular_threshold:.0f}°\n"
            f"  Estimator: {self._temporal_steps}x{self._num_one_step_obs} → force_dim={self._force_dim}\n"
            f"  KL weight: {est_cfg.get('kl_loss_weight', 1.0)}"
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
                    raw_obs = obs["policy"][:, :self._num_one_step_obs]

                    # Update history and store for training
                    self._history_buffer.insert(raw_obs)
                    self._est_obs_history[step] = self._history_buffer.get_flattened()
                    self._est_current_obs[step] = raw_obs

                    # Inference: use student if distillation started, else teacher
                    if self._student_active:
                        force_hat, _ = self.estimator.get_latent(
                            self._history_buffer.get_flattened()
                        )
                    elif self._force_active:
                        # Use teacher during phase 2 (needs GT force)
                        gt_force = self._get_gt_force()
                        force_hat, _ = self.estimator.get_teacher_latent(raw_obs, gt_force)
                    else:
                        force_hat = torch.zeros(
                            self.env.num_envs, self._force_dim, device=self.device
                        )

                    if not self._force_active:
                        force_hat = force_hat * 0.0

                    self.env.unwrapped._force_estimate_xy = force_hat

                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
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
                self.alg.compute_returns(obs)

            # ── PPO update ───────────────────────────────────────────────
            loss_dict = self.alg.update()

            # ── Estimator update ─────────────────────────────────────────
            if self._force_active:
                est_stats = self._train_estimator()
                self._last_est_stats = est_stats
                if "force_loss" in est_stats:
                    self._est_loss_buf.append(est_stats["force_loss"])

            # ── Phase gates ──────────────────────────────────────────────
            self._check_force_gate(rewbuffer, it)
            self._check_student_gate(it)
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

    def _get_gt_force(self) -> torch.Tensor:
        """Get GT force/wrench from the sim."""
        asset = self.env.unwrapped.scene["robot"]
        if self._force_dim <= 3:
            return asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :self._force_dim]
        elif self._force_dim == 4:
            gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :3]
            gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:, 0, 2:3]
            return torch.cat([gt_f, gt_t], dim=-1)
        else:
            gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :3]
            gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:, 0, :3]
            return torch.cat([gt_f, gt_t], dim=-1)

    def _train_estimator(self) -> dict:
        """Train teacher or student depending on phase."""
        gt_force = self._get_gt_force()

        num_steps = self.num_steps_per_env
        num_envs = self.env.num_envs
        batch_size = num_steps * num_envs
        mini_batch_size = batch_size // 4

        # Flatten rollout data
        obs_hist_flat = self._est_obs_history.reshape(batch_size, -1)
        current_obs_flat = self._est_current_obs.reshape(batch_size, -1)
        next_obs_flat = self._est_next_raw_obs.reshape(batch_size, -1)
        gt_force_flat = gt_force.unsqueeze(0).expand(num_steps, -1, -1).reshape(batch_size, -1)

        indices = torch.randperm(batch_size, device=self.device)

        total_stats = {}
        for i in range(4):
            idx = indices[i * mini_batch_size: (i + 1) * mini_batch_size]

            if self._student_active:
                stats = self.estimator.update_student(
                    obs_hist_flat[idx],
                    current_obs_flat[idx],
                    gt_force_flat[idx],
                )
            else:
                stats = self.estimator.update_teacher(
                    current_obs_flat[idx],
                    gt_force_flat[idx],
                    next_obs_flat[idx],
                )

            if not total_stats:
                total_stats = {k: v for k, v in stats.items()}
            else:
                for k, v in stats.items():
                    total_stats[k] = (total_stats[k] + v) / 2

        return total_stats

    # ── Phase gates ───────────────────────────────────────────────────────

    def _check_force_gate(self, rewbuffer: deque, it: int) -> None:
        """Phase 1 → 2: Activate forces."""
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
                f"  [TeacherStudentRunner] PHASE 2: reward {mean_rew:.1f} >= {self._force_threshold:.1f}\n"
                f"  Forces activated. Teacher training begins.\n"
                f"{'=' * 80}"
            )

    def _check_student_gate(self, it: int) -> None:
        """Phase 2 → 3: Start student distillation when teacher is good."""
        if self._student_active or not self._force_active:
            return
        angle_err = self._last_est_stats.get("angle_err_median_deg", 999.0)
        if angle_err < self._teacher_angular_threshold:
            self._student_active = True
            self.estimator.set_student_phase()
            print(
                f"\n{'=' * 80}\n"
                f"  [TeacherStudentRunner] PHASE 3: Teacher angular error {angle_err:.1f}° "
                f"< {self._teacher_angular_threshold:.1f}°\n"
                f"  Student distillation begins. Teacher frozen.\n"
                f"{'=' * 80}"
            )

    def _check_mapping_gate(self, it: int) -> None:
        """Phase 3 → 4: Activate mapping when student is good."""
        if self._mapping_active or not self._student_active:
            return
        angle_err = self._last_est_stats.get("angle_err_median_deg", 999.0)
        if angle_err < self._student_angular_threshold:
            self._mapping_active = True
            isaac_env = self.env.unwrapped
            isaac_env._mapping_active = True
            isaac_env._compliance_alpha = self._compliance_alpha
            isaac_env._compliance_beta = self._compliance_beta
            print(
                f"\n{'=' * 80}\n"
                f"  [TeacherStudentRunner] PHASE 4: Student angular error {angle_err:.1f}° "
                f"< {self._student_angular_threshold:.1f}°\n"
                f"  Linear mapping activated.\n"
                f"{'=' * 80}"
            )

    # ── Logging ───────────────────────────────────────────────────────────

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)

        it = locs["it"]
        rewbuffer = locs["rewbuffer"]

        if self._mapping_active:
            phase = "PHASE 4 (mapping)"
        elif self._student_active:
            phase = "PHASE 3 (student distill)"
        elif self._force_active:
            phase = "PHASE 2 (teacher)"
        else:
            phase = "PHASE 1 (walking)"

        phase_num = 4 if self._mapping_active else (3 if self._student_active else (2 if self._force_active else 1))
        self.writer.add_scalar("TeacherStudent/phase", phase_num, it)

        est = self._last_est_stats
        if est:
            for key in ["force_loss", "angle_loss", "rec_loss", "kl_loss",
                        "mae_total", "mae_x", "mae_y", "mae_z", "mae_tau_yaw",
                        "angle_err_mean_deg", "angle_err_median_deg",
                        "gt_force_mean_mag", "pred_force_mean_mag"]:
                if key in est:
                    self.writer.add_scalar(f"Estimator/{key}", est[key], it)

        # Terminal output
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
            if self._student_active and "kl_loss" in est:
                term_str += f" kl={est['kl_loss']:.4f}"
        print(term_str)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        ckpt["teacher_student_state_dict"] = self.estimator.state_dict()
        ckpt["teacher_student_phase"] = {
            "force_active": self._force_active,
            "student_active": self._student_active,
            "mapping_active": self._mapping_active,
            "training_phase": self.estimator.training_phase,
        }
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        ckpt = torch.load(path, weights_only=False, map_location=map_location)
        if "teacher_student_state_dict" in ckpt:
            self.estimator.load_state_dict(ckpt["teacher_student_state_dict"])
            print("[TeacherStudentRunner] Estimator weights loaded.")
        state = ckpt.get("teacher_student_phase", {})
        if state.get("force_active"):
            self._force_active = True
            isaac_env = self.env.unwrapped
            event_cfg = isaac_env.event_manager.get_term_cfg(self._force_event_term)
            event_cfg.params["force_range"] = (0.0, self._max_force)
        if state.get("student_active"):
            self._student_active = True
            if state.get("training_phase") == "student":
                self.estimator.set_student_phase()
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
