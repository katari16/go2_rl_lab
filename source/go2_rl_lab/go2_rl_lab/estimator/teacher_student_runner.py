"""PaintRunner — PAINT-style teacher-student distillation (RA-L 2026).

Two-stage training matching PAINT's approach:

Stage 1 — Teacher (privileged):
    Train PPO policy with GT wrench in observations.
    Policy obs = [proprioceptive(57), GT_wrench(force_dim)].
    Standard CompliantOnPolicyRunner 3-phase training.

Stage 2 — Student (deployable):
    Train new PPO policy with estimated wrench in observations.
    Policy obs = [proprioceptive(57), β_est(force_dim)].
    Joint objective: J_RL - λ_KL * D_KL(π_student || π_teacher)  (PAINT Eq. 9-10)
    Intent estimator trained with supervised regression (PAINT Eq. 8).
    Teacher policy frozen, provides target action distribution.

Usage:
    1. Run Stage 1: train teacher with CompliantOnPolicyRunner (Go2-LowLevel-v0)
    2. Run Stage 2: train student with PaintRunner, passing teacher checkpoint
       --task Go2-Ablation-D1-v0 --resume --checkpoint <teacher_model.pt>
"""

from __future__ import annotations

import math
import os
import statistics
import time
import torch
from collections import deque
from torch.distributions import Normal
from rsl_rl.runners import OnPolicyRunner

from go2_rl_lab.estimator.teacher_student_estimator import IntentEstimator
from go2_rl_lab.estimator.obs_history_buffer import ObsHistoryBuffer


class PaintRunner(OnPolicyRunner):
    """PAINT-style student training with KL distillation from frozen teacher.

    Requires a pre-trained teacher checkpoint (from CompliantOnPolicyRunner).
    The teacher's actor network is loaded and frozen. The student policy
    is trained with PPO + KL regularization to the teacher + supervised
    intent estimator.
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # ── Parse config ─────────────────────────────────────────────────
        est_cfg: dict = train_cfg.get("estimator", {})
        self._temporal_steps: int = est_cfg.get("temporal_steps", 20)
        self._force_dim: int = est_cfg.get("force_dim", 3)
        self._kl_weight: float = train_cfg.get("kl_weight", 1.0)

        # Force gates (same as CompliantOnPolicyRunner)
        self._force_threshold: float = train_cfg.get("force_activation_reward_threshold", 30.0)
        self._force_event_term: str = train_cfg.get("force_event_term_name", "persistent_xyz_force")
        self._max_force: float = train_cfg.get("max_force", 20.0)

        # Compliance
        self._compliance_alpha: float = train_cfg.get("compliance_alpha", 5.0)
        self._compliance_beta: float = train_cfg.get("compliance_beta", 50.0)
        self._angular_threshold: float = train_cfg.get("estimator_angular_threshold", 7.0)

        # ── Initialize force estimate on env ─────────────────────────────
        env.unwrapped._force_estimate_xy = torch.zeros(env.num_envs, self._force_dim, device=device)

        # ── Infer obs dim ────────────────────────────────────────────────
        raw_obs = env.get_observations()
        policy_obs_dim = raw_obs["policy"].shape[-1]
        self._num_one_step_obs: int = policy_obs_dim - self._force_dim

        # ── Create intent estimator (PAINT Eq. 7-8) ─────────────────────
        self.intent_estimator = IntentEstimator(
            temporal_steps=self._temporal_steps,
            num_one_step_obs=self._num_one_step_obs,
            force_dim=self._force_dim,
            hidden_dims=est_cfg.get("hidden_dims", [128, 64, 32]),
            activation=est_cfg.get("activation", "elu"),
            learning_rate=est_cfg.get("learning_rate", 1e-3),
            max_grad_norm=est_cfg.get("max_grad_norm", 10.0),
        ).to(device)

        # ── History buffer ───────────────────────────────────────────────
        self._history_buffer = ObsHistoryBuffer(
            num_envs=env.num_envs,
            temporal_steps=self._temporal_steps,
            obs_dim=self._num_one_step_obs,
            device=device,
        )

        # ── Call parent __init__ (creates PPO alg + student policy) ──────
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # ── Rollout buffers for intent estimator training ────────────────
        num_envs = env.num_envs
        history_flat_dim = self._temporal_steps * self._num_one_step_obs
        self._est_obs_history = torch.zeros(
            self.num_steps_per_env, num_envs, history_flat_dim, device=device
        )

        # ── Teacher policy (loaded from checkpoint, frozen) ──────────────
        # Will be populated by load() when checkpoint is provided
        self._teacher_actor = None
        self._teacher_obs_normalizer = None

        # ── State ────────────────────────────────────────────────────────
        self._force_active: bool = False
        self._mapping_active: bool = False
        self._est_loss_buf: deque = deque(maxlen=20)
        self._last_est_stats: dict = {}
        self._last_kl: float = 0.0

        print(
            f"[PaintRunner] PAINT-style student training\n"
            f"  KL weight: {self._kl_weight}\n"
            f"  Force gate: reward >= {self._force_threshold}\n"
            f"  Mapping gate: angle_err < {self._angular_threshold}°\n"
            f"  Estimator: {self._temporal_steps}x{self._num_one_step_obs} → force_dim={self._force_dim}"
        )

    # ── Main training loop ────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._prepare_logging_writer()

        if self._teacher_actor is None:
            raise RuntimeError(
                "[PaintRunner] No teacher loaded! Provide a teacher checkpoint via --resume. "
                "The teacher should be trained with CompliantOnPolicyRunner first."
            )

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

                    # Update history buffer
                    self._history_buffer.insert(raw_obs)
                    self._est_obs_history[step] = self._history_buffer.get_flattened()

                    # Intent estimator inference → β_est
                    if self._force_active:
                        beta_est = self.intent_estimator.predict(
                            self._history_buffer.get_flattened()
                        )
                    else:
                        beta_est = torch.zeros(
                            self.env.num_envs, self._force_dim, device=self.device
                        )

                    # Set on env for ForceEstimateObsTerm and compliance reward
                    self.env.unwrapped._force_estimate_xy = beta_est

                    # PPO act (student policy sees [proprioceptive, β_est])
                    actions = self.alg.act(obs)

                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # Reset history for terminated envs
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

            # ── PPO update + KL regularization (PAINT Eq. 9-10) ─────────
            loss_dict = self._update_with_kl()

            # ── Intent estimator update (PAINT Eq. 8) ────────────────────
            if self._force_active:
                est_stats = self._train_intent_estimator()
                self._last_est_stats = est_stats
                if "est_loss" in est_stats:
                    self._est_loss_buf.append(est_stats["est_loss"])

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

    # ── PPO + KL update (PAINT Eq. 9-10) ─────────────────────────────────

    def _update_with_kl(self) -> dict[str, float]:
        """Standard PPO update with added KL divergence loss to teacher.

        PAINT Eq. 9: max J_RL - λ_KL * D_KL(π_student || π_teacher)
        """
        ppo = self.alg
        mean_kl_loss = 0.0
        num_batches = 0

        # Get mini batch generator
        if ppo.policy.is_recurrent:
            generator = ppo.storage.recurrent_mini_batch_generator(
                ppo.num_mini_batches, ppo.num_learning_epochs
            )
        else:
            generator = ppo.storage.mini_batch_generator(
                ppo.num_mini_batches, ppo.num_learning_epochs
            )

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

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
            num_batches += 1

            # ── Standard PPO forward ────────────────────────────────────
            ppo.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = ppo.policy.get_actions_log_prob(actions_batch)
            value_batch = ppo.policy.evaluate(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
            )
            mu_batch = ppo.policy.action_mean
            sigma_batch = ppo.policy.action_std
            entropy_batch = ppo.policy.entropy

            # Adaptive LR based on KL
            if ppo.desired_kl is not None and ppo.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2))
                        / (2.0 * sigma_batch.pow(2))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = kl.mean()
                    if kl_mean > ppo.desired_kl * 2.0:
                        ppo.learning_rate = max(1e-5, ppo.learning_rate / 1.5)
                    elif kl_mean < ppo.desired_kl / 2.0 and kl_mean > 0.0:
                        ppo.learning_rate = min(1e-2, ppo.learning_rate * 1.5)
                    for pg in ppo.optimizer.param_groups:
                        pg["lr"] = ppo.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze())
            surr = -advantages_batch.squeeze() * ratio
            surr_clipped = -advantages_batch.squeeze() * torch.clamp(
                ratio, 1.0 - ppo.clip_param, 1.0 + ppo.clip_param
            )
            surrogate_loss = torch.max(surr, surr_clipped).mean()

            # Value loss
            if ppo.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -ppo.clip_param, ppo.clip_param
                )
                vl = (value_batch - returns_batch).pow(2)
                vl_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(vl, vl_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ── KL divergence to teacher (PAINT Eq. 10) ─────────────────
            kl_to_teacher = self._compute_kl_to_teacher(obs_batch, mu_batch, sigma_batch)

            # ── Total loss ──────────────────────────────────────────────
            loss = (
                surrogate_loss
                + ppo.value_loss_coef * value_loss
                - ppo.entropy_coef * entropy_batch.mean()
                + self._kl_weight * kl_to_teacher
            )

            ppo.optimizer.zero_grad()
            loss.backward()
            if ppo.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(ppo.policy.parameters(), ppo.max_grad_norm)
            ppo.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl_loss += kl_to_teacher.item()

        n = max(num_batches, 1)
        self._last_kl = mean_kl_loss / n
        ppo.storage.clear()

        return {
            "value": mean_value_loss / n,
            "surrogate": mean_surrogate_loss / n,
            "entropy": mean_entropy / n,
            "kl_to_teacher": self._last_kl,
        }

    def _compute_kl_to_teacher(
        self,
        obs_batch,
        student_mu: torch.Tensor,
        student_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """KL(π_student || π_teacher) for the given obs batch.

        Teacher sees [proprioceptive, GT_wrench] instead of [proprioceptive, β_est].
        We replace the last force_dim dims of obs with GT wrench for teacher input.
        """
        with torch.no_grad():
            # Build teacher obs: replace estimated wrench with GT wrench
            teacher_obs = obs_batch.clone()
            gt_wrench = self._get_gt_force()

            # obs_batch["policy"] shape: [batch, policy_obs_dim]
            # The last force_dim dims are the force estimate — replace with GT
            policy_obs = teacher_obs["policy"].clone()
            batch_size = policy_obs.shape[0]
            num_envs = self.env.num_envs

            # GT wrench needs to be expanded to match batch size
            # (batch may be a subset of envs × steps)
            gt_expanded = gt_wrench.unsqueeze(0).expand(
                batch_size // num_envs + 1, -1, -1
            ).reshape(-1, self._force_dim)[:batch_size]

            policy_obs[:, -self._force_dim:] = gt_expanded
            teacher_obs["policy"] = policy_obs

            # Teacher forward pass
            teacher_mu = self._teacher_actor(
                self._teacher_obs_normalizer(policy_obs)
            )
            # Use same std as student (teacher std isn't stored separately)
            teacher_sigma = student_sigma.detach()

        # KL(student || teacher) for diagonal Gaussian
        kl = torch.sum(
            torch.log(teacher_sigma / student_sigma + 1e-5)
            + (student_sigma.pow(2) + (student_mu - teacher_mu).pow(2))
            / (2.0 * teacher_sigma.pow(2))
            - 0.5,
            dim=-1,
        )
        return kl.mean()

    # ── Intent estimator training (PAINT Eq. 8) ──────────────────────────

    def _get_gt_force(self) -> torch.Tensor:
        """Get GT force/wrench from sim."""
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

    def _train_intent_estimator(self) -> dict:
        """Train intent estimator on rollout data (supervised regression)."""
        gt_wrench = self._get_gt_force()

        num_steps = self.num_steps_per_env
        num_envs = self.env.num_envs
        batch_size = num_steps * num_envs

        obs_hist_flat = self._est_obs_history.reshape(batch_size, -1)
        gt_flat = gt_wrench.unsqueeze(0).expand(num_steps, -1, -1).reshape(batch_size, -1)

        indices = torch.randperm(batch_size, device=self.device)
        mini_batch_size = batch_size // 4

        total_stats = {}
        for i in range(4):
            idx = indices[i * mini_batch_size: (i + 1) * mini_batch_size]
            stats = self.intent_estimator.update(obs_hist_flat[idx], gt_flat[idx])
            if not total_stats:
                total_stats = {k: v for k, v in stats.items()}
            else:
                for k, v in stats.items():
                    total_stats[k] = (total_stats[k] + v) / 2

        return total_stats

    # ── Phase gates ───────────────────────────────────────────────────────

    def _check_force_gate(self, rewbuffer: deque, it: int) -> None:
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
                f"  [PaintRunner] PHASE 2: reward {mean_rew:.1f} >= {self._force_threshold:.1f}\n"
                f"  Forces activated. Intent estimator + KL training begins.\n"
                f"{'=' * 80}"
            )

    def _check_mapping_gate(self, it: int) -> None:
        if self._mapping_active or not self._force_active:
            return
        angle_err = self._last_est_stats.get("angle_err_mean_deg", 999.0)
        if angle_err < self._angular_threshold:
            self._mapping_active = True
            isaac_env = self.env.unwrapped
            isaac_env._mapping_active = True
            isaac_env._compliance_alpha = self._compliance_alpha
            isaac_env._compliance_beta = self._compliance_beta
            print(
                f"\n{'=' * 80}\n"
                f"  [PaintRunner] PHASE 3: Estimator angle error {angle_err:.1f}° "
                f"< {self._angular_threshold:.1f}°\n"
                f"  Linear mapping activated.\n"
                f"{'=' * 80}"
            )

    # ── Logging ───────────────────────────────────────────────────────────

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)

        it = locs["it"]
        rewbuffer = locs["rewbuffer"]

        if self._mapping_active:
            phase = "PHASE 3 (mapping)"
        elif self._force_active:
            phase = "PHASE 2 (estimator+KL)"
        else:
            phase = "PHASE 1 (walking)"

        phase_num = 3 if self._mapping_active else (2 if self._force_active else 1)
        self.writer.add_scalar("Paint/phase", phase_num, it)
        self.writer.add_scalar("Paint/kl_to_teacher", self._last_kl, it)

        est = self._last_est_stats
        if est:
            for key in ["est_loss", "mae_total", "mae_x", "mae_y", "mae_z",
                        "mae_tau_yaw", "angle_err_mean_deg"]:
                if key in est:
                    self.writer.add_scalar(f"Estimator/{key}", est[key], it)

        term_str = f"  [{phase}]"
        if not self._force_active and len(rewbuffer) > 0:
            mean_rew = statistics.mean(rewbuffer)
            pct = min(100.0, mean_rew / self._force_threshold * 100.0)
            term_str += f"  reward={mean_rew:.1f}/{self._force_threshold:.1f} ({pct:.0f}%)"
        if self._force_active and est:
            mae = est.get("mae_total", 0)
            eloss = est.get("est_loss", 0)
            term_str += f"  est: loss={eloss:.4f} mae={mae:.3f} kl={self._last_kl:.4f}"
        print(term_str)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        ckpt["intent_estimator_state_dict"] = self.intent_estimator.state_dict()
        ckpt["intent_estimator_optimizer_state_dict"] = self.intent_estimator.optimizer.state_dict()
        ckpt["paint_state"] = {
            "force_active": self._force_active,
            "mapping_active": self._mapping_active,
        }
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load checkpoint. Extracts teacher actor from the checkpoint's PPO policy."""
        ckpt = torch.load(path, weights_only=False, map_location=map_location or self.device)

        # ── Load teacher from checkpoint ─────────────────────────────────
        # The checkpoint contains an ActorCritic state_dict with "actor.*" keys
        if "model_state_dict" in ckpt:
            model_sd = ckpt["model_state_dict"]

            # Extract teacher actor weights
            teacher_actor_sd = {}
            for k, v in model_sd.items():
                if k.startswith("actor."):
                    teacher_actor_sd[k.replace("actor.", "")] = v

            if teacher_actor_sd:
                # Build teacher actor MLP matching the student's architecture
                from rsl_rl.networks import MLP
                num_obs = list(teacher_actor_sd.values())[0].shape[1]  # input dim from first weight
                num_actions = list(teacher_actor_sd.values())[-1].shape[0]  # output dim from last weight

                # Infer hidden dims from weight shapes
                hidden_dims = []
                weight_keys = sorted([k for k in teacher_actor_sd if "weight" in k and "." in k])
                for k in weight_keys[:-1]:  # exclude output layer
                    hidden_dims.append(teacher_actor_sd[k].shape[0])

                self._teacher_actor = MLP(num_obs, num_actions, hidden_dims, "elu").to(self.device)
                self._teacher_actor.load_state_dict(teacher_actor_sd)
                self._teacher_actor.eval()
                for p in self._teacher_actor.parameters():
                    p.requires_grad = False

                # Teacher obs normalizer
                if "actor_obs_normalizer.running_mean" in model_sd:
                    from rsl_rl.networks import EmpiricalNormalization
                    self._teacher_obs_normalizer = EmpiricalNormalization(num_obs).to(self.device)
                    teacher_norm_sd = {}
                    for k, v in model_sd.items():
                        if k.startswith("actor_obs_normalizer."):
                            teacher_norm_sd[k.replace("actor_obs_normalizer.", "")] = v
                    self._teacher_obs_normalizer.load_state_dict(teacher_norm_sd)
                    self._teacher_obs_normalizer.eval()
                else:
                    self._teacher_obs_normalizer = torch.nn.Identity()

                print(f"[PaintRunner] Teacher actor loaded: {num_obs} → {hidden_dims} → {num_actions}")
            else:
                print("[PaintRunner] WARNING: No actor weights found in checkpoint")

        # ── Load intent estimator if present (resume student training) ───
        if "intent_estimator_state_dict" in ckpt:
            self.intent_estimator.load_state_dict(ckpt["intent_estimator_state_dict"])
            if load_optimizer and "intent_estimator_optimizer_state_dict" in ckpt:
                self.intent_estimator.optimizer.load_state_dict(
                    ckpt["intent_estimator_optimizer_state_dict"]
                )
            print("[PaintRunner] Intent estimator loaded from checkpoint.")

        # ── Load student PPO policy (don't overwrite with teacher weights)
        # Only load PPO state if this is a resume (has intent_estimator = student checkpoint)
        paint_state = ckpt.get("paint_state", {})
        if "intent_estimator_state_dict" in ckpt:
            # This is a student checkpoint — load PPO normally
            infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        else:
            # This is a teacher checkpoint — DON'T load PPO weights into student
            # Student starts fresh, only teacher actor is extracted above
            infos = {}
            print("[PaintRunner] Teacher checkpoint detected — student policy starts fresh.")

        # ── Restore phase state ──────────────────────────────────────────
        if paint_state.get("force_active"):
            self._force_active = True
            isaac_env = self.env.unwrapped
            event_cfg = isaac_env.event_manager.get_term_cfg(self._force_event_term)
            event_cfg.params["force_range"] = (0.0, self._max_force)
        if paint_state.get("mapping_active"):
            self._mapping_active = True
            isaac_env = self.env.unwrapped
            isaac_env._mapping_active = True
            isaac_env._compliance_alpha = self._compliance_alpha
            isaac_env._compliance_beta = self._compliance_beta

        return infos

    # ── Mode helpers ──────────────────────────────────────────────────────

    def train_mode(self) -> None:
        super().train_mode()
        self.intent_estimator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.intent_estimator.eval()
