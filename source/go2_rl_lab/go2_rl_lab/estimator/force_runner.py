"""ForceOnPolicyRunner — PPO + force estimator in the mini-batch loop.

Extends rsl_rl's OnPolicyRunner with:
1. A ForceEstimator trained per PPO mini-batch (HAC-LOCO pattern).
2. An EstimatorEnvWrapper that augments policy observations with the latent.
3. Estimator-specific tensorboard logging alongside standard PPO stats.
4. Estimator weights saved/loaded with the policy checkpoint.

Key behaviours:
    - Force estimation only trains when forces are active (not on zero-force data).
    - Estimator has its own fixed LR, decoupled from PPO's adaptive schedule.
"""

from __future__ import annotations

import os
import statistics
import time
import warnings
from collections import deque

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups

from .estimator_env_wrapper import EstimatorEnvWrapper
from .force_estimator import ForceEstimator
from .force_ppo import ForceEstimatorPPO
from .obs_history_buffer import ObsHistoryBuffer


class ForceOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with force estimator trained per PPO mini-batch.

    Extra keys expected in train_cfg (under "estimator"):
        temporal_steps          int     History window length
        enc_hidden_dims         list    Encoder hidden dims
        f_head_dims             list    Force-head dims
        force_dim               int     Force output dim (default: 2 for XY)
        dec_hidden_dims         list    Decoder dims
        activation              str     Activation (default: "elu")
        learning_rate           float   Estimator LR (default: 1e-3)
        force_loss_weight       float   Weight for force loss
        rec_loss_weight         float   Weight for reconstruction loss
        max_grad_norm           float   Gradient clip
        gt_force_obs_start_idx  int     Slice start in critic obs for gt_force
        force_activation_reward_threshold  float  XY tracking reward gate (default: 0.8)
        force_event_term_name   str     Event term name for force (default: "persistent_xy_force")
        max_force               float   Force magnitude once activated (default: 20.0)
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # ── Parse estimator config ────────────────────────────────────────
        est_cfg: dict = train_cfg.get("estimator", {})
        self._temporal_steps: int = est_cfg.get("temporal_steps", 20)
        self._gt_force_start: int = est_cfg.get("gt_force_obs_start_idx", -1)
        self._force_dim: int = est_cfg.get("force_dim", 2)

        # Infer single-step obs dim from env (before augmentation)
        raw_obs = env.get_observations()
        self._num_one_step_obs: int = raw_obs["policy"].shape[-1]

        # ── Create force estimator network ──────────────────────────────
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

        print(
            f"[ForceRunner] ForceEstimator: "
            f"input={self._temporal_steps}x{self._num_one_step_obs}="
            f"{self._temporal_steps * self._num_one_step_obs}  "
            f"latent_dim={self.estimator.latent_dim}  "
            f"policy_input={self._num_one_step_obs + self.estimator.latent_dim}  "
            f"estimator_lr={est_cfg.get('learning_rate', 1e-3)}"
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
        super().__init__(self._wrapped_env, train_cfg, log_dir=log_dir, device=device)

        # ── Allocate estimator rollout buffers ────────────────────────────
        num_envs = env.num_envs
        history_flat_dim = self._temporal_steps * self._num_one_step_obs
        self._est_obs_history = torch.zeros(
            self.num_steps_per_env, num_envs, history_flat_dim, device=device
        )
        self._est_next_raw_obs = torch.zeros(
            self.num_steps_per_env, num_envs, self._num_one_step_obs, device=device
        )

        # Running stats for logging
        self._est_force_loss_buf: deque = deque(maxlen=20)
        self._est_rec_loss_buf: deque = deque(maxlen=20)

        # Observation group names for per-category logging.
        # Maps (start_idx, end_idx) → category name within the single-step obs.
        self._obs_categories: list[tuple[str, int, int]] = [
            ("base_ang_vel", 0, 3),
            ("proj_gravity", 3, 6),
            ("vel_commands", 6, 9),
            ("joint_pos", 9, 21),
            ("joint_vel", 21, 33),
            ("last_action", 33, 45),
            ("applied_torque", 45, 57),
            ("foot_forces", 57, 61),
        ]

        # ── Force activation gate ─────────────────────────────────────────
        # Forces start at 0 and activate once mean episode reward >= threshold.
        self._force_activation_threshold: float = est_cfg.get("force_activation_reward_threshold", 30.0)
        self._force_event_term_name: str = est_cfg.get("force_event_term_name", "persistent_xy_force")
        self._max_force: float = est_cfg.get("max_force", 20.0)
        self._force_active: bool = self._force_activation_threshold <= 0.0
        if not self._force_active:
            print(
                f"[ForceRunner] Forces gated on mean episode reward "
                f">= {self._force_activation_threshold:.1f}  "
                f"(estimator training SKIPPED until forces activate)"
            )

    # ── Override algorithm construction ────────────────────────────────────

    def _construct_algorithm(self, obs: TensorDict) -> ForceEstimatorPPO:
        """Construct ForceEstimatorPPO instead of standard PPO."""
        # Resolve RND config (pass through even if unused)
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        # Resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the actor-critic policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Pop class_name from alg_cfg before passing to constructor
        self.alg_cfg.pop("class_name", None)

        # Initialize ForceEstimatorPPO (instead of standard PPO)
        alg = ForceEstimatorPPO(
            force_estimator=self.estimator,
            gt_force_start=self._gt_force_start,
            force_dim=self._force_dim,
            policy=actor_critic,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **self.alg_cfg,
        )

        # Initialize the rollout storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    # ── Main training loop ────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Full training loop with estimator updates inside PPO mini-batches."""
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

            # ── Pass estimator data + force state to PPO, then update ────
            self.alg.set_estimator_data(self._est_obs_history, self._est_next_raw_obs)
            self.alg.set_force_active(self._force_active)
            loss_dict = self.alg.update()

            # Track estimator losses (only present when force is active)
            if "force_loss" in loss_dict:
                self._est_force_loss_buf.append(loss_dict["force_loss"])
            if "rec_loss" in loss_dict:
                self._est_rec_loss_buf.append(loss_dict["rec_loss"])

            # ── Force activation gate ─────────────────────────────────────
            if not self._force_active and len(rewbuffer) >= 10:
                mean_ep_rew = statistics.mean(rewbuffer)
                if mean_ep_rew >= self._force_activation_threshold:
                    self._force_active = True
                    # Activate forces via event manager
                    underlying_env = self._wrapped_env._env.unwrapped
                    event_cfg = underlying_env.event_manager.get_term_cfg(
                        self._force_event_term_name
                    )
                    event_cfg.params["force_range"] = (0.0, self._max_force)
                    print(
                        f"\n[ForceRunner] Mean episode reward reached "
                        f"{mean_ep_rew:.1f} >= {self._force_activation_threshold:.1f} "
                        f"— activating forces at {self._max_force:.0f}N (iter {it})"
                    )

            learn_time = time.time() - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self._log_with_estimator(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_with_estimator(self, locs: dict) -> None:
        """Log PPO stats + estimator stats to tensorboard and terminal."""
        loss_dict = locs["loss_dict"]
        it = locs["it"]

        # Standard OnPolicyRunner log (handles reward, losses, FPS, ...)
        self.log(locs)

        # ── Force gate ───────────────────────────────────────────────────
        self.writer.add_scalar("Estimator/force_active", float(self._force_active), it)
        rewbuffer = locs["rewbuffer"]
        if len(rewbuffer) > 0:
            self.writer.add_scalar("Estimator/mean_ep_reward_gate", statistics.mean(rewbuffer), it)

        # ── Learning rates (both) ────────────────────────────────────────
        if "ppo_lr" in loss_dict:
            self.writer.add_scalar("LR/ppo", loss_dict["ppo_lr"], it)
        if "estimator_lr" in loss_dict:
            self.writer.add_scalar("LR/estimator", loss_dict["estimator_lr"], it)

        # ── Estimator diagnostics (only when force active) ───────────────
        if "force_loss" in loss_dict:
            # Core losses
            self.writer.add_scalar("Estimator/force_loss", loss_dict["force_loss"], it)
            self.writer.add_scalar("Estimator/rec_loss", loss_dict["rec_loss"], it)
            if len(self._est_force_loss_buf) > 0:
                self.writer.add_scalar(
                    "Estimator/force_loss_smooth", statistics.mean(self._est_force_loss_buf), it
                )
            if len(self._est_rec_loss_buf) > 0:
                self.writer.add_scalar(
                    "Estimator/rec_loss_smooth", statistics.mean(self._est_rec_loss_buf), it
                )

            # GT force stats
            if "gt_force_mean_mag" in loss_dict:
                self.writer.add_scalar("Estimator/gt_force_mean_mag", loss_dict["gt_force_mean_mag"], it)
                self.writer.add_scalar("Estimator/gt_force_max_mag", loss_dict["gt_force_max_mag"], it)
                self.writer.add_scalar("Estimator/gt_force_std_mag", loss_dict["gt_force_std_mag"], it)

            # Predicted force stats
            if "pred_force_mean_mag" in loss_dict:
                self.writer.add_scalar("Estimator/pred_force_mean_mag", loss_dict["pred_force_mean_mag"], it)

            # Per-component MAE
            if "mae_x" in loss_dict:
                self.writer.add_scalar("Estimator/mae_x", loss_dict["mae_x"], it)
                self.writer.add_scalar("Estimator/mae_y", loss_dict["mae_y"], it)
                self.writer.add_scalar("Estimator/mae_total", loss_dict["mae_total"], it)

            # Angular loss and error
            if "angle_loss" in loss_dict:
                self.writer.add_scalar("Estimator/angle_loss", loss_dict["angle_loss"], it)
            if "angle_err_mean_deg" in loss_dict:
                self.writer.add_scalar("Estimator/angle_err_mean_deg", loss_dict["angle_err_mean_deg"], it)
                self.writer.add_scalar("Estimator/angle_err_median_deg", loss_dict["angle_err_median_deg"], it)

            # Gradient norms
            if "grad_norm_encoder" in loss_dict:
                self.writer.add_scalar("Estimator/grad_norm_encoder", loss_dict["grad_norm_encoder"], it)
                self.writer.add_scalar("Estimator/grad_norm_f_head", loss_dict["grad_norm_f_head"], it)
                self.writer.add_scalar("Estimator/grad_norm_decoder", loss_dict["grad_norm_decoder"], it)

        # ── Observation statistics + weight importance (every 50 iters) ──
        if it % 50 == 0:
            self._log_obs_stats(it)
            self._log_weight_importance(it)

        # ── Terminal output ──────────────────────────────────────────────
        pad = 35
        if "force_loss" in loss_dict:
            force_status = f"ACTIVE {self._max_force:.0f}N" if self._force_active else "waiting"
            est_str = (
                f"\n{'─' * 80}\n"
                f"{'[ForceEstimator]':>{pad}} force_loss={loss_dict['force_loss']:.5f}  "
                f"rec_loss={loss_dict['rec_loss']:.5f}  "
                f"force={force_status}"
            )
            if "mae_total" in loss_dict:
                est_str += f"  mae={loss_dict['mae_total']:.4f}"
            if "gt_force_mean_mag" in loss_dict:
                est_str += f"  gt_mag={loss_dict['gt_force_mean_mag']:.2f}"
            if "pred_force_mean_mag" in loss_dict:
                est_str += f"  pred_mag={loss_dict['pred_force_mean_mag']:.2f}"
            if "angle_err_mean_deg" in loss_dict:
                est_str += f"  ang_err={loss_dict['angle_err_mean_deg']:.1f}deg"
            if "estimator_lr" in loss_dict:
                est_str += f"  lr={loss_dict['estimator_lr']:.1e}"
            if len(self._est_force_loss_buf) >= 5:
                est_str += f"\n{'':>{pad}} smooth_frc={statistics.mean(self._est_force_loss_buf):.5f}"
            if len(self._est_rec_loss_buf) >= 5:
                est_str += f"  smooth_rec={statistics.mean(self._est_rec_loss_buf):.5f}"
            print(est_str)
        elif not self._force_active and len(rewbuffer) > 0:
            mean_rew = statistics.mean(rewbuffer)
            wait_str = (
                f"\n{'─' * 80}\n"
                f"{'[ForceEstimator]':>{pad}} WAITING  "
                f"mean_ep_rew={mean_rew:.1f}"
                f"/{self._force_activation_threshold:.1f}  "
                f"(estimator not training)"
            )
            print(wait_str)

    # ── Observation & weight diagnostics ─────────────────────────────────

    @torch.no_grad()
    def _log_obs_stats(self, it: int) -> None:
        """Log per-category mean and std of the raw policy observations."""
        # Use the last collected rollout step's raw obs from the history buffer.
        raw_obs = self._wrapped_env.get_last_raw_policy_obs()
        if raw_obs is None:
            return
        # raw_obs: [num_envs, obs_dim]
        for name, s, e in self._obs_categories:
            chunk = raw_obs[:, s:e]
            self.writer.add_scalar(f"ObsStats/{name}_mean", chunk.mean().item(), it)
            self.writer.add_scalar(f"ObsStats/{name}_std", chunk.std().item(), it)
            self.writer.add_scalar(f"ObsStats/{name}_absmax", chunk.abs().max().item(), it)

    @torch.no_grad()
    def _log_weight_importance(self, it: int) -> None:
        """Log encoder first-layer weight importance per observation category."""
        enc_w = None
        for name, param in self.estimator.named_parameters():
            if name == "encoder.0.weight":
                enc_w = param
                break
        if enc_w is None:
            return
        # enc_w: [hidden, temporal_steps * obs_dim]
        obs_dim = self._num_one_step_obs
        temporal = self._temporal_steps
        if enc_w.shape[1] != temporal * obs_dim:
            return
        # Reshape to [hidden, temporal, obs_dim], average abs weight over hidden & temporal
        w = enc_w.abs().reshape(enc_w.shape[0], temporal, obs_dim).mean(dim=(0, 1))
        for name, s, e in self._obs_categories:
            importance = w[s:e].mean().item()
            self.writer.add_scalar(f"WeightImportance/{name}", importance, it)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save policy + estimator weights to the same checkpoint file."""
        super().save(path, infos)
        # Append estimator state to the same file
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        ckpt["force_estimator_state_dict"] = self.estimator.state_dict()
        ckpt["force_estimator_optimizer_state_dict"] = self.estimator.optimizer.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load policy + estimator weights from checkpoint."""
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        ckpt = torch.load(path, weights_only=False, map_location=map_location)
        if "force_estimator_state_dict" in ckpt:
            self.estimator.load_state_dict(ckpt["force_estimator_state_dict"])
            if load_optimizer and "force_estimator_optimizer_state_dict" in ckpt:
                self.estimator.optimizer.load_state_dict(ckpt["force_estimator_optimizer_state_dict"])
            print("[ForceRunner] Force estimator weights loaded from checkpoint.")
        else:
            print("[ForceRunner] No force estimator weights found in checkpoint — starting fresh.")
        return infos

    # ── Mode helpers ──────────────────────────────────────────────────────

    def train_mode(self) -> None:
        super().train_mode()
        self.estimator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.estimator.eval()
