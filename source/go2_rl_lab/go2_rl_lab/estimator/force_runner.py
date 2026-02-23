"""ForceOnPolicyRunner — PPO + force estimator in the mini-batch loop.

Extends rsl_rl's OnPolicyRunner with:
1. A ForceEstimator trained per PPO mini-batch (HAC-LOCO pattern).
2. An EstimatorEnvWrapper that augments policy observations with the latent.
3. Estimator-specific tensorboard logging alongside standard PPO stats.
4. Estimator weights saved/loaded with the policy checkpoint.

Key difference from EstimatorOnPolicyRunner:
    - No velocity head — force only.
    - Estimator updates happen INSIDE the PPO mini-batch loop (via ForceEstimatorPPO),
      not in a separate pass after PPO.
    - No training gate — force estimation trains from iteration 0.
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
            rec_loss_weight=est_cfg.get("rec_loss_weight", 1.0),
            max_grad_norm=est_cfg.get("max_grad_norm", 10.0),
        ).to(device)

        print(
            f"[ForceRunner] ForceEstimator: "
            f"input={self._temporal_steps}x{self._num_one_step_obs}="
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

        # ── Force activation gate ─────────────────────────────────────────
        # Forces start at 0 and activate once XY tracking reward >= threshold.
        self._force_activation_threshold: float = est_cfg.get("force_activation_reward_threshold", 0.8)
        self._force_event_term_name: str = est_cfg.get("force_event_term_name", "persistent_xy_force")
        self._max_force: float = est_cfg.get("max_force", 20.0)
        self._force_active: bool = self._force_activation_threshold <= 0.0
        self._xy_reward_buf: deque = deque(maxlen=50)
        if not self._force_active:
            print(
                f"[ForceRunner] Forces gated on XY tracking reward "
                f">= {self._force_activation_threshold:.2f}"
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

            # ── Pass estimator data to PPO, then update ───────────────────
            self.alg.set_estimator_data(self._est_obs_history, self._est_next_raw_obs)
            loss_dict = self.alg.update()

            # Track estimator losses
            if "force_loss" in loss_dict:
                self._est_force_loss_buf.append(loss_dict["force_loss"])
            if "rec_loss" in loss_dict:
                self._est_rec_loss_buf.append(loss_dict["rec_loss"])

            # ── Force activation gate ─────────────────────────────────────
            if not self._force_active:
                # Compute XY tracking reward from critic obs:
                #   critic layout: [0:2]=base_lin_vel_xy, [9:11]=vel_cmd_xy
                #   reward = exp(-||v_cmd - v_actual||² / 0.25)
                critic_obs = self.alg.storage.observations["critic"]
                vel_xy = critic_obs[:, :, 0:2]
                cmd_xy = critic_obs[:, :, 9:11]
                error_sq = ((cmd_xy - vel_xy) ** 2).sum(dim=-1)
                mean_xy_rew = torch.exp(-error_sq / 0.25).mean().item()
                self._xy_reward_buf.append(mean_xy_rew)

                if len(self._xy_reward_buf) >= 10:
                    smooth_xy = statistics.mean(self._xy_reward_buf)
                    if smooth_xy >= self._force_activation_threshold:
                        self._force_active = True
                        # Activate forces via event manager
                        underlying_env = self._wrapped_env._env.unwrapped
                        event_cfg = underlying_env.event_manager.get_term_cfg(
                            self._force_event_term_name
                        )
                        event_cfg.params["force_range"] = (0.0, self._max_force)
                        print(
                            f"\n[ForceRunner] XY tracking reward reached "
                            f"{smooth_xy:.3f} >= {self._force_activation_threshold:.2f} "
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

        # Force gate tensorboard
        self.writer.add_scalar("Estimator/force_active", float(self._force_active), it)
        if len(self._xy_reward_buf) > 0:
            self.writer.add_scalar("Estimator/xy_tracking_reward", statistics.mean(self._xy_reward_buf), it)

        # Estimator-specific tensorboard
        if "force_loss" in loss_dict:
            self.writer.add_scalar("Estimator/force_loss", loss_dict["force_loss"], it)
        if "rec_loss" in loss_dict:
            self.writer.add_scalar("Estimator/rec_loss", loss_dict["rec_loss"], it)
        if len(self._est_force_loss_buf) > 0:
            self.writer.add_scalar(
                "Estimator/force_loss_smooth", statistics.mean(self._est_force_loss_buf), it
            )
        if len(self._est_rec_loss_buf) > 0:
            self.writer.add_scalar(
                "Estimator/rec_loss_smooth", statistics.mean(self._est_rec_loss_buf), it
            )

        # Terminal — append estimator block after normal log
        pad = 35
        if "force_loss" in loss_dict:
            force_status = f"ACTIVE {self._max_force:.0f}N" if self._force_active else "waiting"
            est_str = (
                f"\n{'─' * 80}\n"
                f"{'[ForceEstimator]':>{pad}} force_loss={loss_dict['force_loss']:.5f}  "
                f"rec_loss={loss_dict['rec_loss']:.5f}  "
                f"force={force_status}"
            )
            if len(self._est_force_loss_buf) >= 5:
                est_str += f"  |  smooth_frc={statistics.mean(self._est_force_loss_buf):.5f}"
            if len(self._est_rec_loss_buf) >= 5:
                est_str += f"  smooth_rec={statistics.mean(self._est_rec_loss_buf):.5f}"
            if not self._force_active and self._xy_reward_buf:
                est_str += f"  xy_rew={statistics.mean(self._xy_reward_buf):.3f}/{self._force_activation_threshold:.2f}"
            print(est_str)

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
