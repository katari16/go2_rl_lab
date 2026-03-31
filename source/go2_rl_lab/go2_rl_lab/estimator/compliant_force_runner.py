"""CompliantForceRunner — single-policy compliant locomotion with frozen estimator.

Loads a pre-trained force estimator from a stage-1 checkpoint, freezes it,
and uses it to augment policy observations. The policy is trained end-to-end
(standard PPO, 127->12 joint actions) with a force activation gate:

Phase 1: No forces. Estimator outputs ~0. Standard locomotion training.
Phase 2: Forces activated (mean reward > threshold). Velocity tracking
         reward modulated with k*f_hat for compliant yielding behaviour.

This is simpler than HAC-LOCO's hierarchical approach — one policy, one env,
standard PPO. The compliance behaviour emerges from reward shaping.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from .estimator_env_wrapper import EstimatorEnvWrapper
from .force_estimator import ForceEstimator
from .obs_history_buffer import ObsHistoryBuffer


class CompliantForceRunner(OnPolicyRunner):
    """Trains a locomotion policy with frozen force estimator for compliance.

    Extra keys expected in train_cfg (under "estimator"):
        temporal_steps          int     History window (default: 20)
        enc_hidden_dims         list    Encoder hidden dims
        f_head_dims             list    Force-head dims
        force_dim               int     Force output dim (default: 2)
        dec_hidden_dims         list    Decoder dims
        activation              str     Activation (default: "elu")

    Extra keys expected in train_cfg (under "compliance"):
        estimator_checkpoint    str     Path to stage-1 .pt file (REQUIRED)
        compliance_k            float   Compliance gain for v* = v + k*f_hat (default: 0.003)
        ema_alpha               float   EMA smoothing for force estimate (default: 0.1)
        force_activation_reward_threshold  float  Gate threshold (default: 30.0)
        force_event_term_name   str     Event term name (default: "persistent_xy_force")
        max_force               float   Max force magnitude (default: 20.0)
    """

    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
    ) -> None:
        est_cfg: dict = train_cfg.get("estimator", {})
        comp_cfg: dict = train_cfg.get("compliance", {})

        # ── Load pre-trained estimator from checkpoint ──────────────────
        ckpt_path = comp_cfg.get("estimator_checkpoint", "")
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Estimator checkpoint not found: '{ckpt_path}'. "
                "Set compliance.estimator_checkpoint in the agent config."
            )
        print(f"[CompliantRunner] Loading estimator from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

        # Infer obs dim from env
        raw_obs = env.get_observations()
        num_one_step_obs: int = raw_obs["policy"].shape[-1]
        temporal_steps = est_cfg.get("temporal_steps", 20)

        # Reconstruct and freeze the estimator
        self.estimator = ForceEstimator(
            temporal_steps=temporal_steps,
            num_one_step_obs=num_one_step_obs,
            enc_hidden_dims=est_cfg.get("enc_hidden_dims", [128, 64]),
            f_head_dims=est_cfg.get("f_head_dims", [32, 16]),
            force_dim=est_cfg.get("force_dim", 2),
            dec_hidden_dims=est_cfg.get("dec_hidden_dims", [256, 128]),
            activation=est_cfg.get("activation", "elu"),
        ).to(device)

        if "force_estimator_state_dict" in ckpt:
            self.estimator.load_state_dict(ckpt["force_estimator_state_dict"])
            print("[CompliantRunner] Force estimator loaded from checkpoint.")
        else:
            print(
                "[CompliantRunner] WARNING: No force_estimator_state_dict in checkpoint. "
                "Estimator starts with random weights!"
            )
        self.estimator.eval()
        for p in self.estimator.parameters():
            p.requires_grad = False
        print(
            f"[CompliantRunner] Frozen estimator: "
            f"{temporal_steps}x{num_one_step_obs} -> latent_dim={self.estimator.latent_dim}  "
            f"(frozen, no grad)"
        )

        # ── Create obs history buffer ───────────────────────────────────
        history_buffer = ObsHistoryBuffer(
            num_envs=env.num_envs,
            temporal_steps=temporal_steps,
            obs_dim=num_one_step_obs,
            device=device,
        )

        # ── Wrap env with estimator ─────────────────────────────────────
        self._wrapped_env = EstimatorEnvWrapper(
            env=env,
            estimator=self.estimator,
            history_buffer=history_buffer,
            device=device,
        )
        # Enable tracking reward correction (active when compliance_k > 0)
        self._wrapped_env.tracking_correction_enabled = True
        self._wrapped_env.compliance_k = 0.0  # Start disabled
        self._wrapped_env.ema_alpha = comp_cfg.get("ema_alpha", 0.1)

        # ── Call parent __init__ with the WRAPPED env ───────────────────
        super().__init__(self._wrapped_env, train_cfg, log_dir=log_dir, device=device)

        # ── Load stage-1 policy weights ─────────────────────────────────
        # The frozen estimator was co-trained with this policy — loading it
        # ensures the policy already walks and can use the estimator latent.
        # Fine-tuning starts from a working locomotion policy, not scratch.
        if "model_state_dict" in ckpt:
            self.alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
            print("[CompliantRunner] Policy weights loaded from stage-1 checkpoint (fine-tuning).")
        if "optimizer_state_dict" in ckpt:
            self.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[CompliantRunner] Optimizer state loaded from stage-1 checkpoint.")

        # ── Force activation gate ───────────────────────────────────────
        self._force_activation_threshold: float = comp_cfg.get(
            "force_activation_reward_threshold", 30.0
        )
        self._force_event_term_name: str = comp_cfg.get(
            "force_event_term_name", "persistent_xy_force"
        )
        self._max_force: float = comp_cfg.get("max_force", 20.0)
        self._force_active: bool = self._force_activation_threshold <= 0.0

        # ── Compliance config ───────────────────────────────────────────
        self._compliance_k_target: float = comp_cfg.get("compliance_k", 0.003)
        self._estimator_checkpoint = ckpt_path

        if not self._force_active:
            print(
                f"[CompliantRunner] Phase 1: standard locomotion "
                f"(forces gated on mean reward >= {self._force_activation_threshold:.1f})"
            )
        print(
            f"[CompliantRunner] Phase 2: compliant tracking with "
            f"k={self._compliance_k_target:.4f}  max_force={self._max_force:.0f}N"
        )

    # ── Main training loop ────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Training loop with force activation gate for compliance."""
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

            # ── Rollout collection (standard PPO) ─────────────────────
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
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

            # ── PPO update (standard) ─────────────────────────────────
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it

            # ── Force activation gate ─────────────────────────────────
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
                    # Enable compliance modulation
                    self._wrapped_env.compliance_k = self._compliance_k_target
                    print(
                        f"\n[CompliantRunner] ═══ PHASE 2 ACTIVATED (iter {it}) ═══\n"
                        f"  Mean reward reached {mean_ep_rew:.1f} >= "
                        f"{self._force_activation_threshold:.1f}\n"
                        f"  Forces: (0, {self._max_force:.0f}N)\n"
                        f"  Compliance: k={self._compliance_k_target:.4f}  "
                        f"(v* = v + k*f̂, tracking reward corrected)"
                    )

            # ── Logging & saving ──────────────────────────────────────
            if self.log_dir is not None and not self.disable_logs:
                self._log_compliant(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_compliant(self, locs: dict) -> None:
        """Log standard PPO stats + compliance info."""
        it = locs["it"]
        rewbuffer = locs["rewbuffer"]

        # Standard OnPolicyRunner logging
        self.log(locs)

        # Compliance-specific scalars
        self.writer.add_scalar("Compliance/force_active", float(self._force_active), it)
        self.writer.add_scalar("Compliance/k", self._wrapped_env.compliance_k, it)
        if len(rewbuffer) > 0:
            self.writer.add_scalar(
                "Compliance/mean_ep_reward_gate", statistics.mean(rewbuffer), it
            )

        # Terminal output
        pad = 35
        if len(rewbuffer) > 0:
            mean_rew = statistics.mean(rewbuffer)
            phase = "PHASE 2 (compliant)" if self._force_active else "PHASE 1 (locomotion)"
            k_now = self._wrapped_env.compliance_k
            print(
                f"\n{'─' * 80}\n"
                f"{'[Compliance]':>{pad}} {phase}  "
                f"mean_rew={mean_rew:.1f}"
                f"/{self._force_activation_threshold:.1f}  "
                f"k={k_now:.4f}"
            )

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save policy + frozen estimator weights."""
        super().save(path, infos)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        ckpt["force_estimator_state_dict"] = self.estimator.state_dict()
        ckpt["estimator_checkpoint"] = self._estimator_checkpoint
        ckpt["compliance_config"] = {
            "k": self._compliance_k_target,
            "force_active": self._force_active,
        }
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load policy weights (estimator stays frozen from init checkpoint)."""
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        if "compliance_config" in loaded:
            cc = loaded["compliance_config"]
            print(f"[CompliantRunner] Loaded compliance config: k={cc['k']}, force_active={cc['force_active']}")
            if cc.get("force_active", False):
                self._force_active = True
                self._wrapped_env.compliance_k = self._compliance_k_target
                # Re-activate forces
                try:
                    underlying_env = self._wrapped_env._env.unwrapped
                    event_cfg = underlying_env.event_manager.get_term_cfg(
                        self._force_event_term_name
                    )
                    event_cfg.params["force_range"] = (0.0, self._max_force)
                except Exception:
                    pass
        return infos

    # ── Mode helpers ──────────────────────────────────────────────────────

    def train_mode(self) -> None:
        super().train_mode()
        self.estimator.eval()  # Always frozen

    def eval_mode(self) -> None:
        super().eval_mode()
        self.estimator.eval()
