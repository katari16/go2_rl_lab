"""ComplianceOnPolicyRunner — HAC-LOCO stage 2 compliance training.

Loads a stage-1 checkpoint, FREEZES the low-level policy, keeps the
force estimator TRAINABLE, and trains a lightweight high-level compliance
policy that outputs residual velocity commands [Dvx, Dvy, Dwz].

Key differences from CompliantOnPolicyRunner (stage 1):
- Low-level policy is FROZEN (verified every N iterations)
- Force estimator is TRAINABLE (loaded from stage-1, keeps learning)
- High-level policy: [256, 128] -> 3 actions
- Compliance reward from HAC-LOCO paper (eq. 7)
- Forces always active at max (no curriculum gating)
"""

from __future__ import annotations

import hashlib
import os
import time
import statistics
import torch
from collections import deque
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic

from go2_rl_lab.estimator.force_estimator import ForceEstimator
from go2_rl_lab.estimator.obs_history_buffer import ObsHistoryBuffer
from go2_rl_lab.estimator.compliance_env_wrapper import ComplianceEnvWrapper


def _param_checksum(module: torch.nn.Module) -> str:
    """Compute MD5 checksum of all parameters for freeze verification."""
    h = hashlib.md5()
    for p in module.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()


class ComplianceOnPolicyRunner(OnPolicyRunner):
    """Stage 2 runner: trains high-level compliance policy with frozen low-level.

    Extra keys in train_cfg:
        compliance.stage1_checkpoint    str    Path to stage-1 .pt file (REQUIRED)
        compliance.alpha                float  Force threshold (N) (default: 10.0)
        compliance.beta                 float  Virtual impedance (default: 10.0)
        compliance.compliance_reward_weight float  Weight for r_comp (default: 0.5)
        compliance.max_force            float  Max force during training (default: 20.0)

        estimator.*                     dict   Force estimator architecture (must match stage-1)
    """

    def __init__(
        self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
    ) -> None:
        comp_cfg: dict = train_cfg.get("compliance", {})
        est_cfg: dict = train_cfg.get("estimator", {})

        stage1_path = comp_cfg.get("stage1_checkpoint", "")
        if not stage1_path or not os.path.isfile(stage1_path):
            raise FileNotFoundError(
                f"Stage-1 checkpoint not found: '{stage1_path}'. "
                "Set compliance.stage1_checkpoint in the agent config."
            )

        # ── Parse estimator config ────────────────────────────────────────
        temporal_steps = est_cfg.get("temporal_steps", 20)
        force_dim = est_cfg.get("force_dim", 3)
        enc_latent_dim = est_cfg.get("enc_hidden_dims", [128, 64])[-1]
        latent_dim = force_dim + enc_latent_dim  # 3 + 64 = 67

        # ── Initialize force estimate on env BEFORE get_observations ──────
        # ForceEstimateObsTerm in dummy mode reads env._force_estimate_xy
        # It must be the right size (3 for XYZ) before obs manager runs
        underlying = env.unwrapped if hasattr(env, "unwrapped") else env
        underlying._force_estimate_xy = torch.zeros(env.num_envs, force_dim, device=device)

        # ── Infer raw obs dim from env ────────────────────────────────────
        raw_obs = env.get_observations()
        policy_obs_dim = raw_obs["policy"].shape[-1]  # 60
        raw_obs_dim = policy_obs_dim - force_dim      # 57

        print(f"[ComplianceRunner] Env policy obs: {policy_obs_dim}, raw: {raw_obs_dim}, force_dim: {force_dim}")

        # ── Load stage-1 checkpoint ───────────────────────────────────────
        print(f"[ComplianceRunner] Loading stage-1 checkpoint: {stage1_path}")
        ckpt = torch.load(stage1_path, weights_only=False, map_location=device)

        # ── Reconstruct and FREEZE low-level policy ──────────────────────
        num_augmented_obs = policy_obs_dim  # 60 (low-level sees raw + force_est)
        num_critic_obs = 70  # doesn't matter for frozen inference

        mock_obs = TensorDict(
            {
                "policy": torch.zeros(1, num_augmented_obs, device=device),
                "critic": torch.zeros(1, num_critic_obs, device=device),
            },
            batch_size=[1],
            device=device,
        )
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        frozen_policy = ActorCritic(
            mock_obs, obs_groups, num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ).to(device)
        frozen_policy.load_state_dict(ckpt["model_state_dict"])
        frozen_policy.eval()
        for p in frozen_policy.parameters():
            p.requires_grad = False

        self._frozen_policy_checksum = _param_checksum(frozen_policy)
        print(
            f"[ComplianceRunner] Frozen low-level policy: "
            f"{num_augmented_obs} -> [512,256,128] -> 12  (FROZEN, checksum={self._frozen_policy_checksum[:8]})"
        )

        # ── Reconstruct TRAINABLE force estimator ────────────────────────
        estimator = ForceEstimator(
            temporal_steps=temporal_steps,
            num_one_step_obs=raw_obs_dim,
            enc_hidden_dims=est_cfg.get("enc_hidden_dims", [128, 64]),
            f_head_dims=est_cfg.get("f_head_dims", [32, 16]),
            force_dim=force_dim,
            dec_hidden_dims=est_cfg.get("dec_hidden_dims", [256, 128]),
            activation=est_cfg.get("activation", "elu"),
            learning_rate=est_cfg.get("learning_rate", 1e-3),
            force_loss_weight=est_cfg.get("force_loss_weight", 1.0),
            angle_loss_weight=est_cfg.get("angle_loss_weight", 3.0),
            rec_loss_weight=est_cfg.get("rec_loss_weight", 1.0),
            angle_min_force=est_cfg.get("angle_min_force", 1.0),
            max_grad_norm=est_cfg.get("max_grad_norm", 10.0),
        ).to(device)

        if "force_estimator_state_dict" in ckpt:
            estimator.load_state_dict(ckpt["force_estimator_state_dict"])
            print("[ComplianceRunner] Force estimator loaded from stage-1 (TRAINABLE, keeps learning)")
        else:
            print("[ComplianceRunner] WARNING: No estimator weights in stage-1 checkpoint!")

        # Estimator stays in train mode — NOT frozen
        estimator.train()

        # ── History buffer ────────────────────────────────────────────────
        history_buffer = ObsHistoryBuffer(
            num_envs=env.num_envs,
            temporal_steps=temporal_steps,
            obs_dim=raw_obs_dim,
            device=device,
        )

        # ── Activate forces (always on at max in stage 2) ────────────────
        max_force = comp_cfg.get("max_force", 20.0)
        force_event_name = train_cfg.get("force_event_term_name", "persistent_xyz_force")
        try:
            underlying_env = env.unwrapped
            event_cfg = underlying_env.event_manager.get_term_cfg(force_event_name)
            event_cfg.params["force_range"] = (0.0, max_force)
            print(f"[ComplianceRunner] Forces activated: (0, {max_force:.0f}N) via {force_event_name}")
        except Exception as e:
            print(f"[ComplianceRunner] Could not activate forces: {e}")

        # ── GT force indices in critic obs ────────────────────────────────
        # Critic layout: base_lin_vel(3) + base_ang_vel(3) + gravity(3) + vel_cmd(3) +
        #   jpos(12) + jvel(12) + last_action(12) + torque(12) + foot(4) + GT_force_xyz(3) + force_est(3) = 70
        # GT force starts at index 64
        gt_force_start = 64
        gt_force_dim = force_dim  # 3

        # ── Wrap env ──────────────────────────────────────────────────────
        self._compliance_wrapper = ComplianceEnvWrapper(
            env=env,
            frozen_policy=frozen_policy,
            estimator=estimator,
            history_buffer=history_buffer,
            alpha=comp_cfg.get("alpha", 10.0),
            beta=comp_cfg.get("beta", 10.0),
            compliance_reward_weight=comp_cfg.get("compliance_reward_weight", 0.5),
            gt_force_start=gt_force_start,
            gt_force_dim=gt_force_dim,
            device=device,
            reward_type=comp_cfg.get("reward_type", "penalty"),
            gravity_correction=comp_cfg.get("gravity_correction", False),
            use_tracking_reward=comp_cfg.get("use_tracking_reward", True),
            robot_mass=comp_cfg.get("robot_mass", 15.0),
        )

        self._stage1_checkpoint = stage1_path
        self._estimator = estimator
        self._frozen_policy = frozen_policy
        self._force_dim = force_dim
        self._raw_obs_dim = raw_obs_dim
        self._temporal_steps = temporal_steps

        # ── Call parent with WRAPPED env ──────────────────────────────────
        super().__init__(self._compliance_wrapper, train_cfg, log_dir=log_dir, device=device)

        print(
            f"[ComplianceRunner] High-level actor: "
            f"{ComplianceEnvWrapper.HIGH_LEVEL_OBS_DIM} -> "
            f"{train_cfg.get('policy', {}).get('actor_hidden_dims', [256, 128])} -> "
            f"{ComplianceEnvWrapper.HIGH_LEVEL_ACTION_DIM}"
        )
        print(
            f"[ComplianceRunner] Compliance: alpha={comp_cfg.get('alpha', 10.0):.1f}  "
            f"beta={comp_cfg.get('beta', 10.0):.1f}  "
            f"w_comp={comp_cfg.get('compliance_reward_weight', 0.5):.2f}  "
            f"reward={comp_cfg.get('reward_type', 'penalty')}  "
            f"gravity_corr={comp_cfg.get('gravity_correction', False)}  "
            f"tracking_rew={comp_cfg.get('use_tracking_reward', True)}"
        )

    # ── Freeze verification ───────────────────────────────────────────────

    def _verify_frozen(self, it: int) -> None:
        """Verify low-level policy params haven't changed."""
        current = _param_checksum(self._frozen_policy)
        if current != self._frozen_policy_checksum:
            print(
                f"\n{'!' * 80}\n"
                f"  [ComplianceRunner] CRITICAL: Low-level policy weights CHANGED at iteration {it}!\n"
                f"  Expected: {self._frozen_policy_checksum[:16]}\n"
                f"  Got:      {current[:16]}\n"
                f"{'!' * 80}\n"
            )
        else:
            if it % 100 == 0:
                print(f"  [ComplianceRunner] Freeze verified OK (iter {it})")

    # ── Estimator training ────────────────────────────────────────────────

    def _train_estimator_step(self, obs_history: torch.Tensor, gt_force: torch.Tensor,
                               next_raw_obs: torch.Tensor) -> dict:
        """One gradient step on the estimator using rollout data."""
        return self.estimator.train_step(obs_history, gt_force, next_raw_obs)

    # ── Logging ───────────────────────────────────────────────────────────

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        it = locs["it"]
        w = self._compliance_wrapper

        mean_r_comp = w.last_compliance_reward.mean().item()
        mean_force_mag = w.last_gt_force_mag.mean().item()
        mean_action_norm = w.last_action_norm.mean().item()
        mean_track_corr = w.last_tracking_correction.mean().item()

        rewbuffer = locs.get("rewbuffer", [])
        lenbuffer = locs.get("lenbuffer", [])
        collection_time = locs.get("collection_time", 0)
        learn_time = locs.get("learn_time", 0)
        fps = int(self.num_steps_per_env * self.env.num_envs / (collection_time + learn_time + 1e-9))

        mean_rew = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else 0.0
        mean_len = statistics.mean(lenbuffer) if len(lenbuffer) > 0 else 0.0

        # ── Extract PPO loss values ───────────────────────────────────────
        loss_dict = locs.get("loss_dict", {})
        mean_std = self.alg.policy.action_std.mean().item()

        # ── Tensorboard (write everything, including PPO internals) ───────
        if self.log_dir is not None:
            for key, value in loss_dict.items():
                self.writer.add_scalar(f"Loss/{key}", value, it)
            self.writer.add_scalar("Policy/mean_noise_std", mean_std, it)
            self.writer.add_scalar("Train/mean_reward", mean_rew, it)
            self.writer.add_scalar("Train/mean_episode_length", mean_len, it)
            self.writer.add_scalar("Train/fps", fps, it)
            self.writer.add_scalar("Compliance/r_comp_step", mean_r_comp, it)
            self.writer.add_scalar("Compliance/tracking_correction_step", mean_track_corr, it)
            self.writer.add_scalar("Compliance/force_mag_step", mean_force_mag, it)
            self.writer.add_scalar("Compliance/action_norm_step", mean_action_norm, it)
            self.writer.add_scalar("Compliance/alpha", w.alpha, it)
            self.writer.add_scalar("Compliance/beta", w.beta, it)

            # Log ep_infos from the env (base reward breakdown)
            ep_infos = locs.get("ep_infos", [])
            if len(ep_infos) > 0:
                for key in ep_infos[0]:
                    vals = [float(info[key]) for info in ep_infos if key in info]
                    if vals:
                        self.writer.add_scalar(f"Episode/{key}", statistics.mean(vals), it)

        # Verify frozen policy integrity
        self._verify_frozen(it)

        # ── Terminal: compliance-focused only ─────────────────────────────
        iter_time = collection_time + learn_time
        tot_iter = locs.get("tot_iter", it + 1)
        remaining = (tot_iter - it) * iter_time
        eta_str = f"{int(remaining // 3600):02d}:{int((remaining % 3600) // 60):02d}:{int(remaining % 60):02d}"

        print(
            f"\n{'=' * width}\n"
            f"  HAC-LOCO Stage 2 — iter {it}/{tot_iter}  "
            f"({fps} steps/s, {iter_time:.2f}s/iter)  ETA: {eta_str}\n"
            f"{'=' * width}\n"
            f"  Total reward: {mean_rew:.2f}    Episode length: {mean_len:.0f}\n"
            f"\n"
            f"  r_compliance:         {mean_r_comp:+.4f}\n"
            f"  tracking_correction:  {mean_track_corr:+.4f}\n"
            f"  GT force:             {mean_force_mag:.1f} N\n"
            f"  ||a'||:               {mean_action_norm:.4f}\n"
            f"  alpha={w.alpha:.1f}  beta={w.beta:.1f}\n"
            f"\n"
            f"  value_loss: {loss_dict.get('value_function', 0):.4f}  "
            f"surrogate: {loss_dict.get('surrogate', 0):.6f}  "
            f"std: {mean_std:.3f}\n"
            f"{'=' * width}"
        )

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        ckpt["stage1_checkpoint"] = self._stage1_checkpoint
        ckpt["force_estimator_state_dict"] = self._estimator.state_dict()
        ckpt["force_estimator_optimizer_state_dict"] = self._estimator.optimizer.state_dict()
        ckpt["compliance_config"] = {
            "alpha": self._compliance_wrapper.alpha,
            "beta": self._compliance_wrapper.beta,
            "compliance_reward_weight": self._compliance_wrapper.compliance_reward_weight,
        }
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        if "force_estimator_state_dict" in loaded:
            self._estimator.load_state_dict(loaded["force_estimator_state_dict"])
            print("[ComplianceRunner] Estimator weights loaded from checkpoint")
        if load_optimizer and "force_estimator_optimizer_state_dict" in loaded:
            self._estimator.optimizer.load_state_dict(loaded["force_estimator_optimizer_state_dict"])
        if "stage1_checkpoint" in loaded:
            print(f"[ComplianceRunner] Stage-1 ref: {loaded['stage1_checkpoint']}")
        return infos
