"""Agent configuration for EstimatorOnPolicyRunner.

Uses the same PPO hyperparameters as PPORunnerCfg but:
  - class_name = "EstimatorOnPolicyRunner"
  - adds an `estimator` block with VelocityEstimator hyperparameters
  - actor input dim is automatically expanded to num_obs + latent_dim

Env: Go2-Testbench-Estimator-v0  (same env as Go2-Testbench-v0)
Runner: go2_rl_lab.estimator.EstimatorOnPolicyRunner

Critic obs layout for Go2-Testbench-v0 (determines gt_vel_obs_start_idx):
    [0:3]   base_lin_vel    ← gt_velocity lives here
    [3:6]   base_ang_vel
    [6:9]   projected_gravity
    [9:12]  velocity_commands
    [12:24] joint_pos_rel
    [24:36] joint_vel_rel
    [36:48] last_action
    total = 48 dims
"""

import math

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class EstimatorRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner with concurrent velocity-estimator training."""

    class_name: str = "EstimatorOnPolicyRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 30000
    save_interval: int = 200
    experiment_name: str = "go2_estimator"

    # ── PPO policy (actor input dim auto-expanded by EstimatorEnvWrapper) ─
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # ── Estimator hyperparameters ──────────────────────────────────────────
    # These are passed to VelocityEstimator via train_cfg["estimator"].
    estimator: dict = {
        # History window: how many past obs steps the encoder sees
        "temporal_steps": 10,

        # Encoder: obs_history → z_t  (enc_hidden_dims[-1] = latent dim = 64)
        "enc_hidden_dims": [128, 64],

        # Velocity head: z_t → v̂_t (3-dim)
        "v_head_dims": [32, 16],

        # Decoder: l_t → ô_{t+1}  (reconstruction regulariser)
        "dec_hidden_dims": [512, 256, 128],

        "activation": "elu",

        # Supervised learning rate (separate from PPO LR)
        "learning_rate": 1e-3,

        # Loss weights (L_total = w1*L_vel + w3*L_rec)
        "vel_loss_weight": 1.0,
        "rec_loss_weight": 1.0,

        "max_grad_norm": 10.0,

        # Slice start of gt_velocity in critic obs (base_lin_vel is at index 0)
        "gt_vel_obs_start_idx": 0,

        # How many mini-batches to use for each estimator update pass
        "num_estimator_mini_batches": 4,
    }
