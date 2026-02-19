"""Agent configuration for force-estimator training.

Same PPO hyperparams as EstimatorRunnerCfg but with f_head enabled.

Critic obs layout for Go2-Force-Estimator-v0:
    [0:3]   base_lin_vel          ← gt for v_head
    [3:6]   base_ang_vel
    [6:9]   projected_gravity
    [9:12]  velocity_commands
    [12:24] joint_pos_rel
    [24:36] joint_vel_rel
    [36:48] last_action
    [48:50] base_applied_force_xy ← gt for f_head
    total = 50 dims
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ForceEstimatorRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner with concurrent velocity + force estimator training."""

    class_name: str = "EstimatorOnPolicyRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 30000
    save_interval: int = 200
    experiment_name: str = "go2_force_estimator"

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

    estimator: dict = {
        "temporal_steps": 10,
        "enc_hidden_dims": [128, 64],
        "v_head_dims": [32, 16],
        # Force head — this is the new addition
        "f_head_dims": [32, 16],
        "force_dim": 2,
        "dec_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "learning_rate": 1e-3,
        # Loss weights: L = w1*L_vel + w2*L_force + w3*L_rec
        "vel_loss_weight": 1.0,
        "force_loss_weight": 1.0,
        "rec_loss_weight": 1.0,
        "max_grad_norm": 10.0,
        # Ground truth indices in critic obs
        "gt_vel_obs_start_idx": 0,    # base_lin_vel at [0:3]
        "gt_force_obs_start_idx": 48,  # base_applied_force_xy at [48:50]
        "num_estimator_mini_batches": 4,
    }
