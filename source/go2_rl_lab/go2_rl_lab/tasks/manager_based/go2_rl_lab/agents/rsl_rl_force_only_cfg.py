"""Agent configuration for force-only estimator training.

Uses ForceOnPolicyRunner with ForceEstimatorPPO (HAC-LOCO style optimizer).
No velocity head — force estimation only.

Critic obs layout for Go2-Force-Only-v0:
    [0:3]   base_lin_vel               ← privileged
    [3:6]   base_ang_vel
    [6:9]   projected_gravity
    [9:12]  velocity_commands
    [12:24] joint_pos_rel
    [24:36] joint_vel_rel
    [36:48] last_action
    [48:60] applied_torque (scale=0.1)
    [60:64] foot_contact_force_norms (scale=0.01)
    [64:66] base_applied_force_xy      ← gt for f_head
    total = 66 dims
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ForceOnlyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner with force estimator updated per mini-batch (HAC-LOCO style)."""

    class_name: str = "ForceOnPolicyRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 30000
    save_interval: int = 200
    experiment_name: str = "go2_force_only"

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
        "temporal_steps": 20,
        # Force estimator architecture
        "enc_hidden_dims": [128, 64],
        "f_head_dims": [32, 16],
        "force_dim": 2,
        "dec_hidden_dims": [256, 128],
        "activation": "elu",
        "learning_rate": 1e-3,
        # Loss weights: L = w1*L_force + w2*L_angle + w3*L_rec
        "force_loss_weight": 1.0,
        "angle_loss_weight": 1.0,
        "rec_loss_weight": 1.0,
        "angle_min_force": 1.0,  # skip angular loss when |f_gt| < 1N
        "max_grad_norm": 10.0,
        # Ground truth index in critic obs
        "gt_force_obs_start_idx": 64,  # base_applied_force_xy at [64:66]
        # Force activation gate: forces start at 0 and activate once
        # mean episode reward exceeds threshold (policy must be robust first).
        "force_activation_reward_threshold": 30.0,
        "force_event_term_name": "persistent_xy_force",
        "max_force": 20.0,
    }
