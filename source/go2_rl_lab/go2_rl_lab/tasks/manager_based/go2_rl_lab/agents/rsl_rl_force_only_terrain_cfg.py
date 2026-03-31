"""Agent configuration for force-only estimator training with terrain curriculum.

Same as rsl_rl_force_only_cfg.py but with:
1. Force activation threshold set to 9999.0 (effectively disabled).
2. Separate experiment name for terrain logs/checkpoints.

The user will first evaluate stair-climbing performance without forces,
then lower the threshold to activate force estimation.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ForceOnlyTerrainRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner with force estimator (disabled initially) on terrain curriculum."""

    class_name: str = "ForceOnPolicyRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 30000
    save_interval: int = 200
    experiment_name: str = "go2_force_only_terrain"

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
        # Force activation gate: effectively disabled (threshold unreachable)
        # Lower this once stair-climbing performance is evaluated.
        "force_activation_reward_threshold": 9999.0,
        "force_event_term_name": "persistent_xy_force",
        "max_force": 20.0,
    }
