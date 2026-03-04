"""Agent configuration for compliant locomotion — NO foot contacts, 3D XYZ force estimation.

Estimator trains from scratch (no checkpoint). Angular error threshold = 6 deg.
Force estimator predicts fx, fy, fz (force_dim=3).

Usage:
    python train.py --task Go2-Compliant-NoFoot-XYZ-v0
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CompliantNoFootXyzRunnerCfg(RslRlOnPolicyRunnerCfg):
    """CompliantOnPolicyRunner config — no foot contacts, 3D force estimator from scratch."""

    class_name: str = "CompliantOnPolicyRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 200
    experiment_name: str = "go2_compliant_no_foot_xyz"

    # Phase gates
    force_activation_reward_threshold: float = 30.0
    estimator_angular_threshold: float = 6.0
    force_event_term_name: str = "persistent_xyz_force"
    max_force: float = 20.0

    # Compliance parameters (XY only — fz does not modulate velocity)
    compliance_alpha: float = 5.0
    compliance_beta: float = 50.0

    # Force estimator architecture — 3D force output
    estimator: dict = {
        "temporal_steps": 20,
        "enc_hidden_dims": [128, 64],
        "f_head_dims": [32, 16],
        "force_dim": 3,
        "dec_hidden_dims": [256, 128],
        "activation": "elu",
        "learning_rate": 1e-3,
        "force_loss_weight": 1.0,
        "angle_loss_weight": 1.0,
        "rec_loss_weight": 1.0,
        "angle_min_force": 1.0,
        "max_grad_norm": 10.0,
    }

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
