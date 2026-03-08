"""Agent configuration for HAC-LOCO stage 2 — high-level compliance training.

Trains a lightweight compliance policy [256, 128] -> 3 residual velocity commands
on top of a frozen low-level locomotion policy from stage 1.

The force estimator is TRAINABLE (continues learning from stage-1 checkpoint).

Usage:
    python train.py --task Go2-HAC-LOCO-Stage2-v0 \
        --compliance_stage1_checkpoint /path/to/stage1/model_XXXX.pt
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HacLocoStage2RunnerCfg(RslRlOnPolicyRunnerCfg):
    """ComplianceOnPolicyRunner config for HAC-LOCO stage 2."""

    class_name: str = "ComplianceOnPolicyRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 100
    experiment_name: str = "go2_hac_loco_stage2"

    # Force event (same env as stage 1)
    force_event_term_name: str = "persistent_xyz_force"

    # Compliance parameters (HAC-LOCO paper)
    compliance: dict = {
        "stage1_checkpoint": "",  # MUST be set via CLI or override
        "alpha": 10.0,           # Force threshold (N)
        "beta": 10.0,            # Virtual impedance
        "compliance_reward_weight": 0.5,
        "max_force": 20.0,
    }

    # Force estimator architecture (must match stage-1)
    estimator: dict = {
        "temporal_steps": 20,
        "enc_hidden_dims": [128, 64],
        "f_head_dims": [32, 16],
        "force_dim": 3,
        "dec_hidden_dims": [256, 128],
        "activation": "elu",
        "learning_rate": 1e-3,
        "force_loss_weight": 1.0,
        "angle_loss_weight": 3.0,
        "rec_loss_weight": 1.0,
        "angle_min_force": 1.0,
        "max_grad_norm": 10.0,
    }

    # High-level policy: lightweight [256, 128]
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.25,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,       # reduced from 0.01 to discourage entropy growth
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,     # reduced from 1e-3 for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,         # tighter KL constraint (was 0.01)
        max_grad_norm=1.0,
    )
