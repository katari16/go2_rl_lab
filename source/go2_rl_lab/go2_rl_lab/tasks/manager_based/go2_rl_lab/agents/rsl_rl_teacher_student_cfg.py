"""Runner configs for PAINT-style teacher-student ablation (Group D).

PAINT two-stage approach:
  Stage 1: Train teacher with CompliantOnPolicyRunner (Go2-LowLevel-v0)
           → teacher sees GT wrench, learns privileged policy
  Stage 2: Train student with PaintRunner, loading teacher checkpoint
           → student sees estimated wrench from intent estimator
           → PPO + KL(π_student || π_teacher) + supervised estimator

| ID | Force dims | History | KL weight | Notes |
|----|-----------|---------|-----------|-------|
| D1 | 3D | 20 | 1.0 | PAINT baseline |
| D2 | 4D | 20 | 1.0 | + τ_yaw estimation |

Usage:
  # Stage 1: train teacher (same as baseline)
  python train.py --task Go2-LowLevel-v0 --headless

  # Stage 2: train student with teacher checkpoint
  python train.py --task Go2-Ablation-D1-v0 --headless \\
      --resume --checkpoint <path/to/teacher/model_XXXX.pt>
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PaintBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Base config for PAINT-style student training."""

    class_name: str = "PaintRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 20000
    save_interval: int = 500

    # KL distillation weight (PAINT λ_KL in Eq. 9)
    kl_weight: float = 1.0

    # Phase gates
    force_activation_reward_threshold: float = 30.0
    estimator_angular_threshold: float = 7.0
    force_event_term_name: str = "persistent_xyz_force"
    max_force: float = 20.0

    # Compliance
    compliance_alpha: float = 5.0
    compliance_beta: float = 50.0

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


@configclass
class AblationD1Cfg(PaintBaseRunnerCfg):
    """D1: PAINT-style, 3D force, h=20."""
    experiment_name: str = "ablation_D1_h20_3d_paint"
    estimator: dict = {
        "temporal_steps": 20,
        "hidden_dims": [128, 64, 32],
        "force_dim": 3,
        "activation": "elu",
        "learning_rate": 1e-3,
        "max_grad_norm": 10.0,
    }


@configclass
class AblationD2Cfg(PaintBaseRunnerCfg):
    """D2: PAINT-style, 4D (Fx,Fy,Fz,τ_yaw), h=20."""
    experiment_name: str = "ablation_D2_h20_4d_paint"
    force_event_term_name: str = "persistent_wrench"
    estimator: dict = {
        "temporal_steps": 20,
        "hidden_dims": [128, 64, 32],
        "force_dim": 4,
        "activation": "elu",
        "learning_rate": 1e-3,
        "max_grad_norm": 10.0,
    }
