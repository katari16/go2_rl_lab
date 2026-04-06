"""Runner configs for teacher-student force estimator ablation (Group D).

| ID | Force dims | History | Distillation | KL weight |
|----|-----------|---------|--------------|-----------|
| D1 | 3D | 20 | Yes | 1.0 |
| D2 | 4D | 20 | Yes | 1.0 |
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TeacherStudentBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Base config for teacher-student runner."""

    class_name: str = "TeacherStudentRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 20000
    save_interval: int = 500

    # Phase gates
    force_activation_reward_threshold: float = 30.0
    teacher_angular_threshold: float = 7.0
    student_angular_threshold: float = 7.0
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
class AblationD1Cfg(TeacherStudentBaseRunnerCfg):
    experiment_name: str = "ablation_D1_h20_3d_teacherstudent"
    estimator: dict = {
        "temporal_steps": 20,
        "enc_hidden_dims": [128, 64],
        "f_head_dims": [32, 16],
        "force_dim": 3,
        "dec_hidden_dims": [256, 128],
        "activation": "elu",
        "teacher_lr": 1e-3,
        "student_lr": 1e-3,
        "force_loss_weight": 1.0,
        "angle_loss_weight": 3.0,
        "rec_loss_weight": 1.0,
        "kl_loss_weight": 1.0,
        "angle_min_force": 1.0,
        "max_grad_norm": 10.0,
    }


@configclass
class AblationD2Cfg(TeacherStudentBaseRunnerCfg):
    experiment_name: str = "ablation_D2_h20_4d_teacherstudent"
    force_event_term_name: str = "persistent_wrench"
    estimator: dict = {
        "temporal_steps": 20,
        "enc_hidden_dims": [128, 64],
        "f_head_dims": [32, 16],
        "force_dim": 4,
        "dec_hidden_dims": [256, 128],
        "activation": "elu",
        "teacher_lr": 1e-3,
        "student_lr": 1e-3,
        "force_loss_weight": 1.0,
        "angle_loss_weight": 3.0,
        "rec_loss_weight": 1.0,
        "kl_loss_weight": 1.0,
        "angle_min_force": 1.0,
        "max_grad_norm": 10.0,
    }
