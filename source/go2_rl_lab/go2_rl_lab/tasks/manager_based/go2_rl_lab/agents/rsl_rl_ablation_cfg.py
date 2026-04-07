"""Runner configs for force estimator ablation study.

11 configs for Groups A, B, C, E (branch: feature/estimator-ablation).
All inherit from LowLevelRunnerCfg, overriding estimator params.

| ID | Force dims | History | Recon | Network | Torque losses | Compliance |
|----|-----------|---------|-------|---------|---------------|------------|
| A1 | 3D | 10 | Yes | default | No  | No |
| A2 | 3D | 40 | Yes | default | No  | No |
| B1 | 6D | 10 | Yes | default | No  | No |
| B2 | 6D | 40 | Yes | default | No  | No |
| B3 | 6D | 40 | Yes | bigger  | No  | No |
| B4 | 6D | 40 | Yes | bigger  | Yes | No |
| B5 | 6D | 60 | Yes | bigger  | Yes | No |
| C1 | 3D | 10 | No  | default | No  | No |
| C2 | 3D | 40 | No  | default | No  | No |
| E1 | 3D | 20 | Yes | default | No  | force |
| E2 | 4D | 20 | Yes | default | No  | force+torque |
"""

from isaaclab.utils import configclass
from .rsl_rl_lowlevel_cfg import LowLevelRunnerCfg


def _est(
    temporal_steps: int = 20,
    force_dim: int = 3,
    rec_loss_weight: float = 1.0,
    enc_hidden_dims: list[int] | None = None,
    f_head_dims: list[int] | None = None,
    torque_angle_loss_weight: float = 0.0,
    torque_angle_min: float = 0.3,
    yaw_loss_weight: float = 0.0,
) -> dict:
    """Build estimator config dict, overriding only what changes."""
    d = {
        "temporal_steps": temporal_steps,
        "enc_hidden_dims": enc_hidden_dims or [128, 64],
        "f_head_dims": f_head_dims or [32, 16],
        "force_dim": force_dim,
        "dec_hidden_dims": [256, 128],
        "activation": "elu",
        "learning_rate": 1e-3,
        "force_loss_weight": 1.0,
        "angle_loss_weight": 3.0,
        "rec_loss_weight": rec_loss_weight,
        "angle_min_force": 1.0,
        "max_grad_norm": 10.0,
        "torque_angle_loss_weight": torque_angle_loss_weight,
        "torque_angle_min": torque_angle_min,
        "yaw_loss_weight": yaw_loss_weight,
    }
    return d


# ── Group A: History length (3D, with reconstruction) ───────────────────────


@configclass
class AblationA1Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_A1_h10_3d_rec"
    estimator: dict = _est(temporal_steps=10, force_dim=3, rec_loss_weight=1.0)


@configclass
class AblationA2Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_A2_h40_3d_rec"
    estimator: dict = _est(temporal_steps=40, force_dim=3, rec_loss_weight=1.0)


# ── Group B: 6D wrench estimation ──────────────────────────────────────────


@configclass
class AblationB1Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_B1_h10_6d_rec"
    force_event_term_name: str = "persistent_wrench"
    estimator: dict = _est(temporal_steps=10, force_dim=6, rec_loss_weight=1.0)


@configclass
class AblationB2Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_B2_h40_6d_rec"
    force_event_term_name: str = "persistent_wrench"
    estimator: dict = _est(temporal_steps=40, force_dim=6, rec_loss_weight=1.0)


@configclass
class AblationB3Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_B3_h40_6d_rec_big"
    force_event_term_name: str = "persistent_wrench"
    estimator: dict = _est(
        temporal_steps=40,
        force_dim=6,
        rec_loss_weight=1.0,
        enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32],
    )


@configclass
class AblationB4Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_B4_h40_6d_rec_big_tqloss"
    force_event_term_name: str = "persistent_wrench"
    estimator: dict = _est(
        temporal_steps=40,
        force_dim=6,
        rec_loss_weight=1.0,
        enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32],
        torque_angle_loss_weight=3.0,
        torque_angle_min=0.3,
        yaw_loss_weight=3.0,
    )


@configclass
class AblationB5Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_B5_h60_6d_rec_big_tqloss"
    force_event_term_name: str = "persistent_wrench"
    estimator: dict = _est(
        temporal_steps=60,
        force_dim=6,
        rec_loss_weight=1.0,
        enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32],
        torque_angle_loss_weight=3.0,
        torque_angle_min=0.3,
        yaw_loss_weight=3.0,
    )


# ── Group C: No reconstruction loss ────────────────────────────────────────


@configclass
class AblationC1Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_C1_h10_3d_norec"
    estimator: dict = _est(temporal_steps=10, force_dim=3, rec_loss_weight=0.0)


@configclass
class AblationC2Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_C2_h40_3d_norec"
    estimator: dict = _est(temporal_steps=40, force_dim=3, rec_loss_weight=0.0)


# ── Group E: Compliance reward (GT force/torque) ───────────────────────────


@configclass
class AblationE1Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_E1_h20_3d_compliance"
    estimator: dict = _est(temporal_steps=20, force_dim=3, rec_loss_weight=1.0)


@configclass
class AblationE2Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_E2_h20_4d_compliance"
    force_event_term_name: str = "persistent_wrench"
    estimator: dict = _est(temporal_steps=20, force_dim=4, rec_loss_weight=1.0)
