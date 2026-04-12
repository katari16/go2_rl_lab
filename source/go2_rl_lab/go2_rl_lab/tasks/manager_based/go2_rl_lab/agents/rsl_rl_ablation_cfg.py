"""Runner configs for force estimator ablation study.

Groups A-E (batch 1, 20N): 11 configs.
Group H (batch 2, 50N/100N): 9 configs x 2 force levels = 18 configs.

Batch 1 (20N):
| ID | Dims | h  | Recon | Network | Torque losses | Compliance |
|----|------|----|-------|---------|---------------|------------|
| A1 | 3D   | 10 | Yes   | default | No            | No         |
| A2 | 3D   | 40 | Yes   | default | No            | No         |
| B1 | 6D   | 10 | Yes   | default | No            | No         |
| B2 | 6D   | 40 | Yes   | default | No            | No         |
| B3 | 6D   | 40 | Yes   | bigger  | No            | No         |
| B4 | 6D   | 40 | Yes   | bigger  | Yes           | No         |
| B5 | 6D   | 60 | Yes   | bigger  | Yes           | No         |
| C1 | 3D   | 10 | No    | default | No            | No         |
| C2 | 3D   | 40 | No    | default | No            | No         |
| E1 | 3D   | 20 | Yes   | default | No            | force      |
| E2 | 4D   | 20 | Yes   | default | No            | force+tq   |

Batch 2 (50N / 100N each):
| ID  | Dims | h  | Network | Losses                   | Est accuracy reward |
|-----|------|----|---------|--------------------------|---------------------|
| H1  | 3D   | 30 | default | force+angle+rec          | No                  |
| H3a | 6D   | 40 | bigger  | force+angle+rec+yaw_loss | No                  |
| H3b | 6D   | 30 | bigger  | force+angle+rec+yaw_loss | No                  |
| H3c | 6D   | 30 | default | force+angle+rec+yaw_loss | No                  |
| H5  | 4D   | 30 | default | force+angle+rec+yaw_loss | No                  |
| H6  | 4D   | 30 | default | force+angle+rec          | No                  |
| H7  | 4D   | 30 | default | force+angle+rec          | Yes                 |
| H8  | 3D   | 30 | default | force+angle+rec          | Yes                 |
| H9  | 2D   | 30 | default | force+angle+rec          | No                  |
"""

from isaaclab.utils import configclass
from .rsl_rl_lowlevel_cfg import LowLevelRunnerCfg


def _est(
    temporal_steps: int = 20,
    force_dim: int = 3,
    rec_loss_weight: float = 1.0,
    enc_hidden_dims: list[int] | None = None,
    f_head_dims: list[int] | None = None,
    angle_loss_weight: float = 3.0,
    torque_angle_loss_weight: float = 0.0,
    torque_angle_min: float = 0.3,
    yaw_loss_weight: float = 0.0,
    tcn_mode: str = "none",
    tcn_channels: list[int] | None = None,
    tcn_kernel_size: int = 3,
    tcn_dilations: list[int] | None = None,
    temporal_decay: str = "none",
    force_layout: str = "auto",
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
        "angle_loss_weight": angle_loss_weight,
        "rec_loss_weight": rec_loss_weight,
        "angle_min_force": 1.0,
        "max_grad_norm": 10.0,
        "torque_angle_loss_weight": torque_angle_loss_weight,
        "torque_angle_min": torque_angle_min,
        "yaw_loss_weight": yaw_loss_weight,
        "tcn_mode": tcn_mode,
        "tcn_channels": tcn_channels,
        "tcn_kernel_size": tcn_kernel_size,
        "tcn_dilations": tcn_dilations,
        "temporal_decay": temporal_decay,
        "force_layout": force_layout,
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
    max_torque: float = 5.0
    estimator: dict = _est(temporal_steps=10, force_dim=6, rec_loss_weight=1.0)


@configclass
class AblationB2Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_B2_h40_6d_rec"
    force_event_term_name: str = "persistent_wrench"
    max_torque: float = 5.0
    estimator: dict = _est(temporal_steps=40, force_dim=6, rec_loss_weight=1.0)


@configclass
class AblationB3Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_B3_h40_6d_rec_big"
    force_event_term_name: str = "persistent_wrench"
    max_torque: float = 5.0
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
    max_torque: float = 5.0
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
    max_torque: float = 10.0
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
    max_torque: float = 5.0
    estimator: dict = _est(temporal_steps=20, force_dim=4, rec_loss_weight=1.0)


# ── Group H: Batch 2 dimension sweep (50N / 100N) ─────────────────────────
# Each config has a 50N and 100N variant (suffix _50N / _100N).


def _h_cfg(name_suffix: str, force_dim: int, temporal_steps: int, max_force: float,
           force_event: str = "persistent_xyz_force",
           max_torque: float = 5.0,
           enc_hidden_dims: list[int] | None = None,
           f_head_dims: list[int] | None = None,
           yaw_loss_weight: float = 0.0,
           torque_angle_loss_weight: float = 0.0,
           rec_loss_weight: float = 1.0) -> dict:
    """Return a dict of class attributes for H-series configs."""
    result = {
        "experiment_name": f"ablation_{name_suffix}",
        "force_event_term_name": force_event,
        "max_force": max_force,
        "estimator": _est(
            temporal_steps=temporal_steps,
            force_dim=force_dim,
            rec_loss_weight=rec_loss_weight,
            enc_hidden_dims=enc_hidden_dims,
            f_head_dims=f_head_dims,
            yaw_loss_weight=yaw_loss_weight,
            torque_angle_loss_weight=torque_angle_loss_weight,
        ),
    }
    if force_event != "persistent_xyz_force":
        result["max_torque"] = max_torque
    return result


# ── H1: 3D, h=30, default ────────────────────────────────────────────────

@configclass
class AblationH1_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H1_3d_h30_50N"
    max_force: float = 50.0
    estimator: dict = _est(temporal_steps=30, force_dim=3)


@configclass
class AblationH1_100Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H1_3d_h30_100N"
    max_force: float = 100.0
    estimator: dict = _est(temporal_steps=30, force_dim=3)


# ── H3a: 6D, h=40, bigger, yaw_loss ──────────────────────────────────────

@configclass
class AblationH3a_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H3a_6d_h40_big_yaw_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
    )


@configclass
class AblationH3a_100Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H3a_6d_h40_big_yaw_100N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 100.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
    )


# ── H3b: 6D, h=30, bigger, yaw_loss ──────────────────────────────────────

@configclass
class AblationH3b_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H3b_6d_h30_big_yaw_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
    )


@configclass
class AblationH3b_100Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H3b_6d_h30_big_yaw_100N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 100.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
    )


# ── H3c: 6D, h=30, default, yaw_loss ─────────────────────────────────────

@configclass
class AblationH3c_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H3c_6d_h30_def_yaw_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=6, yaw_loss_weight=3.0,
        torque_angle_loss_weight=3.0,
    )


@configclass
class AblationH3c_100Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H3c_6d_h30_def_yaw_100N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 100.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=6, yaw_loss_weight=3.0,
        torque_angle_loss_weight=3.0,
    )


# ── H5: 4D, h=30, default, yaw_loss ──────────────────────────────────────

@configclass
class AblationH5_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H5_4d_h30_yaw_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=4, yaw_loss_weight=3.0,
    )


@configclass
class AblationH5_100Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H5_4d_h30_yaw_100N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 100.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=4, yaw_loss_weight=3.0,
    )


# ── H6: 4D, h=30, default, no yaw_loss ───────────────────────────────────

@configclass
class AblationH6_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H6_4d_h30_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(temporal_steps=30, force_dim=4)


@configclass
class AblationH6_100Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H6_4d_h30_100N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 100.0
    max_torque: float = 5.0
    estimator: dict = _est(temporal_steps=30, force_dim=4)


# ── H7: 4D, h=30, default, + force est accuracy reward ───────────────────

@configclass
class AblationH7_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H7_4d_h30_estrew_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(temporal_steps=30, force_dim=4)


@configclass
class AblationH7_100Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H7_4d_h30_estrew_100N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 100.0
    max_torque: float = 5.0
    estimator: dict = _est(temporal_steps=30, force_dim=4)


# ── H8: 3D, h=30, default, + force est accuracy reward ───────────────────

@configclass
class AblationH8_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H8_3d_h30_estrew_50N"
    max_force: float = 50.0
    estimator: dict = _est(temporal_steps=30, force_dim=3)


@configclass
class AblationH8_100Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H8_3d_h30_estrew_100N"
    max_force: float = 100.0
    estimator: dict = _est(temporal_steps=30, force_dim=3)


# ── H9: 2D, h=30, default ────────────────────────────────────────────────

@configclass
class AblationH9_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H9_2d_h30_50N"
    max_force: float = 50.0
    estimator: dict = _est(temporal_steps=30, force_dim=2)


@configclass
class AblationH9_100Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H9_2d_h30_100N"
    max_force: float = 100.0
    estimator: dict = _est(temporal_steps=30, force_dim=2)


# ── H12a: 6D, h=40, bigger, yaw_loss, TCN preprocessor (based on H3a) ───

@configclass
class AblationH12a_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H12a_6d_h40_tcnpre_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
        tcn_mode="preprocessor", tcn_channels=[64, 64], tcn_kernel_size=3,
        tcn_dilations=[1, 2],
    )


# ── H12b: 6D, h=40, bigger, yaw_loss, TCN replacement (based on H3a) ────

@configclass
class AblationH12b_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H12b_6d_h40_tcnrep_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
        tcn_mode="replacement", tcn_channels=[64, 64], tcn_kernel_size=3,
        tcn_dilations=[1, 2],
    )


# ── H13a: 4D, h=30, default, est reward, TCN preprocessor (based on H7) ─

@configclass
class AblationH13a_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H13a_4d_h30_tcnpre_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=4,
        tcn_mode="preprocessor", tcn_channels=[64, 64], tcn_kernel_size=3,
        tcn_dilations=[1, 2],
    )


# ── H13b: 4D, h=30, default, est reward, TCN replacement (based on H7) ──

@configclass
class AblationH13b_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H13b_4d_h30_tcnrep_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=4,
        tcn_mode="replacement", tcn_channels=[64, 64], tcn_kernel_size=3,
        tcn_dilations=[1, 2],
    )


# ── H15: 6D, h=40, bigger, yaw_loss, constant force (H3a rerun with fixes) ───

@configclass
class AblationH15_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H15_6d_h40_big_yaw_fixed_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
    )


# ── H14: 6D, h=40, bigger, yaw_loss, trapezoid force profile (based on H3a) ─

@configclass
class AblationH14_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H14_6d_h40_big_yaw_trap_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
    )


# ── H16: 6D, h=40, bigger, EQUAL loss weights (all 1.0) ────────────────────

@configclass
class AblationH16_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H16_6d_h40_big_equal_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], angle_loss_weight=1.0,
        yaw_loss_weight=1.0, torque_angle_loss_weight=1.0,
    )


# ── H17: 6D, h=40, bigger, linear decay temporal weighting ─────────────────

@configclass
class AblationH17_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H17_6d_h40_big_lindecay_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
        temporal_decay="linear",
    )


# ── H18: 6D, h=40, bigger, TCN preprocessor + detached recon loss ──────────

@configclass
class AblationH18_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H18_6d_h40_big_tcnpre_detach_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0, torque_angle_loss_weight=3.0,
        tcn_mode="preprocessor", tcn_channels=[64, 64], tcn_kernel_size=3,
        tcn_dilations=[1, 2],
    )


# ── H19: 3D planar wrench (Fx, Fy, τ_yaw) ─────────────────────────────────

@configclass
class AblationH19_50Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_H19_xy_yaw_h40_big_50N"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=40, force_dim=3, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0,
        force_layout="xy_yaw",
    )
