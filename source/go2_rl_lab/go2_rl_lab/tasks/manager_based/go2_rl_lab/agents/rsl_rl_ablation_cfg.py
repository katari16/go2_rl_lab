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


# ══════════════════════════════════════════════════════════════════════════════
# Group S: Frozen-policy estimator-only training (v3 base policy)
#
# All runs: h=30, 50N, 10% force-free, FrozenPolicyEstimatorRunner
# Policy loaded from v3 checkpoint via --checkpoint CLI arg
#
# | Run | Dims | Network | Key feature                    |
# |-----|------|---------|--------------------------------|
# | S1  | 2D   | default | simplest baseline (Fx, Fy)     |
# | S2  | 3D   | default | standard XYZ                   |
# | S3  | 3D   | bigger  | planar xy_yaw (Fx, Fy, τ_yaw)  |
# | S4  | 4D   | default | xyz + yaw                      |
# | S5  | 6D   | bigger  | equal weights (all 1.0)        |
# | S6  | 6D   | bigger  | weighted (yaw=3, tq_angle=3)   |
# | S7  | 6D   | bigger  | + linear decay                 |
# | S8  | 6D   | bigger  | + TCN preprocessor             |
# | S9  | 6D   | bigger  | + TCN replacement              |
# ══════════════════════════════════════════════════════════════════════════════

def _frozen_base(**overrides) -> dict:
    """Base config dict for frozen-policy S-series runs."""
    d = {
        "class_name": "FrozenPolicyEstimatorRunner",
        "force_free_fraction": 0.1,
        "max_force": 50.0,
        "max_torque": 5.0,
    }
    d.update(overrides)
    return d


@configclass
class AblationS1Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_S1_2d_h30_frozen"
    class_name: str = "FrozenPolicyEstimatorRunner"
    max_iterations: int = 5000
    force_free_fraction: float = 0.1
    force_event_term_name: str = "persistent_xyz_force"
    max_force: float = 50.0
    estimator: dict = _est(temporal_steps=30, force_dim=2)


@configclass
class AblationS2Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_S2_3d_h30_frozen"
    class_name: str = "FrozenPolicyEstimatorRunner"
    max_iterations: int = 5000
    force_free_fraction: float = 0.1
    force_event_term_name: str = "persistent_xyz_force"
    max_force: float = 50.0
    estimator: dict = _est(temporal_steps=30, force_dim=3)


@configclass
class AblationS3Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_S3_xy_yaw_h30_frozen"
    class_name: str = "FrozenPolicyEstimatorRunner"
    max_iterations: int = 5000
    force_free_fraction: float = 0.1
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=3, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0,
        force_layout="xy_yaw",
    )


@configclass
class AblationS4Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_S4_4d_h30_frozen"
    class_name: str = "FrozenPolicyEstimatorRunner"
    max_iterations: int = 5000
    force_free_fraction: float = 0.1
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=4,
        yaw_loss_weight=3.0,
    )


@configclass
class AblationS5Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_S5_6d_h30_equal_frozen"
    class_name: str = "FrozenPolicyEstimatorRunner"
    max_iterations: int = 5000
    force_free_fraction: float = 0.1
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32],
    )


@configclass
class AblationS6Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_S6_6d_h30_weighted_frozen"
    class_name: str = "FrozenPolicyEstimatorRunner"
    max_iterations: int = 5000
    force_free_fraction: float = 0.1
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0,
        torque_angle_loss_weight=3.0,
    )


@configclass
class AblationS7Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_S7_6d_h30_lindecay_frozen"
    class_name: str = "FrozenPolicyEstimatorRunner"
    max_iterations: int = 5000
    force_free_fraction: float = 0.1
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0,
        torque_angle_loss_weight=3.0, temporal_decay="linear",
    )


@configclass
class AblationS8Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_S8_6d_h30_tcnpre_frozen"
    class_name: str = "FrozenPolicyEstimatorRunner"
    max_iterations: int = 5000
    force_free_fraction: float = 0.1
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0,
        torque_angle_loss_weight=3.0,
        tcn_mode="preprocessor", tcn_channels=[64, 64],
        tcn_kernel_size=3, tcn_dilations=[1, 2],
    )


@configclass
class AblationS9Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_S9_6d_h30_tcnrep_frozen"
    class_name: str = "FrozenPolicyEstimatorRunner"
    max_iterations: int = 5000
    force_free_fraction: float = 0.1
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 50.0
    max_torque: float = 5.0
    estimator: dict = _est(
        temporal_steps=30, force_dim=6, enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32], yaw_loss_weight=3.0,
        torque_angle_loss_weight=3.0,
        tcn_mode="replacement", tcn_channels=[64, 64],
        tcn_kernel_size=3, tcn_dilations=[1, 2],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Group J: 4D wrench + estimation accuracy reward sweep
#
# All runs: CompliantOnPolicyRunner, 4D, h=40, 30N force, 10Nm yaw torque
#
# | Run | Weight | Sigma | TCN        | Rec loss | Notes              |
# |-----|--------|-------|------------|----------|--------------------|
# | J1  | 1      | 1.0   | no         | yes      | weight sweep       |
# | J2  | 10     | 1.0   | no         | yes      | weight sweep       |
# | J3  | 50     | 1.0   | no         | yes      | weight sweep (base)|
# | J4  | 100    | 1.0   | no         | yes      | weight sweep       |
# | J5  | 50     | 1.0   | preprocess | yes      | TCN detached       |
# | J6  | 50     | 1.0   | no         | no       | no rec loss        |
# | J7  | 50     | 0.25  | no         | yes      | sigma sharp        |
# | J8  | 50     | 4.0   | no         | yes      | sigma lenient      |
# ══════════════════════════════════════════════════════════════════════════════

_J_BASE = dict(
    force_event_term_name="persistent_wrench",
    max_force=30.0,
    max_torque=10.0,
)


@configclass
class AblationJ1Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_J1_4d_h40_estrew_w1_30N"
    force_event_term_name: str = _J_BASE["force_event_term_name"]
    max_force: float = _J_BASE["max_force"]
    max_torque: float = _J_BASE["max_torque"]
    estimator: dict = _est(temporal_steps=40, force_dim=4, yaw_loss_weight=3.0)


@configclass
class AblationJ2Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_J2_4d_h40_estrew_w10_30N"
    force_event_term_name: str = _J_BASE["force_event_term_name"]
    max_force: float = _J_BASE["max_force"]
    max_torque: float = _J_BASE["max_torque"]
    estimator: dict = _est(temporal_steps=40, force_dim=4, yaw_loss_weight=3.0)


@configclass
class AblationJ3Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_J3_4d_h40_estrew_w50_30N"
    force_event_term_name: str = _J_BASE["force_event_term_name"]
    max_force: float = _J_BASE["max_force"]
    max_torque: float = _J_BASE["max_torque"]
    estimator: dict = _est(temporal_steps=40, force_dim=4, yaw_loss_weight=3.0)


@configclass
class AblationJ4Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_J4_4d_h40_estrew_w100_30N"
    force_event_term_name: str = _J_BASE["force_event_term_name"]
    max_force: float = _J_BASE["max_force"]
    max_torque: float = _J_BASE["max_torque"]
    estimator: dict = _est(temporal_steps=40, force_dim=4, yaw_loss_weight=3.0)


@configclass
class AblationJ5Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_J5_4d_h40_estrew_w50_tcnpre_30N"
    force_event_term_name: str = _J_BASE["force_event_term_name"]
    max_force: float = _J_BASE["max_force"]
    max_torque: float = _J_BASE["max_torque"]
    estimator: dict = _est(
        temporal_steps=40, force_dim=4, yaw_loss_weight=3.0,
        tcn_mode="preprocessor", tcn_channels=[64, 64],
        tcn_kernel_size=3, tcn_dilations=[1, 2],
    )


@configclass
class AblationJ6Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_J6_4d_h40_estrew_w50_norec_30N"
    force_event_term_name: str = _J_BASE["force_event_term_name"]
    max_force: float = _J_BASE["max_force"]
    max_torque: float = _J_BASE["max_torque"]
    estimator: dict = _est(
        temporal_steps=40, force_dim=4, yaw_loss_weight=3.0,
        rec_loss_weight=0.0,
    )


@configclass
class AblationJ7Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_J7_4d_h40_estrew_w50_s025_30N"
    force_event_term_name: str = _J_BASE["force_event_term_name"]
    max_force: float = _J_BASE["max_force"]
    max_torque: float = _J_BASE["max_torque"]
    estimator: dict = _est(temporal_steps=40, force_dim=4, yaw_loss_weight=3.0)


@configclass
class AblationJ8Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_J8_4d_h40_estrew_w50_s4_30N"
    force_event_term_name: str = _J_BASE["force_event_term_name"]
    max_force: float = _J_BASE["max_force"]
    max_torque: float = _J_BASE["max_torque"]
    estimator: dict = _est(temporal_steps=40, force_dim=4, yaw_loss_weight=3.0)


# ══════════════════════════════════════════════════════════════════════════════
# Group K: B4 re-run (yaw bug fixed) + estimation accuracy reward (20N, 5Nm)
#
# | Run | Est-acc weight | Notes                          |
# |-----|----------------|--------------------------------|
# | K1  | —              | B4 re-run, yaw bug now fixed   |
# | K2  | 50             | + est accuracy reward           |
# | K3  | 100            | + est accuracy reward           |
# ══════════════════════════════════════════════════════════════════════════════

_K_EST = dict(
    temporal_steps=40, force_dim=6, rec_loss_weight=1.0,
    enc_hidden_dims=[256, 128], f_head_dims=[64, 32],
    torque_angle_loss_weight=3.0, torque_angle_min=0.3,
    yaw_loss_weight=3.0,
)


@configclass
class AblationK1Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_K1_6d_h40_big_yawfix_20N"
    force_event_term_name: str = "persistent_wrench"
    max_torque: float = 5.0
    estimator: dict = _est(**_K_EST)


@configclass
class AblationK2Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_K2_6d_h40_big_estrew_w50_20N"
    force_event_term_name: str = "persistent_wrench"
    max_torque: float = 5.0
    estimator: dict = _est(**_K_EST)


@configclass
class AblationK3Cfg(LowLevelRunnerCfg):
    experiment_name: str = "ablation_K3_6d_h40_big_estrew_w100_20N"
    force_event_term_name: str = "persistent_wrench"
    max_torque: float = 5.0
    estimator: dict = _est(**_K_EST)


# ── P-series: PAINT-profile ablations ─────────────────────────────────────

_PAINT_BASE = dict(
    temporal_steps=30, force_dim=4, rec_loss_weight=1.0,
    enc_hidden_dims=[128, 64], f_head_dims=[32, 16],
    yaw_loss_weight=3.0,
)


@configclass
class AblationP1Cfg(LowLevelRunnerCfg):
    """P1: History sweep h10."""
    experiment_name: str = "ablation_P1_h10_4d"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 10})


@configclass
class AblationP2Cfg(LowLevelRunnerCfg):
    """P2: History sweep h20."""
    experiment_name: str = "ablation_P2_h20_4d"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 20})


@configclass
class AblationP3Cfg(LowLevelRunnerCfg):
    """P3: History sweep h30 (baseline)."""
    experiment_name: str = "ablation_P3_h30_4d"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_PAINT_BASE)


@configclass
class AblationP4Cfg(LowLevelRunnerCfg):
    """P4: History sweep h40."""
    experiment_name: str = "ablation_P4_h40_4d"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 40})


@configclass
class AblationP5Cfg(LowLevelRunnerCfg):
    """P5: Network size — half (enc=[64,32], f_head=[16,8])."""
    experiment_name: str = "ablation_P5_h30_4d_half"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "enc_hidden_dims": [64, 32], "f_head_dims": [16, 8]})


@configclass
class AblationP6Cfg(LowLevelRunnerCfg):
    """P6: Network size — double (enc=[256,128], f_head=[64,32])."""
    experiment_name: str = "ablation_P6_h30_4d_double"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "enc_hidden_dims": [256, 128], "f_head_dims": [64, 32]})


@configclass
class AblationP7Cfg(LowLevelRunnerCfg):
    """P7: Estimator accuracy reward w=50."""
    experiment_name: str = "ablation_P7_h30_4d_estrew_w50"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_PAINT_BASE)


@configclass
class AblationP8Cfg(LowLevelRunnerCfg):
    """P8: Compliance weight 0.5."""
    experiment_name: str = "ablation_P8_h30_4d_comp_w0p5"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_PAINT_BASE)


@configclass
class AblationP9Cfg(LowLevelRunnerCfg):
    """P9: Compliance weight 1.0."""
    experiment_name: str = "ablation_P9_h30_4d_comp_w1p0"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_PAINT_BASE)


@configclass
class AblationP10Cfg(LowLevelRunnerCfg):
    """P10: Compliance weight 5.0."""
    experiment_name: str = "ablation_P10_h30_4d_comp_w5p0"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_PAINT_BASE)


@configclass
class AblationP11Cfg(LowLevelRunnerCfg):
    """P11: No reconstruction loss."""
    experiment_name: str = "ablation_P11_h30_4d_norec"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "rec_loss_weight": 0.0})


@configclass
class AblationP12Cfg(LowLevelRunnerCfg):
    """P12: TCN encoder."""
    experiment_name: str = "ablation_P12_h30_4d_tcn"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "tcn_mode": "encoder", "tcn_channels": [64, 128], "tcn_kernel_size": 3, "tcn_dilations": [1, 2, 4]})


@configclass
class AblationP13Cfg(LowLevelRunnerCfg):
    """P13: Force dim 2D (fx, fy)."""
    experiment_name: str = "ablation_P13_h30_2d"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(temporal_steps=30, force_dim=2, rec_loss_weight=1.0, enc_hidden_dims=[128, 64], f_head_dims=[32, 16])


@configclass
class AblationP14Cfg(LowLevelRunnerCfg):
    """P14: Force dim xy_yaw (fx, fy, yaw)."""
    experiment_name: str = "ablation_P14_h30_xy_yaw"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(temporal_steps=30, force_dim=3, force_layout="xy_yaw", rec_loss_weight=1.0, enc_hidden_dims=[128, 64], f_head_dims=[32, 16], yaw_loss_weight=3.0)


@configclass
class AblationP15Cfg(LowLevelRunnerCfg):
    """P15: Force dim 4D (same as P3, explicit dimension ablation control)."""
    experiment_name: str = "ablation_P15_h30_4d"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_PAINT_BASE)


@configclass
class AblationP16Cfg(LowLevelRunnerCfg):
    """P16: Force dim 6D (default net)."""
    experiment_name: str = "ablation_P16_h30_6d"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(temporal_steps=30, force_dim=6, rec_loss_weight=1.0, enc_hidden_dims=[128, 64], f_head_dims=[32, 16], torque_angle_loss_weight=3.0, torque_angle_min=0.3, yaw_loss_weight=3.0)


@configclass
class AblationP17Cfg(LowLevelRunnerCfg):
    """P17: Force dim 6D (bigger net)."""
    experiment_name: str = "ablation_P17_h30_6d_big"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(temporal_steps=30, force_dim=6, rec_loss_weight=1.0, enc_hidden_dims=[256, 128], f_head_dims=[64, 32], torque_angle_loss_weight=3.0, torque_angle_min=0.3, yaw_loss_weight=3.0)


@configclass
class AblationP18Cfg(LowLevelRunnerCfg):
    """P18: Payload (randomized 0-4kg mass per episode), estimator mirrors R1.

    Estimator is now 6D wrench (was 4D), bigger net [256,128], TCN preprocessor,
    no reconstruction loss — same architecture as R1/R2/R3 so comparisons are
    apples-to-apples. Env (LowLevelPayloadEnvCfg) already subclasses
    PSeriesWrenchEnvCfg (= R1 env), so no env change needed.
    """
    experiment_name: str = "ablation_P18_h30_6d_big_tcn_norec_payload"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(
        force_dim=6,
        temporal_steps=30,
        enc_hidden_dims=[256, 128],
        f_head_dims=[64, 32],
        rec_loss_weight=0.0,
        angle_loss_weight=3.0,
        torque_angle_loss_weight=3.0,
        torque_angle_min=0.3,
        yaw_loss_weight=3.0,
        tcn_mode="preprocessor",
        tcn_channels=[64, 64],
        tcn_kernel_size=3,
        tcn_dilations=[1, 2],
    )


@configclass
class AblationP19Cfg(LowLevelRunnerCfg):
    """P19: Lower torque 5Nm (for comparison with 10Nm baseline)."""
    experiment_name: str = "ablation_P19_h30_4d_torque5nm"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 5.0
    estimator: dict = _est(**_PAINT_BASE)


@configclass
class AblationP20Cfg(LowLevelRunnerCfg):
    """P20: Default PD gains Kp=25, Kd=0.5 (baseline uses Kp=8, Kd=0.4)."""
    experiment_name: str = "ablation_P20_h30_4d_pd25"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_PAINT_BASE)


@configclass
class AblationP21Cfg(LowLevelRunnerCfg):
    """P21: Trapezoid force profile (ramp up/hold/ramp down/zero cycles)."""
    experiment_name: str = "ablation_P21_h30_4d_trapezoid"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_PAINT_BASE)


@configclass
class AblationP22Cfg(LowLevelRunnerCfg):
    """P22: Stratified force buckets (envs divided into 4 magnitude bands)."""
    experiment_name: str = "ablation_P22_h30_4d_stratified"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_PAINT_BASE)


# ── P23-P34: PAINT trapezoid + short history ablations ───────────────────────

@configclass
class AblationP23Cfg(LowLevelRunnerCfg):
    """P23: PAINT trapezoid profile, h=4."""
    experiment_name: str = "ablation_P23_h4_4d_paint_trap"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 4})


@configclass
class AblationP24Cfg(LowLevelRunnerCfg):
    """P24: PAINT trapezoid profile, h=8 (baseline for this series)."""
    experiment_name: str = "ablation_P24_h8_4d_paint_trap"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP25Cfg(LowLevelRunnerCfg):
    """P25: PAINT trapezoid profile, h=10."""
    experiment_name: str = "ablation_P25_h10_4d_paint_trap"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 10})


@configclass
class AblationP26Cfg(LowLevelRunnerCfg):
    """P26: PAINT trapezoid, h=8, no reconstruction loss."""
    experiment_name: str = "ablation_P26_h8_4d_paint_trap_norec"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8, "rec_loss_weight": 0.0})


@configclass
class AblationP27Cfg(LowLevelRunnerCfg):
    """P27: PAINT trapezoid, h=8, 20% zero-wrench probability."""
    experiment_name: str = "ablation_P27_h8_4d_paint_trap_zero20"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP28Cfg(LowLevelRunnerCfg):
    """P28: Constant profile, h=8 (comparison baseline for trapezoid series)."""
    experiment_name: str = "ablation_P28_h8_4d_constant"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP29Cfg(LowLevelRunnerCfg):
    """P29: Cyclical trapezoid (multi-cycle), h=8."""
    experiment_name: str = "ablation_P29_h8_4d_cyclic_trap"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP30Cfg(LowLevelRunnerCfg):
    """P30: Constant profile, h=4."""
    experiment_name: str = "ablation_P30_h4_4d_constant"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 4})


@configclass
class AblationP31Cfg(LowLevelRunnerCfg):
    """P31: Constant profile, h=8 (paired with P30 for history isolation)."""
    experiment_name: str = "ablation_P31_h8_4d_constant_b"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP32aCfg(LowLevelRunnerCfg):
    """P32a: PAINT trapezoid, h=8, compliance B=30, sigma=0.25."""
    experiment_name: str = "ablation_P32a_h8_4d_paint_trap_comp_b30_s025"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP32bCfg(LowLevelRunnerCfg):
    """P32b: PAINT trapezoid, h=8, compliance B=30, sigma=0.5."""
    experiment_name: str = "ablation_P32b_h8_4d_paint_trap_comp_b30_s05"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP32cCfg(LowLevelRunnerCfg):
    """P32c: PAINT trapezoid, h=8, compliance B=40, sigma=0.25."""
    experiment_name: str = "ablation_P32c_h8_4d_paint_trap_comp_b40_s025"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP32dCfg(LowLevelRunnerCfg):
    """P32d: PAINT trapezoid, h=8, compliance B=40, sigma=0.5."""
    experiment_name: str = "ablation_P32d_h8_4d_paint_trap_comp_b40_s05"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP33Cfg(LowLevelRunnerCfg):
    """P33: PAINT trapezoid, h=8, est accuracy reward (w=50, sigma=1.0)."""
    experiment_name: str = "ablation_P33_h8_4d_paint_trap_estacc"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP34Cfg(LowLevelRunnerCfg):
    """P34: PAINT trapezoid, h=8, compliance(B=30,σ=0.25) + est acc(w=50)."""
    experiment_name: str = "ablation_P34_h8_4d_paint_trap_both_b30_s025"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8})


@configclass
class AblationP35Cfg(LowLevelRunnerCfg):
    """P35: PAINT trapezoid, h=8, est acc (w=50) + TCN preprocessor."""
    experiment_name: str = "ablation_P35_h8_4d_paint_trap_estacc_tcn"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_PAINT_BASE, "temporal_steps": 8,
                               "tcn_mode": "preprocessor",
                               "tcn_channels": [64, 64],
                               "tcn_kernel_size": 3,
                               "tcn_dilations": [1, 2]})


# ── Q-series: bigger net + TCN + no-rec + est_acc, PAINT trapezoid ───────────
# Q1: h=10  Q2: h=30
# All: bigger enc [256,128]/[64,32], TCN pre [64,64] k=3 dil=[1,2],
#      rec_loss=0.0, est_acc w=50, 30N/10Nm, PAINT trapezoid

_Q_BASE = dict(
    force_dim=4, yaw_loss_weight=3.0, rec_loss_weight=0.0,
    enc_hidden_dims=[256, 128], f_head_dims=[64, 32],
    tcn_mode="preprocessor", tcn_channels=[64, 64],
    tcn_kernel_size=3, tcn_dilations=[1, 2],
)


@configclass
class AblationQ1Cfg(LowLevelRunnerCfg):
    """Q1: PAINT trap, h=10, bigger net, TCN pre, no rec loss, est acc w=50."""
    experiment_name: str = "ablation_Q1_h10_4d_paint_trap_big_tcn_norec_estacc"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_Q_BASE, "temporal_steps": 10})


@configclass
class AblationQ2Cfg(LowLevelRunnerCfg):
    """Q2: PAINT trap, h=30, bigger net, TCN pre, no rec loss, est acc w=50."""
    experiment_name: str = "ablation_Q2_h30_4d_paint_trap_big_tcn_norec_estacc"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**{**_Q_BASE, "temporal_steps": 30})


# ── R-series: 6D wrench, persistent wrench, h=30, bigger net, TCN, no rec ────
# R1: no est-acc reward   R2: est-acc reward w=50
# persistent_wrench: fz_scale=0.8, force_free_fraction=0.05, 30N/10Nm
# enc=[256,128], f_head=[64,32], TCN pre [64,64] k=3 dil=[1,2], rec_loss=0.0

_R_BASE = dict(
    force_dim=6,
    temporal_steps=30,
    enc_hidden_dims=[256, 128],
    f_head_dims=[64, 32],
    rec_loss_weight=0.0,
    angle_loss_weight=3.0,
    torque_angle_loss_weight=3.0,
    torque_angle_min=0.3,
    yaw_loss_weight=3.0,
    tcn_mode="preprocessor",
    tcn_channels=[64, 64],
    tcn_kernel_size=3,
    tcn_dilations=[1, 2],
)


@configclass
class AblationR1Cfg(LowLevelRunnerCfg):
    """R1: 6D wrench, h=30, bigger net, TCN pre, no rec loss, no est-acc reward."""
    experiment_name: str = "ablation_R1_h30_6d_big_tcn_norec"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_R_BASE)


@configclass
class AblationR2Cfg(LowLevelRunnerCfg):
    """R2: 6D wrench, h=30, bigger net, TCN pre, no rec loss, est-acc reward w=50."""
    experiment_name: str = "ablation_R2_h30_6d_big_tcn_norec_estacc"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_R_BASE)


@configclass
class AblationR3Cfg(LowLevelRunnerCfg):
    """R3: R1 variant with fixed base mass (no mass randomization)."""
    experiment_name: str = "ablation_R3_h30_6d_big_tcn_norec_nomassrand"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_R_BASE)


@configclass
class AblationR4Cfg(LowLevelRunnerCfg):
    """R4: R1 variant — no mass rand, no push, no observation noise (observability study)."""
    experiment_name: str = "ablation_R4_h30_6d_big_tcn_norec_norand"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_R_BASE)


@configclass
class AblationR5Cfg(LowLevelRunnerCfg):
    """R5: R4 + flat terrain only (strict observability test)."""
    experiment_name: str = "ablation_R5_h30_6d_big_tcn_norec_norand_flat"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_R_BASE)


@configclass
class Ablation6DctrlCfg(LowLevelRunnerCfg):
    """6Dctrl: R1 + commanded roll/pitch/height (UniformVelocityPoseCommand)."""
    experiment_name: str = "ablation_6Dctrl_h30_6d_big_tcn_norec_posecmd"
    class_name: str = "CompliantOnPolicyRunner"
    force_event_term_name: str = "persistent_wrench"
    max_force: float = 30.0
    max_torque: float = 10.0
    estimator: dict = _est(**_R_BASE)


@configclass
class Ablation6DctrlExcludedCfg(Ablation6DctrlCfg):
    """6Dctrl variant: force gate sums all rewards EXCEPT the new pose tracking ones.
    Threshold stays at the R1 baseline (30) — locomotion bar is the same.
    """
    experiment_name: str = "ablation_6Dctrl_curr_excluded"
    force_activation_reward_threshold: float = 30.0
    force_gate_mode: str = "excluded"
    force_gate_excluded_terms: tuple = ("track_roll_pitch", "track_height")


@configclass
class Ablation6DctrlTrackingCfg(Ablation6DctrlCfg):
    """6Dctrl variant: force gate triggers when ALL tracking rewards
    sustain their per-channel % thresholds for `force_gate_dwell_iters` consecutive iterations.
    """
    experiment_name: str = "ablation_6Dctrl_curr_tracking"
    force_gate_mode: str = "tracking"
    force_gate_dwell_iters: int = 50
    force_gate_tracking_thresholds: dict = {
        "track_lin_vel_xy_exp": 0.85,
        "track_ang_vel_z_exp": 0.70,
        "track_roll_pitch": 0.75,
        "track_height": 0.80,
    }


@configclass
class Ablation6DctrlTotal50Cfg(Ablation6DctrlCfg):
    """6Dctrl variant: default total-reward gate, threshold bumped to 50.

    Locomotion + pose tracking plateau at ~49 total reward before forces fire
    on the earlier 6Dctrl runs; 50 triggers activation just after the policy
    has settled. Same estimator/env as the other 6Dctrl ablations — only the
    curriculum gate changes.
    """
    experiment_name: str = "ablation_6Dctrl_curr_total50"
    force_gate_mode: str = "total"
    force_activation_reward_threshold: float = 50.0


@configclass
class Ablation6DctrlTotal50EstAccW10Cfg(Ablation6DctrlTotal50Cfg):
    """6Dctrl-Total50 + force_est_accuracy reward w=10."""
    experiment_name: str = "ablation_6Dctrl_curr_total50_estacc_w10"


@configclass
class Ablation6DctrlTotal50EstAccW25Cfg(Ablation6DctrlTotal50Cfg):
    """6Dctrl-Total50 + force_est_accuracy reward w=25."""
    experiment_name: str = "ablation_6Dctrl_curr_total50_estacc_w25"


@configclass
class Ablation6DctrlTotal50EstAccW50Cfg(Ablation6DctrlTotal50Cfg):
    """6Dctrl-Total50 + force_est_accuracy reward w=50 (matches R2)."""
    experiment_name: str = "ablation_6Dctrl_curr_total50_estacc_w50"


# ── Stand-still ablations ────────────────────────────────────────────────────
# Target: robot does not learn to stop under zero command in the base 6Dctrl
# run. Each variant flips a single knob vs Ablation6DctrlTotal50Cfg.

@configclass
class Ablation6DctrlStandEnv10Cfg(Ablation6DctrlTotal50Cfg):
    """rel_standing_envs 0.02 -> 0.10 (5x more null-command envs)."""
    experiment_name: str = "ablation_6Dctrl_stand_env10"


@configclass
class Ablation6DctrlStandW2Cfg(Ablation6DctrlTotal50Cfg):
    """standing_pose weight -0.5 -> -2.0 (4x stronger null-cmd penalty)."""
    experiment_name: str = "ablation_6Dctrl_stand_w2"


@configclass
class Ablation6DctrlStandBothCfg(Ablation6DctrlTotal50Cfg):
    """Both levers combined."""
    experiment_name: str = "ablation_6Dctrl_stand_both"
