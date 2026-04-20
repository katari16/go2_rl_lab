# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# ── Low-level locomotion (self-contained V3 config) ──────────────────────────

gym.register(
    id="Go2-LowLevel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_lowlevel_cfg:LowLevelRunnerCfg",
    },
)

gym.register(
    id="Go2-LowLevel-NoEst-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelNoEstEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_lowlevel_cfg:LowLevelNoEstRunnerCfg",
    },
)

gym.register(
    id="Go2-LowLevel-PACE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelPaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_lowlevel_cfg:LowLevelPaceRunnerCfg",
    },
)

gym.register(
    id="Go2-LowLevel-PACE-April-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelPaceAprilEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_lowlevel_cfg:LowLevelPaceAprilRunnerCfg",
    },
)

gym.register(
    id="Go2-LowLevel-PACE-April14-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelPaceApril14EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_lowlevel_cfg:LowLevelPaceApril14RunnerCfg",
    },
)


# ── Payload variants ───────────────────────────────────────────────────────────

gym.register(
    id="Go2-LowLevel-Payload3D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_payload_env_cfg:LowLevelPayload3DEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_lowlevel_cfg:LowLevelRunnerCfg",
    },
)

gym.register(
    id="Go2-LowLevel-Payload-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_payload_env_cfg:LowLevelPayloadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_lowlevel_cfg:LowLevelRunnerCfg",
    },
)


# ── High-level non-linear compliance (frozen low-level + trainable high-level) ──

gym.register(
    id="Go2-HighLevel-NonLinear-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_highlevel_nonlinear_env_cfg:HighLevelNonLinearEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_hac_loco_stage2_cfg:HacLocoStage2RunnerCfg",
    },
)

# ── Force estimator ablation study (A1, A2, B1-B5, C1, C2, E1, E2) ─────────

# Group A: History length (3D, with reconstruction) — use baseline env
for _id, _cfg in [("A1", "AblationA1Cfg"), ("A2", "AblationA2Cfg")]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# Group B: 6D wrench estimation — use wrench env (5 Nm torque)
for _id, _cfg in [("B1", "AblationB1Cfg"), ("B2", "AblationB2Cfg"), ("B3", "AblationB3Cfg"),
                   ("B4", "AblationB4Cfg")]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# B5: 6D wrench with higher torque range (10 Nm)
gym.register(
    id="Go2-Ablation-B5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchHighTorqueEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationB5Cfg",
    },
)

# Group C: No reconstruction loss — use baseline env
for _id, _cfg in [("C1", "AblationC1Cfg"), ("C2", "AblationC2Cfg")]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# Group E: Compliance rewards
gym.register(
    id="Go2-Ablation-E1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelComplianceRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationE1Cfg",
    },
)

gym.register(
    id="Go2-Ablation-E2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchComplianceRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationE2Cfg",
    },
)


# ── Group D: Teacher-student distillation (D1, D2) ──────────────────────────

gym.register(
    id="Go2-Ablation-D1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_student_cfg:AblationD1Cfg",
    },
)

gym.register(
    id="Go2-Ablation-D2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_student_cfg:AblationD2Cfg",
    },
)


# ── Group H: Batch 2 dimension sweep (50N / 100N) ──────────────────────────

# H1, H9: baseline env (force only, 2D/3D)
for _id, _cfg in [
    ("H1-50N", "AblationH1_50Cfg"), ("H1-100N", "AblationH1_100Cfg"),
    ("H9-50N", "AblationH9_50Cfg"), ("H9-100N", "AblationH9_100Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# H3a/b/c, H5, H6: wrench env (6D/4D)
for _id, _cfg in [
    ("H3a-50N", "AblationH3a_50Cfg"), ("H3a-100N", "AblationH3a_100Cfg"),
    ("H3b-50N", "AblationH3b_50Cfg"), ("H3b-100N", "AblationH3b_100Cfg"),
    ("H3c-50N", "AblationH3c_50Cfg"), ("H3c-100N", "AblationH3c_100Cfg"),
    ("H5-50N", "AblationH5_50Cfg"), ("H5-100N", "AblationH5_100Cfg"),
    ("H6-50N", "AblationH6_50Cfg"), ("H6-100N", "AblationH6_100Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# H7: wrench env + force est accuracy reward (4D)
for _id, _cfg in [("H7-50N", "AblationH7_50Cfg"), ("H7-100N", "AblationH7_100Cfg")]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEstAccuracyEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# H8: baseline env + force est accuracy reward (3D)
for _id, _cfg in [("H8-50N", "AblationH8_50Cfg"), ("H8-100N", "AblationH8_100Cfg")]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelEstAccuracyEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# H12a/b: TCN ablation (6D wrench env, based on H3a)
for _id, _cfg in [
    ("H12a-50N", "AblationH12a_50Cfg"),
    ("H12b-50N", "AblationH12b_50Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# H13a/b: TCN ablation (4D wrench + est accuracy reward env, based on H7)
for _id, _cfg in [
    ("H13a-50N", "AblationH13a_50Cfg"),
    ("H13b-50N", "AblationH13b_50Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEstAccuracyEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )


# H15: H3a rerun with bug fixes (constant force, yaw loss + torque gating fixed)
gym.register(
    id="Go2-Ablation-H15-50N-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationH15_50Cfg",
    },
)

# H14: Trapezoid force profile ablation (6D wrench, based on H3a)
gym.register(
    id="Go2-Ablation-H14-50N-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchTrapezoidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationH14_50Cfg",
    },
)

# H16: 6D, equal loss weights (all 1.0)
# H17: 6D, linear decay temporal weighting
# H18: 6D, TCN preprocessor with detached recon loss
for _id, _cfg in [
    ("H16-50N", "AblationH16_50Cfg"),
    ("H17-50N", "AblationH17_50Cfg"),
    ("H18-50N", "AblationH18_50Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# H19: 3D planar wrench (Fx, Fy, τ_yaw) — uses wrench env for torque generation
gym.register(
    id="Go2-Ablation-H19-50N-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationH19_50Cfg",
    },
)


# ── Group J: 4D wrench + estimation accuracy reward sweep (30N) ──────────────

# J1-J4: weight sweep (1, 10, 50, 100), sigma=1.0
for _id, _cfg, _env_cfg in [
    ("J1", "AblationJ1Cfg", "JSeriesWrenchEstAccW1Cfg"),
    ("J2", "AblationJ2Cfg", "JSeriesWrenchEstAccW10Cfg"),
    ("J3", "AblationJ3Cfg", "JSeriesWrenchEstAccW50Cfg"),
    ("J4", "AblationJ4Cfg", "JSeriesWrenchEstAccW100Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# J5 (TCN), J6 (no rec): same env as J3 (weight=50, sigma=1.0)
for _id, _cfg in [("J5", "AblationJ5Cfg"), ("J6", "AblationJ6Cfg")]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:JSeriesWrenchEstAccW50Cfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# J7: sigma=0.25, J8: sigma=4.0
for _id, _cfg, _env_cfg in [
    ("J7", "AblationJ7Cfg", "JSeriesWrenchEstAccW50S025Cfg"),
    ("J8", "AblationJ8Cfg", "JSeriesWrenchEstAccW50S4Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# ── Group K: B4 re-run (yaw fix) + est accuracy reward (20N, 6D) ─────────────

# K1: same env as B4 (wrench, no est accuracy reward)
gym.register(
    id="Go2-Ablation-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationK1Cfg",
    },
)

# K2, K3: wrench + est accuracy reward (reuse J-series env cfgs)
for _id, _cfg, _env_cfg in [
    ("K2", "AblationK2Cfg", "JSeriesWrenchEstAccW50Cfg"),
    ("K3", "AblationK3Cfg", "JSeriesWrenchEstAccW100Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# ── Group S: Frozen-policy estimator-only training (no-est base) ─────────────

# S1 (2D), S2 (3D): XYZ force env (no torque)
for _id, _cfg in [("S1", "AblationS1Cfg"), ("S2", "AblationS2Cfg")]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:Stage2NoEstEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# S3-S9: wrench env (6D/4D/xy_yaw need torque generation)
for _id, _cfg in [
    ("S3", "AblationS3Cfg"), ("S4", "AblationS4Cfg"),
    ("S5", "AblationS5Cfg"), ("S6", "AblationS6Cfg"),
    ("S7", "AblationS7Cfg"), ("S8", "AblationS8Cfg"),
    ("S9", "AblationS9Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:Stage2NoEstWrenchEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )


# ── Group P: Ablations (P1-P21) — constant wrench (P21 = trapezoid) ─────────

# P1-P6, P11-P17, P19: constant wrench env
for _id, _cfg in [
    ("P1", "AblationP1Cfg"), ("P2", "AblationP2Cfg"),
    ("P3", "AblationP3Cfg"), ("P4", "AblationP4Cfg"), ("P5", "AblationP5Cfg"),
    ("P6", "AblationP6Cfg"), ("P11", "AblationP11Cfg"), ("P12", "AblationP12Cfg"),
    ("P13", "AblationP13Cfg"), ("P14", "AblationP14Cfg"), ("P15", "AblationP15Cfg"),
    ("P16", "AblationP16Cfg"), ("P17", "AblationP17Cfg"),
    ("P19", "AblationP19Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# P7: est accuracy reward w=50
gym.register(
    id="Go2-Ablation-P7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesEstAccW50EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP7Cfg",
    },
)

# P8-P10: compliance reward (varying weights)
for _id, _cfg, _env_cfg in [
    ("P8", "AblationP8Cfg", "PSeriesComplianceW0p5EnvCfg"),
    ("P9", "AblationP9Cfg", "PSeriesComplianceW1p0EnvCfg"),
    ("P10", "AblationP10Cfg", "PSeriesComplianceW5p0EnvCfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# P18: payload with randomized mass (0-4kg)
gym.register(
    id="Go2-Ablation-P18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_payload_env_cfg:LowLevelPayloadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP18Cfg",
    },
)

# P20: Default PD gains Kp=25, Kd=0.5 (baseline Kp=8, Kd=0.4)
gym.register(
    id="Go2-Ablation-P20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:DefaultPDPSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP20Cfg",
    },
)

# P21: Trapezoid force profile ablation
gym.register(
    id="Go2-Ablation-P21-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchTrapezoidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP21Cfg",
    },
)

# P22: Stratified force buckets (4 magnitude bands)
gym.register(
    id="Go2-Ablation-P22-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesStratifiedWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP22Cfg",
    },
)

# ── P23-P34: PAINT trapezoid + short history ablations ──────────────────────

# P23-P26: PAINT trapezoid, varying history / rec loss
for _id, _cfg in [
    ("P23", "AblationP23Cfg"), ("P24", "AblationP24Cfg"),
    ("P25", "AblationP25Cfg"), ("P26", "AblationP26Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesPaintTrapezoidEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# P27: PAINT trapezoid, 20% zero-wrench probability
gym.register(
    id="Go2-Ablation-P27-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesPaintTrapezoidZero20EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP27Cfg",
    },
)

# P28, P30, P31: constant profile, varying history
for _id, _cfg in [
    ("P28", "AblationP28Cfg"), ("P30", "AblationP30Cfg"), ("P31", "AblationP31Cfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# P29: cyclical trapezoid (multi-cycle, our apply_trapezoid_wrench), h=8
gym.register(
    id="Go2-Ablation-P29-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchTrapezoidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP29Cfg",
    },
)

# P32a-P32d: PAINT trapezoid + compliance reward (B_force x sigma sweep)
for _id, _cfg, _env in [
    ("P32a", "AblationP32aCfg", "PSeriesPaintTrapezoidComplianceB30S025EnvCfg"),
    ("P32b", "AblationP32bCfg", "PSeriesPaintTrapezoidComplianceB30S05EnvCfg"),
    ("P32c", "AblationP32cCfg", "PSeriesPaintTrapezoidComplianceB40S025EnvCfg"),
    ("P32d", "AblationP32dCfg", "PSeriesPaintTrapezoidComplianceB40S05EnvCfg"),
]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:{_env}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
        },
    )

# P33: PAINT trapezoid + est accuracy reward (w=50, sigma=1.0)
gym.register(
    id="Go2-Ablation-P33-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesPaintTrapezoidEstAccEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP33Cfg",
    },
)

# P34: PAINT trapezoid + compliance(B=30,σ=0.25) + est acc(w=50)
gym.register(
    id="Go2-Ablation-P34-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesPaintTrapezoidBothRewardsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP34Cfg",
    },
)

# P35: PAINT trapezoid + est acc (w=50) + TCN preprocessor
gym.register(
    id="Go2-Ablation-P35-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesPaintTrapezoidEstAccEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP35Cfg",
    },
)


# ── Q-series: bigger net + TCN + no-rec + est_acc, PAINT trapezoid ──────────
gym.register(
    id="Go2-Ablation-Q1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesPaintTrapezoidEstAccEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationQ1Cfg",
    },
)

gym.register(
    id="Go2-Ablation-Q2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesPaintTrapezoidEstAccEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationQ2Cfg",
    },
)


# ── High-level non-linear sweep: 8 variations (R1-R8) ───────────────────────
# 2x2x2: reward type (penalty/positive) x gravity correction x tracking reward
for _r in range(1, 9):
    gym.register(
        id=f"Go2-HighLevel-NonLinear-R{_r}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_highlevel_nonlinear_env_cfg:HighLevelNonLinearEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_hac_loco_stage2_sweep_cfg:HacLocoStage2R{_r}Cfg",
        },
    )
