# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
#
# Naming convention: Go2-Est-<AblationAxis>-<Variant>-v0
# Only report-relevant configurations are registered here.
# Internal class names (AblationP1Cfg, etc.) are preserved for checkpoint
# compatibility but are not exposed as gym task IDs.
##


# ── Base locomotion ───────────────────────────────────────────────────────────

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


# ── History Length (4D, default net, H=30 as baseline) ────────────────────────

gym.register(
    id="Go2-Est-History-H10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP1Cfg",
    },
)

gym.register(
    id="Go2-Est-History-H20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP2Cfg",
    },
)

gym.register(
    id="Go2-Est-History-H30-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP3Cfg",
    },
)

gym.register(
    id="Go2-Est-History-H40-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP4Cfg",
    },
)


# ── TCN Preprocessor (4D, H=40, est-accuracy reward) ─────────────────────────

gym.register(
    id="Go2-Est-TCN-None-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:JSeriesWrenchEstAccW50Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationJ3Cfg",
    },
)

gym.register(
    id="Go2-Est-TCN-Pre-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:JSeriesWrenchEstAccW50Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationJ5Cfg",
    },
)


# ── Network Capacity (4D, H=30) ──────────────────────────────────────────────

gym.register(
    id="Go2-Est-NetSize-Half-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP5Cfg",
    },
)

# Go2-Est-NetSize-Default-v0 is the same config as Go2-Est-History-H30-v0 (P3)
gym.register(
    id="Go2-Est-NetSize-Default-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP3Cfg",
    },
)

gym.register(
    id="Go2-Est-NetSize-Double-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP6Cfg",
    },
)


# ── Reconstruction Loss (4D) ─────────────────────────────────────────────────

# Go2-Est-RecLoss-With-v0 is the same config as Go2-Est-History-H30-v0 (P3)
gym.register(
    id="Go2-Est-RecLoss-With-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP3Cfg",
    },
)

gym.register(
    id="Go2-Est-RecLoss-None-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP11Cfg",
    },
)

gym.register(
    id="Go2-Est-RecLoss-NoneEstAcc-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:JSeriesWrenchEstAccW50Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationJ6Cfg",
    },
)


# ── Wrench Dimensionality (H=30, default net unless noted) ───────────────────

gym.register(
    id="Go2-Est-Dim-2D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP13Cfg",
    },
)

gym.register(
    id="Go2-Est-Dim-3DxyYaw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP14Cfg",
    },
)

# Go2-Est-Dim-4D-v0 is the same config as Go2-Est-History-H30-v0 (P3)
gym.register(
    id="Go2-Est-Dim-4D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP3Cfg",
    },
)

gym.register(
    id="Go2-Est-Dim-6D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP16Cfg",
    },
)

gym.register(
    id="Go2-Est-Dim-6DBig-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP17Cfg",
    },
)


# ── PD Gains (4D, H=30) ──────────────────────────────────────────────────────

gym.register(
    id="Go2-Est-PD-Low-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP3Cfg",
    },
)

gym.register(
    id="Go2-Est-PD-Default-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:DefaultPDPSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP20Cfg",
    },
)


# ── Domain Randomization (6D, big net, TCN, no rec, H=30) ────────────────────

gym.register(
    id="Go2-Est-DomRand-Full-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR1Cfg",
    },
)

gym.register(
    id="Go2-Est-DomRand-NoMass-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesNoMassRandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR3Cfg",
    },
)

gym.register(
    id="Go2-Est-DomRand-None-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesNoRandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR4Cfg",
    },
)


# ── Force Curriculum (6D, big net, TCN, no rec, H=30) ─────────────────────────

# Go2-Est-Curriculum-HardGate-v0 is the same config as Go2-Est-DomRand-Full-v0 (R1)
gym.register(
    id="Go2-Est-Curriculum-HardGate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR1Cfg",
    },
)

gym.register(
    id="Go2-Est-Curriculum-LinearRamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR6Cfg",
    },
)

gym.register(
    id="Go2-Est-Curriculum-Bucketed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR8Cfg",
    },
)


# ── Privileged Observations (6D, big net, TCN, no rec, H=30) ─────────────────

gym.register(
    id="Go2-Est-Priv-All-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR9Cfg",
    },
)

gym.register(
    id="Go2-Est-Priv-AllNoRand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesNoRandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR10Cfg",
    },
)

gym.register(
    id="Go2-Est-Priv-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR11Cfg",
    },
)

gym.register(
    id="Go2-Est-Priv-Contacts-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:PSeriesWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationR12Cfg",
    },
)


# ── Deployed Configuration (6D, TCN, H=30, big net, total gate @ 50) ─────────

gym.register(
    id="Go2-Est-Deploy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_6dctrl_env_cfg:Go2SixDControlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:Ablation6DctrlTotal50Cfg",
    },
)


# ── Payload (deployed arch + payload link, 1–3 kg randomized) ─────────────────

gym.register(
    id="Go2-Est-Payload-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_payload_env_cfg:LowLevelPayloadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:AblationP18Cfg",
    },
)
