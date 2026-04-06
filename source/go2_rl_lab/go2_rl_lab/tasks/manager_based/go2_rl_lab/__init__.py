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
    id="Go2-LowLevel-PACE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lowlevel_env_cfg:LowLevelPaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_lowlevel_cfg:LowLevelPaceRunnerCfg",
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

# ── Force estimator ablation study (A1, A2, B1-B3, C1, C2, E1, E2) ─────────

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

# Group B: 6D wrench estimation — use wrench env
for _id, _cfg in [("B1", "AblationB1Cfg"), ("B2", "AblationB2Cfg"), ("B3", "AblationB3Cfg")]:
    gym.register(
        id=f"Go2-Ablation-{_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ablation_cfg:{_cfg}",
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


# ── Group D: PAINT-style teacher-student distillation (D1, D2) ───────────────
# D1/D2 use compliance reward envs so the teacher genuinely benefits from GT force.
# Stage 1 (teacher): train with CompliantOnPolicyRunner on the same env.
# Stage 2 (student): train with PaintRunner, loading teacher checkpoint.

# D1 teacher: 3D force + compliance reward (same env as E1)
gym.register(
    id="Go2-Ablation-D1-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelComplianceRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_student_cfg:AblationD1TeacherCfg",
    },
)

# D1 student: PAINT distillation on same env
gym.register(
    id="Go2-Ablation-D1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelComplianceRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_student_cfg:AblationD1Cfg",
    },
)

# D2 teacher: 4D wrench + compliance rewards (same env as E2)
gym.register(
    id="Go2-Ablation-D2-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchComplianceRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_student_cfg:AblationD2TeacherCfg",
    },
)

# D2 student: PAINT distillation on same env
gym.register(
    id="Go2-Ablation-D2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_ablation_env_cfgs:LowLevelWrenchComplianceRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_student_cfg:AblationD2Cfg",
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
