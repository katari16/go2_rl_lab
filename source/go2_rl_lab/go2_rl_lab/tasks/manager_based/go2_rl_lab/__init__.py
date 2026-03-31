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
