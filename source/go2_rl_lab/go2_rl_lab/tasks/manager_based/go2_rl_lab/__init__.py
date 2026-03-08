# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# velocity envs


gym.register(                                                                                                       
    id="Go2-Testbench-v0",                                                                                
    entry_point="isaaclab.envs:ManagerBasedRLEnv",                                                                  
    disable_env_checker=True,                                                                                       
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_test_bench_env_cfg:UnitreeGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)



gym.register(                                                                                                       
    id="Go2-Velocity-v0",                                                                                
    entry_point="isaaclab.envs:ManagerBasedRLEnv",                                                                  
    disable_env_checker=True,                                                                                       
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_velocity_env_cfg:UnitreeGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(                                                                                                       
    id="Go2-Velocity-Play-v0",                                                                                
    entry_point="isaaclab.envs:ManagerBasedRLEnv",                                                                  
    disable_env_checker=True,                                                                                       
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_velocity_env_cfg:UnitreeGo2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


# estimator envs — same env as testbench, different runner (EstimatorOnPolicyRunner)

gym.register(
    id="Go2-Testbench-Estimator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_test_bench_env_cfg:UnitreeGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_estimator_cfg:EstimatorRunnerCfg",
    },
)


# force estimator env — persistent XY forces + velocity/force estimator

gym.register(
    id="Go2-Force-Estimator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_force_estimator_env_cfg:UnitreeGo2ForceEstimatorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_force_estimator_cfg:ForceEstimatorRunnerCfg",
    },
)


# force-only estimator env — expanded obs + HAC-LOCO style optimizer

gym.register(
    id="Go2-Force-Only-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_force_only_env_cfg:UnitreeGo2ForceOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_force_only_cfg:ForceOnlyRunnerCfg",
    },
)


# force-only estimator env with terrain curriculum

gym.register(
    id="Go2-Force-Only-Terrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_force_only_terrain_env_cfg:UnitreeGo2ForceOnlyTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_force_only_terrain_cfg:ForceOnlyTerrainRunnerCfg",
    },
)


# compliant locomotion — frozen estimator as obs term, standard runner

gym.register(
    id="Go2-Compliant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_compliant_env_cfg:UnitreeGo2CompliantEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_compliant_cfg:CompliantRunnerCfg",
    },
)

gym.register(
    id="Go2-Compliant-NoFoot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_compliant_no_foot_env_cfg:UnitreeGo2CompliantNoFootEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_compliant_no_foot_cfg:CompliantNoFootRunnerCfg",
    },
)

gym.register(
    id="Go2-Compliant-NoFoot-XYZ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_compliant_no_foot_xyz_env_cfg:UnitreeGo2CompliantNoFootXyzEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_compliant_no_foot_xyz_cfg:CompliantNoFootXyzRunnerCfg",
    },
)

# HAC-LOCO stage 2: high-level compliance on top of frozen low-level
# Uses the SAME env as stage 1 (Go2-Compliant-NoFoot-XYZ) but with a different runner
gym.register(
    id="Go2-HAC-LOCO-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_compliant_no_foot_xyz_env_cfg:UnitreeGo2CompliantNoFootXyzEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_hac_loco_stage2_cfg:HacLocoStage2RunnerCfg",
    },
)


# force envs

gym.register(
    id="Go2-Force-v0",                                                                                
    entry_point="isaaclab.envs:ManagerBasedRLEnv",                                                                  
    disable_env_checker=True,                                                                                       
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_force_env_cfg:UnitreeGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)



gym.register(                                                                                                       
    id="Go2-Force-Play-v0",                                                                                
    entry_point="isaaclab.envs:ManagerBasedRLEnv",                                                                  
    disable_env_checker=True,                                                                                       
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_force_env_cfg:UnitreeGo2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

