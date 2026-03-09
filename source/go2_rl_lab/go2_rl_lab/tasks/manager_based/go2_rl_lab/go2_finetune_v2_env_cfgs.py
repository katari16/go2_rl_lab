"""Low-level locomotion finetuning V2 — 4 variations based on J2 winner.

All use low PD gains (Kp=8, Kd=0.4, action_scale=0.5) with R2 Enhanced Smoothness rewards.

Matrix: 2 reward additions x 2 terrain types = 4 jobs.

Reward additions (on top of R2):
- Standing pose: conditional penalty when cmd ≈ 0 (smooth blend)
- Hip deviation: targeted hip abduction penalty (always active)

Terrain:
- Flat: same as J2 (plane)
- Rough: mild uneven ground (no stairs), with terrain curriculum

V1: R2 + standing pose reward (flat)
V2: R2 + standing pose + hip penalty (flat)
V3: R2 + standing pose reward (rough)
V4: R2 + standing pose + hip penalty (rough)

Force estimator + compliance pipeline preserved (CompliantOnPolicyRunner).
20k iterations from scratch.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .go2_compliant_no_foot_xyz_env_cfg import (
    UnitreeGo2CompliantNoFootXyzEnvCfg,
    CurriculumCfg,
)
from .go2_finetune_env_cfgs import RewardsR2Cfg
from .mdp.rewards import (
    standing_pose_penalty,
    hip_deviation_penalty,
)
from go2_rl_lab.assets.unitree import UNITREE_GO2_LOW_GAIN_CFG


# ═══════════════════════════════════════════════════════════════════════════════
# Reward presets
# ═══════════════════════════════════════════════════════════════════════════════


@configclass
class RewardsV1Cfg(RewardsR2Cfg):
    """R2 + standing pose penalty (conditional on near-zero command)."""

    standing_pose = RewTerm(
        func=standing_pose_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "cmd_deadzone": 0.05,
            "cmd_ramp_end": 0.15,
        },
    )


@configclass
class RewardsV2Cfg(RewardsR2Cfg):
    """R2 + standing pose penalty + hip abduction penalty (always active)."""

    standing_pose = RewTerm(
        func=standing_pose_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "cmd_deadzone": 0.05,
            "cmd_ramp_end": 0.15,
        },
    )
    hip_deviation = RewTerm(
        func=hip_deviation_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Terrain curriculum config for V3/V4
# ═══════════════════════════════════════════════════════════════════════════════


@configclass
class TerrainCurriculumCfg:
    """Terrain curriculum — progress to harder terrain when walking well."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ═══════════════════════════════════════════════════════════════════════════════
# Job configs: Flat terrain (V1, V2)
# ═══════════════════════════════════════════════════════════════════════════════


@configclass
class FinetuneV1EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """V1: Low gains + R2 + standing pose (flat terrain)."""

    rewards: RewardsV1Cfg = RewardsV1Cfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.5


@configclass
class FinetuneV2EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """V2: Low gains + R2 + standing pose + hip penalty (flat terrain)."""

    rewards: RewardsV2Cfg = RewardsV2Cfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Job configs: Rough terrain (V3, V4) — mild uneven ground, no stairs
# ═══════════════════════════════════════════════════════════════════════════════


@configclass
class FinetuneV3EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """V3: Low gains + R2 + standing pose (rough terrain, no stairs)."""

    rewards: RewardsV1Cfg = RewardsV1Cfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.5
        # Enable generated terrain
        self.scene.terrain.terrain_type = "generator"
        # Remove stairs, keep only mild roughness and gentle slopes
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            # Remove stair sub-terrains
            for stair_key in ["pyramid_stairs", "pyramid_stairs_inv"]:
                if stair_key in tg.sub_terrains:
                    del tg.sub_terrains[stair_key]
            # Redistribute proportions to remaining terrains
            # boxes: 0.3, random_rough: 0.4, hf_pyramid_slope: 0.15, hf_pyramid_slope_inv: 0.15
            if "boxes" in tg.sub_terrains:
                tg.sub_terrains["boxes"].proportion = 0.3
                tg.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
            if "random_rough" in tg.sub_terrains:
                tg.sub_terrains["random_rough"].proportion = 0.4
                tg.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
                tg.sub_terrains["random_rough"].noise_step = 0.01
            if "hf_pyramid_slope" in tg.sub_terrains:
                tg.sub_terrains["hf_pyramid_slope"].proportion = 0.15
                tg.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.3)
            if "hf_pyramid_slope_inv" in tg.sub_terrains:
                tg.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.15
                tg.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.3)


@configclass
class FinetuneV4EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """V4: Low gains + R2 + standing pose + hip penalty (rough terrain, no stairs)."""

    rewards: RewardsV2Cfg = RewardsV2Cfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.5
        # Enable generated terrain
        self.scene.terrain.terrain_type = "generator"
        # Remove stairs, keep only mild roughness and gentle slopes
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            for stair_key in ["pyramid_stairs", "pyramid_stairs_inv"]:
                if stair_key in tg.sub_terrains:
                    del tg.sub_terrains[stair_key]
            if "boxes" in tg.sub_terrains:
                tg.sub_terrains["boxes"].proportion = 0.3
                tg.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
            if "random_rough" in tg.sub_terrains:
                tg.sub_terrains["random_rough"].proportion = 0.4
                tg.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
                tg.sub_terrains["random_rough"].noise_step = 0.01
            if "hf_pyramid_slope" in tg.sub_terrains:
                tg.sub_terrains["hf_pyramid_slope"].proportion = 0.15
                tg.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.3)
            if "hf_pyramid_slope_inv" in tg.sub_terrains:
                tg.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.15
                tg.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.3)
