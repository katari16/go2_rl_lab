"""Low-level locomotion finetuning experiment configs (8 variations).

Matrix: 2 gain presets x 4 reward presets = 8 jobs.

Gains:
- Low: Kp=8, Kd=0.4 (uniform), action_scale=0.5
- Baseline: Kp=25, Kd=0.5 (current), action_scale=0.25

Rewards:
- R1: Baseline (current rewards, no changes)
- R2: Enhanced Smoothness (+height, +action_rate_l2, +torque_l2, feet_air_time 0.75)
- R3: Active Gait (+height, +air_time_var, feet_air_time 1.0, positive clearance, relaxed orientation)
- R4: Deep Compliant Style (base_pose_penalty, high torque, foot_height_sparse, high energy pen)

All configs inherit from the stage-1 compliant env (same obs space, 60-dim policy / 70-dim critic).
No observation changes — only gains, action scale, and reward terms differ.
"""

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .go2_compliant_no_foot_xyz_env_cfg import (
    UnitreeGo2CompliantNoFootXyzEnvCfg,
    ActionsCfg,
    RewardsCfg,
)
from .mdp.rewards import (
    air_time_variance_penalty,
    base_pose_penalty,
    compliant_track_lin_vel_xy_exp,
    foot_clearance_reward,
    foot_height_sparse,
)
from go2_rl_lab.assets.unitree import UNITREE_GO2_CFG, UNITREE_GO2_LOW_GAIN_CFG


# ═══════════════════════════════════════════════════════════════════════════════
# Reward presets
# ═══════════════════════════════════════════════════════════════════════════════


@configclass
class RewardsR2Cfg(RewardsCfg):
    """R2: Enhanced Smoothness — adds height, action_rate_l2, torque_l2, higher air time."""

    # ADD: trunk height penalty
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={"target_height": 0.34},
    )
    # ADD: 1st-order action smoothness
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # ADD: explicit torque penalty
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    # CHANGE: feet_air_time weight 0.25 -> 0.75
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )


@configclass
class RewardsR3Cfg(RewardsCfg):
    """R3: Active Gait — positive clearance reward, air time variance, relaxed orientation."""

    # ADD: trunk height penalty
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={"target_height": 0.34},
    )
    # ADD: 1st-order action smoothness
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # ADD: penalize uneven stride timing
    air_time_variance = RewTerm(
        func=air_time_variance_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    # CHANGE: feet_air_time weight 0.25 -> 1.0
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # REPLACE: feet_clearance penalty -> positive clearance reward
    feet_clearance = RewTerm(
        func=foot_clearance_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": 0.10,
            "std": 0.02,
            "tanh_mult": 5.0,
        },
    )
    # CHANGE: relax flat orientation -2.5 -> -1.5
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    # CHANGE: relax pose similarity -0.1 -> -0.05
    pose_similarity = RewTerm(
        func=mdp.pose_similarity,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # CHANGE: relax feet_too_near -1.0 -> -0.5
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-0.5,
        params={
            "threshold": 0.20,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )


@configclass
class RewardsR4Cfg(RewardsCfg):
    """R4: Deep Compliant Style — matches paper reward structure more closely."""

    # CHANGE: tracking weight 1.5 -> 0.8
    track_lin_vel_xy_exp = RewTerm(
        func=compliant_track_lin_vel_xy_exp,
        weight=0.8,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # ADD: 1st-order action smoothness
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # ADD: base pose penalty (roll² + pitch² + 10·height_err²) replaces flat_orientation
    base_pose = RewTerm(
        func=base_pose_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "desired_height": 0.34},
    )
    # ADD: high torque penalty (paper uses -0.0015)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-3)
    # DISABLE: flat_orientation_l2 (subsumed by base_pose)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # CHANGE: energy weight -2e-5 -> -1.5e-2 (paper uses -0.015)
    energy = RewTerm(func=mdp.energy, weight=-1.5e-2)
    # REPLACE: feet_clearance with foot_height_sparse (paper r_h)
    feet_clearance = RewTerm(
        func=foot_height_sparse,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_height": 0.10,
        },
    )
    # CHANGE: feet_air_time weight 0.25 -> 0.5
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Job configs: Low gains (Kp=8, action_scale=0.5)
# ═══════════════════════════════════════════════════════════════════════════════


@configclass
class FinetuneJ1EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """Job 1: Low gains + R1 Baseline rewards."""

    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.5


@configclass
class FinetuneJ2EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """Job 2: Low gains + R2 Enhanced Smoothness."""

    rewards: RewardsR2Cfg = RewardsR2Cfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.5


@configclass
class FinetuneJ3EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """Job 3: Low gains + R3 Active Gait."""

    rewards: RewardsR3Cfg = RewardsR3Cfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.5


@configclass
class FinetuneJ4EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """Job 4: Low gains + R4 Deep Compliant Style."""

    rewards: RewardsR4Cfg = RewardsR4Cfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Job configs: Baseline gains (Kp=25, action_scale=0.25)
# ═══════════════════════════════════════════════════════════════════════════════


@configclass
class FinetuneJ5EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """Job 5: Baseline gains + R1 Baseline rewards (control — retrain longer)."""

    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # No changes — same as original, just train for more iterations


@configclass
class FinetuneJ6EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """Job 6: Baseline gains + R2 Enhanced Smoothness."""

    rewards: RewardsR2Cfg = RewardsR2Cfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class FinetuneJ7EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """Job 7: Baseline gains + R3 Active Gait."""

    rewards: RewardsR3Cfg = RewardsR3Cfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class FinetuneJ8EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """Job 8: Baseline gains + R4 Deep Compliant Style."""

    rewards: RewardsR4Cfg = RewardsR4Cfg()

    def __post_init__(self):
        super().__post_init__()
