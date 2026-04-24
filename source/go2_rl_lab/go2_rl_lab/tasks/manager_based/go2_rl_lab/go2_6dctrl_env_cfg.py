"""6Dctrl env: R1 + commanded roll/pitch/height.

Forks the R1 ablation (PSeriesWrenchEnvCfg) and swaps:
  - CommandsCfg.base_velocity → 6-dim UniformVelocityPoseCommandCfg
  - flat_orientation_l2 → track_roll_pitch_exp
  - base_height_l2 (target=0.34 fixed) → track_height_exp (per-env command)

Mass randomization stays on (R1 behavior).
The estimator picks up the new raw_obs_dim (60) automatically via
CompliantOnPolicyRunner's `policy_obs_dim - force_dim` inference.
"""
from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .go2_ablation_env_cfgs import PSeriesWrenchEnvCfg
from .go2_lowlevel_env_cfg import RewardsCfg
from .mdp.pose_velocity_command import UniformVelocityPoseCommandCfg
from .mdp.rewards import track_height_exp, track_roll_pitch_exp


@configclass
class SixDCtrlCommandsCfg:
    base_velocity = UniformVelocityPoseCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        nominal_height=0.34,
        rel_nominal_roll=0.20,
        rel_nominal_pitch=0.20,
        rel_nominal_height=0.20,
        ranges=UniformVelocityPoseCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
            roll=(-0.25, 0.25),
            pitch=(-0.30, 0.30),
            height=(0.24, 0.38),
        ),
    )


@configclass
class SixDCtrlRewardsCfg(RewardsCfg):
    """Drop the fixed-pose anchors; track the per-env commanded posture instead."""

    flat_orientation_l2 = None
    base_height_l2 = None

    track_roll_pitch = RewTerm(
        func=track_roll_pitch_exp,
        weight=1.0,
        params={
            "std": math.sqrt(0.04),
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_height = RewTerm(
        func=track_height_exp,
        weight=0.5,
        params={
            "std": math.sqrt(0.005),
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class Go2SixDControlEnvCfg(PSeriesWrenchEnvCfg):
    """R1 env + 6Dctrl commands and pose-tracking rewards."""

    commands: SixDCtrlCommandsCfg = SixDCtrlCommandsCfg()
    rewards: SixDCtrlRewardsCfg = SixDCtrlRewardsCfg()
