"""6Dctrl env: R1 + commanded roll/pitch/height.

Forks the R1 ablation (PSeriesWrenchEnvCfg) and swaps:
  - CommandsCfg.base_velocity → 6-dim UniformVelocityPoseCommandCfg
  - flat_orientation_l2 → track_roll_pitch_exp
  - base_height_l2 (target=0.34 fixed) → track_height_exp (per-env command)
  - joint_torques_l2 weight: -1e-4 → -1e-3 (10x; forces low-torque postures
    so commanded crouches / tilts are achieved by reconfiguring the skeleton
    rather than fighting gravity. Reported by an mjlab user to also smooth
    actions and improve velocity tracking on the same task.)

Command ranges widened to expose more of the posture space:
  - height: [0.18, 0.38] (was [0.24, 0.38])
  - roll:   [-0.35, 0.35] (was [-0.25, 0.25])
  - pitch:  [-0.35, 0.35] (was [-0.30, 0.30])

Reward tuning:
  - track_height weight 0.5 → 1.0 (matches track_roll_pitch).
  - track_height std sqrt(0.005)=0.07m → sqrt(0.02)=0.14m so the gradient
    stays informative across the wider 0.20 m range.

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
from .mdp.rewards import force_estimation_accuracy, track_height_exp, track_roll_pitch_exp


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
            roll=(-0.35, 0.35),
            pitch=(-0.35, 0.35),
            height=(0.18, 0.38),
        ),
    )


@configclass
class SixDCtrlRewardsCfg(RewardsCfg):
    """Drop the fixed-pose anchors; track the per-env commanded posture instead.
    Stronger torque penalty (10x base) so commanded postures are achieved by
    skeletal reconfiguration, not by fighting gravity with stiff joints.
    """

    flat_orientation_l2 = None
    base_height_l2 = None

    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3)

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
        weight=1.0,
        params={
            "std": math.sqrt(0.02),
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class Go2SixDControlEnvCfg(PSeriesWrenchEnvCfg):
    """R1 env + 6Dctrl commands and pose-tracking rewards."""

    commands: SixDCtrlCommandsCfg = SixDCtrlCommandsCfg()
    rewards: SixDCtrlRewardsCfg = SixDCtrlRewardsCfg()


def _sixdctrl_est_acc_rewards(weight: float, sigma: float = 1.0):
    """Build a SixDCtrlRewardsCfg subclass with force_est_accuracy added."""
    @configclass
    class _Cfg(SixDCtrlRewardsCfg):
        force_est_accuracy = RewTerm(
            func=force_estimation_accuracy,
            weight=weight,
            params={
                "sigma": sigma,
                "alpha": 2.0,
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            },
        )
    return _Cfg


@configclass
class Go2SixDControlEstAccW10EnvCfg(Go2SixDControlEnvCfg):
    """6Dctrl + force_est_accuracy reward (w=10, sigma=1.0)."""
    rewards = _sixdctrl_est_acc_rewards(10.0)()


@configclass
class Go2SixDControlEstAccW25EnvCfg(Go2SixDControlEnvCfg):
    """6Dctrl + force_est_accuracy reward (w=25, sigma=1.0)."""
    rewards = _sixdctrl_est_acc_rewards(25.0)()


@configclass
class Go2SixDControlEstAccW50EnvCfg(Go2SixDControlEnvCfg):
    """6Dctrl + force_est_accuracy reward (w=50, sigma=1.0) — matches R2."""
    rewards = _sixdctrl_est_acc_rewards(50.0)()


# ── Stand-still ablations ────────────────────────────────────────────────────
# Address the 6Dctrl-Total50 finding that the robot never stops moving under
# zero command. Three variants that should be smoke-testable individually.

@configclass
class Go2SixDControlStandEnv10EnvCfg(Go2SixDControlEnvCfg):
    """Bump rel_standing_envs 0.02 -> 0.10 so 10% of envs resample to a
    full null command (vel + pose all zero/nominal). More training data
    for the stand-still case."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.10


@configclass
class Go2SixDControlStandW2EnvCfg(Go2SixDControlEnvCfg):
    """Bump standing_pose penalty -0.5 -> -2.0 so near-zero-velocity envs
    actually feel pressure to hold the default stance."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.standing_pose.weight = -2.0


@configclass
class Go2SixDControlStandBothEnvCfg(Go2SixDControlEnvCfg):
    """Both levers: 10% null envs + 4x stronger standing_pose."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.rewards.standing_pose.weight = -2.0
