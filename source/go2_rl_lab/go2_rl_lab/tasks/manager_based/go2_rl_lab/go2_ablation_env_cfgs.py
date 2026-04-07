"""Ablation study env configs — wrench and compliance reward variants.

Variants built on LowLevelEnvCfg:

1. LowLevelWrenchEnvCfg — for B1-B4: 6D wrench event (torque 0-5 Nm) + critic GT
1b. LowLevelWrenchHighTorqueEnvCfg — for B5: same but torque 0-10 Nm
2. LowLevelComplianceRewardEnvCfg — for E1: baseline + compliance_force_tracking
3. LowLevelWrenchComplianceRewardEnvCfg — for E2: wrench + both compliance rewards
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .go2_lowlevel_env_cfg import (
    LowLevelEnvCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
)
from .mdp.events import apply_persistent_wrench
from .mdp.observations import (
    ForceEstimateObsTerm,
    applied_torque,
    base_applied_wrench,
    foot_contact_force_norms,
)
from .mdp.rewards import (
    compliance_force_tracking,
    compliance_torque_tracking,
    compliant_track_lin_vel_xy_exp,
    standing_pose_penalty,
)


# ── 1. Wrench env (B1, B2, B3) ──────────────────────────────────────────────


@configclass
class WrenchEventCfg(EventCfg):
    """Swaps persistent_xyz_force → apply_persistent_wrench with torque."""

    persistent_xyz_force = None  # remove parent's XYZ-only event

    persistent_wrench = EventTerm(
        func=apply_persistent_wrench,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),  # curriculum sets to (0, max_force)
            "fz_scale": 0.6,
            "torque_range": (0.0, 5.0),
        },
    )


@configclass
class WrenchObservationsCfg(ObservationsCfg):
    """Same policy obs, but critic gets 6D wrench GT instead of 3D force."""

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations with 6D wrench GT (73 dims)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100, 100))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        applied_torque_obs = ObsTerm(
            func=applied_torque,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot"), "scale": 0.1},
        )
        foot_contact_forces = ObsTerm(
            func=foot_contact_force_norms,
            clip=(-100, 100),
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "scale": 0.01},
        )
        # 6D wrench GT for critic (replaces 3D force)
        base_applied_wrench = ObsTerm(
            func=base_applied_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        force_estimate = ObsTerm(func=ForceEstimateObsTerm)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class LowLevelWrenchEnvCfg(LowLevelEnvCfg):
    """6D wrench env — for ablation runs B1, B2, B3.

    Changes from LowLevelEnvCfg:
    - Event: apply_persistent_wrench (forces + torques)
    - Critic: 6D wrench GT instead of 3D force GT
    - force_event_term_name must be "persistent_wrench" in runner cfg
    """

    events: WrenchEventCfg = WrenchEventCfg()
    observations: WrenchObservationsCfg = WrenchObservationsCfg()


# ── 1b. Wrench env with higher torque range (B5) ───────────────────────────


@configclass
class WrenchHighTorqueEventCfg(EventCfg):
    """Like WrenchEventCfg but with torque_range up to 10 Nm."""

    persistent_xyz_force = None

    persistent_wrench = EventTerm(
        func=apply_persistent_wrench,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "fz_scale": 0.6,
            "torque_range": (0.0, 10.0),
        },
    )


@configclass
class LowLevelWrenchHighTorqueEnvCfg(LowLevelWrenchEnvCfg):
    """6D wrench env with torque_range=(0, 10) Nm — for ablation run B5."""

    events: WrenchHighTorqueEventCfg = WrenchHighTorqueEventCfg()


# ── 2. Compliance reward env (E1) ───────────────────────────────────────────


@configclass
class ComplianceRewardsCfg(RewardsCfg):
    """Baseline rewards + compliance_force_tracking."""

    compliance_force = RewTerm(
        func=compliance_force_tracking,
        weight=0.5,
        params={
            "B_force": 20.0,
            "sigma": 0.25,
            "alpha": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


@configclass
class LowLevelComplianceRewardEnvCfg(LowLevelEnvCfg):
    """Baseline env + compliance force tracking reward — for ablation run E1."""

    rewards: ComplianceRewardsCfg = ComplianceRewardsCfg()


# ── 3. Wrench + both compliance rewards env (E2) ────────────────────────────


@configclass
class WrenchComplianceRewardsCfg(RewardsCfg):
    """Baseline rewards + force AND torque compliance tracking."""

    compliance_force = RewTerm(
        func=compliance_force_tracking,
        weight=0.5,
        params={
            "B_force": 20.0,
            "sigma": 0.25,
            "alpha": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )

    compliance_torque = RewTerm(
        func=compliance_torque_tracking,
        weight=0.5,
        params={
            "B_torque": 10.0,
            "sigma": 0.25,
            "alpha": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


@configclass
class LowLevelWrenchComplianceRewardEnvCfg(LowLevelEnvCfg):
    """Wrench env + both compliance rewards — for ablation run E2.

    E2 uses force_dim=4 (Fx,Fy,Fz,τ_yaw) estimator, but the env still
    applies full 6D wrench. Only τ_yaw maps to velocity commands.
    """

    events: WrenchEventCfg = WrenchEventCfg()
    observations: WrenchObservationsCfg = WrenchObservationsCfg()
    rewards: WrenchComplianceRewardsCfg = WrenchComplianceRewardsCfg()
