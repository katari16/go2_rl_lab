"""Ablation study env configs — wrench and compliance reward variants.

Variants built on LowLevelEnvCfg:

1. LowLevelWrenchEnvCfg — for B1-B4: 6D wrench event (torque 0-5 Nm) + critic GT
1b. LowLevelWrenchHighTorqueEnvCfg — for B5: same but torque 0-10 Nm
1e. Stage2NoEstEnvCfg / Stage2NoEstWrenchEnvCfg — for S-series frozen-policy runs
    (57-dim policy obs, force/wrench event, placeholder for future compliance rewards)
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
    LowLevelNoEstEnvCfg,
    EventCfg,
    NoEstEventCfg,
    ObservationsCfg,
    RewardsCfg,
)
from .mdp.events import apply_persistent_wrench, apply_trapezoid_wrench
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
    force_estimation_accuracy,
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
            "torque_range": (0.0, 0.0),  # curriculum sets to (0, max_torque)
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
            "torque_range": (0.0, 0.0),  # curriculum sets to (0, max_torque)
        },
    )


@configclass
class LowLevelWrenchHighTorqueEnvCfg(LowLevelWrenchEnvCfg):
    """6D wrench env with torque_range=(0, 10) Nm — for ablation run B5."""

    events: WrenchHighTorqueEventCfg = WrenchHighTorqueEventCfg()


# ── 1e. Stage 2 frozen-policy envs (no-est base, S-series) ──────────────────


@configclass
class Stage2NoEstEnvCfg(LowLevelNoEstEnvCfg):
    """Stage 2 env for frozen-policy estimator training — XYZ force event.

    Inherits LowLevelNoEstEnvCfg (57-dim policy, 67-dim critic) so the frozen
    base policy loads without dimension mismatch. Rewards are inherited from
    the base env — the RewardsCfg slot is kept as a hook for future compliance
    reward ablations. The estimator-only runner skips PPO so rewards are only
    used for episode logging.

    Terrain matches stage 1 (same LowLevelEnvCfg terrain) so the estimator
    generalizes to the same distribution the frozen policy was trained on.
    """

    rewards: RewardsCfg = RewardsCfg()


@configclass
class Stage2NoEstWrenchEnvCfg(LowLevelNoEstEnvCfg):
    """Stage 2 env with wrench event — for S3-S9 (4D/6D/xy_yaw).

    Same 57-dim policy obs as Stage2NoEstEnvCfg, but swaps persistent_xyz_force
    for persistent_wrench so torques are generated for the estimator GT.
    """

    events: WrenchEventCfg = WrenchEventCfg()
    rewards: RewardsCfg = RewardsCfg()


# ── 1c. Trapezoid wrench env (PAINT-style force profile) ────────────────────


@configclass
class TrapezoidWrenchEventCfg(EventCfg):
    """Swaps persistent_xyz_force → trapezoid wrench with stratified buckets."""

    persistent_xyz_force = None  # remove parent's XYZ-only event

    persistent_wrench = EventTerm(
        func=apply_trapezoid_wrench,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # fire every control step
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),  # curriculum sets to (0, max_force)
            "fz_scale": 0.6,
            "torque_range": (0.0, 0.0),  # curriculum sets to (0, max_torque)
            "ramp_s_range": (0.2, 0.8),
            "hold_s_range": (2.0, 5.0),
            "zero_s_range": (0.5, 2.0),
            "zero_prob": 0.02,
            "bucket_fracs": (
                (0.0, 0.0), (0.0, 0.2), (0.2, 0.5), (0.5, 1.0),
            ),
        },
    )


@configclass
class LowLevelWrenchTrapezoidEnvCfg(LowLevelEnvCfg):
    """6D wrench env with PAINT-style trapezoid force profile.

    Changes from LowLevelEnvCfg:
    - Event: apply_trapezoid_wrench (forces + torques, stratified buckets)
    - Critic: 6D wrench GT instead of 3D force GT
    - force_event_term_name must be "persistent_wrench" in runner cfg
    """

    events: TrapezoidWrenchEventCfg = TrapezoidWrenchEventCfg()
    observations: WrenchObservationsCfg = WrenchObservationsCfg()


# ── 1d. Force estimation accuracy reward envs (H7, H8) ─────────────────────


@configclass
class EstAccuracyRewardsCfg(RewardsCfg):
    """Baseline rewards + force estimation accuracy reward."""

    force_est_accuracy = RewTerm(
        func=force_estimation_accuracy,
        weight=0.5,
        params={
            "sigma": 1.0,
            "alpha": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


@configclass
class LowLevelEstAccuracyEnvCfg(LowLevelEnvCfg):
    """Baseline env + force estimation accuracy reward — for H8 (3D)."""

    rewards: EstAccuracyRewardsCfg = EstAccuracyRewardsCfg()


@configclass
class LowLevelWrenchEstAccuracyEnvCfg(LowLevelWrenchEnvCfg):
    """Wrench env + force estimation accuracy reward — for H7 (4D)."""

    rewards: EstAccuracyRewardsCfg = EstAccuracyRewardsCfg()


# ── 1f. J-series: 4D wrench + parameterized estimation accuracy reward ─────


def _est_accuracy_rewards(weight: float, sigma: float):
    """Build a RewardsCfg subclass with the given estimation accuracy params."""
    @configclass
    class _Cfg(RewardsCfg):
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
class JSeriesWrenchEstAccW1Cfg(LowLevelWrenchEnvCfg):
    """4D wrench + est accuracy weight=1, sigma=1.0."""
    rewards = _est_accuracy_rewards(1.0, 1.0)()


@configclass
class JSeriesWrenchEstAccW10Cfg(LowLevelWrenchEnvCfg):
    """4D wrench + est accuracy weight=10, sigma=1.0."""
    rewards = _est_accuracy_rewards(10.0, 1.0)()


@configclass
class JSeriesWrenchEstAccW50Cfg(LowLevelWrenchEnvCfg):
    """4D wrench + est accuracy weight=50, sigma=1.0."""
    rewards = _est_accuracy_rewards(50.0, 1.0)()


@configclass
class JSeriesWrenchEstAccW100Cfg(LowLevelWrenchEnvCfg):
    """4D wrench + est accuracy weight=100, sigma=1.0."""
    rewards = _est_accuracy_rewards(100.0, 1.0)()


@configclass
class JSeriesWrenchEstAccW50S025Cfg(LowLevelWrenchEnvCfg):
    """4D wrench + est accuracy weight=50, sigma=0.25 (sharp)."""
    rewards = _est_accuracy_rewards(50.0, 0.25)()


@configclass
class JSeriesWrenchEstAccW50S4Cfg(LowLevelWrenchEnvCfg):
    """4D wrench + est accuracy weight=50, sigma=4.0 (lenient)."""
    rewards = _est_accuracy_rewards(50.0, 4.0)()


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


# ── 4. DefaultPD wrench trapezoid env (P20) ────────────────────────────────


@configclass
class DefaultPDWrenchTrapezoidEnvCfg(LowLevelWrenchTrapezoidEnvCfg):
    """Same as LowLevelWrenchTrapezoidEnvCfg but with default PD gains (Kp=25, Kd=0.5).

    Used by P20 ablation to test high stiffness vs baseline (Kp=8, Kd=0.4).
    """

    def __post_init__(self):
        super().__post_init__()
        from go2_rl_lab.assets.unitree import UNITREE_GO2_CFG
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ── P-series: PAINT-profile env configs ────────────────────────────────────


def _paint_est_accuracy_rewards(weight: float, sigma: float = 1.0):
    """Build RewardsCfg with estimation accuracy reward for PAINT env."""
    @configclass
    class _Cfg(RewardsCfg):
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


def _paint_compliance_rewards(weight: float):
    """Build RewardsCfg with compliance force tracking for PAINT env."""
    @configclass
    class _Cfg(RewardsCfg):
        compliance_force = RewTerm(
            func=compliance_force_tracking,
            weight=weight,
            params={
                "B_force": 20.0,
                "sigma": 0.25,
                "alpha": 2.0,
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            },
        )
    return _Cfg


@configclass
class PaintEstAccW50EnvCfg(LowLevelWrenchTrapezoidEnvCfg):
    """PAINT trapezoid wrench + est accuracy reward w=50 — for P7."""
    rewards = _paint_est_accuracy_rewards(50.0)()


@configclass
class PaintComplianceW0p5EnvCfg(LowLevelWrenchTrapezoidEnvCfg):
    """PAINT trapezoid wrench + compliance reward w=0.5 — for P8."""
    rewards = _paint_compliance_rewards(0.5)()


@configclass
class PaintComplianceW1p0EnvCfg(LowLevelWrenchTrapezoidEnvCfg):
    """PAINT trapezoid wrench + compliance reward w=1.0 — for P9."""
    rewards = _paint_compliance_rewards(1.0)()


@configclass
class PaintComplianceW5p0EnvCfg(LowLevelWrenchTrapezoidEnvCfg):
    """PAINT trapezoid wrench + compliance reward w=5.0 — for P10."""
    rewards = _paint_compliance_rewards(5.0)()
