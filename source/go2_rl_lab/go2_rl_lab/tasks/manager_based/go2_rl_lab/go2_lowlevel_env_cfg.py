"""Self-contained low-level locomotion env config.

This is THE env config for training the deployed low-level policy (V3).
It combines the base compliant-no-foot-XYZ observations/events with
the V3 reward and terrain settings that won the finetuning sweep.

Key settings (must match deployment config):
- Policy obs: 60 dims (57 raw + 3 force estimate XYZ)
- Critic obs: 70 dims (privileged)
- PD gains: Kp=8, Kd=0.4, action_scale=0.5
- Terrain: rough (boxes, random rough, slopes — no stairs)
- Rewards: R2 Enhanced Smoothness + standing pose penalty
- Runner: CompliantOnPolicyRunner (3-phase: walk → forces → linear mapping)
- decimation=4, dt=0.005, control_dt=0.02 (50Hz)

Policy obs layout::

    [0:3]   base_ang_vel                           (noisy)
    [3:6]   projected_gravity                      (noisy)
    [6:9]   velocity_commands
    [9:21]  joint_pos_rel                          (noisy)
    [21:33] joint_vel_rel                          (noisy)
    [33:45] last_action
    [45:57] applied_torque (scale=0.1)             (noisy)
    [57:60] force_estimate (fx, fy, fz from runner)

Critic obs layout::

    [0:3]   base_lin_vel                           <- privileged
    [3:6]   base_ang_vel
    [6:9]   projected_gravity
    [9:12]  velocity_commands
    [12:24] joint_pos_rel
    [24:36] joint_vel_rel
    [36:48] last_action
    [48:60] applied_torque (scale=0.1)
    [60:64] foot_contact_force_norms (scale=0.01)  <- critic only
    [64:67] base_applied_force_xyz                 <- gt for critic (3D)
    [67:70] force_estimate (fx, fy, fz from runner)
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .mdp.events import apply_persistent_xyz_force
from .mdp.observations import (
    ForceEstimateObsTerm,
    applied_torque,
    base_applied_force_xyz,
    foot_contact_force_norms,
)
from .mdp.rewards import (
    compliant_track_lin_vel_xy_exp,
    standing_pose_penalty,
)
from go2_rl_lab.assets.unitree import UNITREE_GO2_LOW_GAIN_CFG, UNITREE_GO2_LOW_GAIN_PACE_CFG, UNITREE_GO2_LOW_GAIN_PACE_APRIL_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


##
# Scene definition
##


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with Go2."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = None
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications — policy 60 dims (no foot contacts, 3D force), critic 70 dims."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations: proprioceptive + dynamics + 3D force estimate (60 dims).

        NO foot contact forces — not available on real robot.
        Force estimate is 3D (fx, fy, fz).
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        applied_torque_obs = ObsTerm(
            func=applied_torque,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot"), "scale": 0.1},
        )
        # Frozen force estimator output: [num_envs, 3] (XYZ)
        force_estimate = ObsTerm(func=ForceEstimateObsTerm)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations (70 dims) — clean + GT XYZ force + force estimate.

        Critic KEEPS foot contact forces (privileged info).
        """

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
        # Ground truth XYZ force for critic (3D)
        base_applied_force_xyz = ObsTerm(
            func=base_applied_force_xyz,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        # Frozen force estimator output (critic also sees the 3D estimate)
        force_estimate = ObsTerm(func=ForceEstimateObsTerm)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Events — persistent XYZ force starts at 0, activated by curriculum."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval — persistent XYZ force (starts at 0, curriculum activates)
    # fz_scale=0.6 means Z force range is 60% of XY range for stability
    persistent_xyz_force = EventTerm(
        func=apply_persistent_xyz_force,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),  # curriculum sets to (0, max_force)
            "fz_scale": 0.6,
        },
    )

    # interval — velocity pushes
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms — R2 Enhanced Smoothness + standing pose (V3 config).

    Built on top of the base compliant rewards with:
    - trunk height penalty (target 0.34m)
    - 1st-order action smoothness
    - explicit torque penalty
    - higher feet air time weight (0.75 vs 0.25)
    - standing pose penalty (conditional on near-zero command)
    """

    # ── Tracking rewards ──────────────────────────────────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=compliant_track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # ── Auxiliary rewards ──────────────────────────────────────────────
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    action_smoothness_2 = RewTerm(func=mdp.action_smoothness_2, weight=-0.01)
    pose_similarity = RewTerm(
        func=mdp.pose_similarity,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearence_dense,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": 0.10,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-1.0,
        params={
            "threshold": 0.20,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    soft_landing = RewTerm(
        func=mdp.soft_landing,
        weight=-1e-3,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    # ── R2 additions: smoothness ──────────────────────────────────────
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={"target_height": 0.34},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)

    # ── Standing pose penalty (V3 addition) ───────────────────────────
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
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Terrain curriculum — progress to harder terrain when walking well."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LowLevelEnvCfg(ManagerBasedRLEnvCfg):
    """Go2 low-level locomotion env — the V3 config that is deployed on the real robot.

    Low PD gains (Kp=8, Kd=0.4), rough terrain (no stairs), R2 + standing pose rewards.
    Trained with CompliantOnPolicyRunner for 3-phase force estimation pipeline.
    """

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # Terrain curriculum
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
            # Remove stairs, keep only mild roughness and gentle slopes
            tg = self.scene.terrain.terrain_generator
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


##
# No-estimator variant (57-dim policy, 67-dim critic)
##


@configclass
class NoEstObservationsCfg:
    """Observations without force estimate — policy 57 dims, critic 67 dims."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive-only policy observations (57 dims). No force estimate."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        applied_torque_obs = ObsTerm(
            func=applied_torque,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot"), "scale": 0.1},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations (67 dims) — GT force but no force estimate."""

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
        base_applied_force_xyz = ObsTerm(
            func=base_applied_force_xyz,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class NoEstEventCfg(EventCfg):
    """Forces active from the start — no estimator gating needed."""

    persistent_xyz_force = EventTerm(
        func=apply_persistent_xyz_force,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 20.0),
            "fz_scale": 0.6,
        },
    )


@configclass
class LowLevelNoEstEnvCfg(LowLevelEnvCfg):
    """V3 locomotion without force estimate in observations.

    Policy obs: 57 dims (proprioceptive only, no force estimate slot).
    Critic obs: 67 dims (privileged + GT force, no force estimate).
    Forces: active from the start at 20N (no estimator gating).

    Train with OnPolicyRunner, then freeze and train estimator on top.
    """

    observations: NoEstObservationsCfg = NoEstObservationsCfg()
    events: NoEstEventCfg = NoEstEventCfg()


@configclass
class LowLevelPaceEnvCfg(LowLevelEnvCfg):
    """Same as LowLevelEnvCfg but with PACE-identified actuator parameters.

    Uses per-joint armature, viscous friction, static friction, encoder bias,
    and delay parameters identified from the real robot.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_PACE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class LowLevelPaceAprilEnvCfg(LowLevelEnvCfg):
    """V3 low-level env with April PACE actuator params (run 26_04_12_23-04-12)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_LOW_GAIN_PACE_APRIL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
