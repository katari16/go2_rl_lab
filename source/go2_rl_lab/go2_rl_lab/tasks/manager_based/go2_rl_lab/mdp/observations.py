from __future__ import annotations
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
import torch
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


class ForceEstimateObsTerm(ManagerTermBase):
    """Frozen force estimator as an observation term.

    Loads a pre-trained ForceEstimator from checkpoint, maintains a rolling
    observation history buffer, and returns the estimated XY force as a
    2-dim observation. The estimator is frozen (no gradients).

    The internal 61-dim observation vector matches stage-1 (force-only) layout::

        [ang_vel(3), gravity(3), vel_cmd(3), jpos_rel(12), jvel_rel(12),
         last_action(12), torque*0.1(12), foot_norms*0.01(4)]

    Stores ``env._force_estimate_xy`` for use by the compliant reward.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        ckpt_path: str = getattr(env.cfg, "force_estimator_checkpoint", "")
        if not ckpt_path:
            self._dummy = True
            print("[ForceEstimateObsTerm] No checkpoint — returning zeros.")
            return

        self._dummy = False

        from go2_rl_lab.estimator.force_estimator import ForceEstimator
        from go2_rl_lab.estimator.obs_history_buffer import ObsHistoryBuffer

        obs_dim = 61
        temporal_steps = 20

        # Load checkpoint and extract estimator weights
        ckpt = torch.load(ckpt_path, map_location=env.device, weights_only=False)
        self._estimator = ForceEstimator(
            temporal_steps=temporal_steps,
            num_one_step_obs=obs_dim,
            enc_hidden_dims=[128, 64],
            f_head_dims=[32, 16],
            force_dim=2,
            dec_hidden_dims=[256, 128],
            activation="elu",
        ).to(env.device)

        if "force_estimator_state_dict" in ckpt:
            self._estimator.load_state_dict(ckpt["force_estimator_state_dict"])
            print(f"[ForceEstimateObsTerm] Loaded estimator from: {ckpt_path}")
        else:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} has no 'force_estimator_state_dict'. "
                "Ensure it was saved by ForceOnPolicyRunner."
            )

        # Freeze all parameters
        for param in self._estimator.parameters():
            param.requires_grad = False
        self._estimator.eval()

        # Rolling history buffer
        self._history = ObsHistoryBuffer(env.num_envs, temporal_steps, obs_dim, env.device)

        # Resolve foot body ids on the contact sensor
        sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
        sensor_cfg.resolve(env.scene)
        self._foot_body_ids = sensor_cfg.body_ids

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        if self._dummy:
            # Passthrough mode: return whatever the runner set, or zeros.
            # The runner sets _force_estimate_xy with shape [N, force_dim] (2 or 3).
            f = getattr(env, "_force_estimate_xy", None)
            if f is not None:
                return f.clone()
            # Default to 2D zeros; the runner will override with correct dim before first use
            return torch.zeros(env.num_envs, 2, device=env.device)

        asset: Articulation = env.scene["robot"]
        contact_sensor = env.scene.sensors["contact_forces"]

        # Build 61-dim obs matching stage-1 layout
        ang_vel = asset.data.root_ang_vel_b                                  # [N, 3]
        gravity = asset.data.projected_gravity_b                             # [N, 3]
        vel_cmd = env.command_manager.get_command("base_velocity")           # [N, 3]
        jpos = asset.data.joint_pos - asset.data.default_joint_pos           # [N, 12]
        jvel = asset.data.joint_vel - asset.data.default_joint_vel           # [N, 12]
        last_action = env.action_manager.action                              # [N, 12]
        torque = asset.data.applied_torque * 0.1                             # [N, 12]
        forces = contact_sensor.data.net_forces_w[:, self._foot_body_ids, :]
        foot_norms = torch.norm(forces, dim=-1) * 0.01                      # [N, 4]

        obs = torch.cat([ang_vel, gravity, vel_cmd, jpos, jvel, last_action, torque, foot_norms], dim=1)

        self._history.insert(obs)
        force_hat, _ = self._estimator.get_latent(self._history.get_flattened())

        env._force_estimate_xy = force_hat
        return force_hat.clone()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if not self._dummy and env_ids is not None and len(env_ids) > 0:
            self._history.reset(env_ids)


def foot_contact_force_norms(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, scale: float = 1.0
) -> torch.Tensor:
    """Norm of net contact forces on foot bodies. Shape: [num_envs, num_feet].

    Args:
        scale: Multiplicative scaling factor (e.g. 0.01 to bring raw ~30-200 N into ~0.3-2.0).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return torch.norm(forces, dim=-1) * scale


def applied_torque(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Scaled joint applied torques. Shape: [num_envs, num_joints].

    Args:
        scale: Multiplicative scaling factor (e.g. 0.1 to bring raw ~5-30 Nm into ~0.5-3.0).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids] * scale


def joint_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Scaled joint accelerations. Shape: [num_envs, num_joints].

    Args:
        scale: Multiplicative scaling factor (e.g. 0.01 to bring raw ~100-500 rad/s² into ~1-5).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids] * scale


def joint_vel_scaled(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Scaled joint velocities relative to default. Shape: [num_envs, num_joints].

    Args:
        scale: Multiplicative scaling factor (e.g. 0.1 to bring raw ~±10 rad/s into ~±1).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]) * scale


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

def base_applied_force_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """XY components of persistent external force applied to the base body.

    Reads from the permanent wrench composer buffer (shape: [num_envs, num_bodies, 3]).
    Returns [num_envs, 2] — ground truth for force estimation head.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # composed_force_as_torch: [num_envs, num_bodies, 3] in body frame
    forces = asset.permanent_wrench_composer.composed_force_as_torch
    # Select the target body and take XY
    return forces[:, asset_cfg.body_ids, :2].squeeze(1)


def base_applied_force_xyz(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """XYZ components of persistent external force applied to the base body.

    Reads from the permanent wrench composer buffer (shape: [num_envs, num_bodies, 3]).
    Returns [num_envs, 3] — ground truth for 3D force estimation head.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forces = asset.permanent_wrench_composer.composed_force_as_torch
    return forces[:, asset_cfg.body_ids, :3].squeeze(1)


def base_applied_wrench(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """Full 6D wrench [Fx, Fy, Fz, tau_roll, tau_pitch, tau_yaw] from permanent wrench composer.

    Returns [num_envs, 6] — ground truth for 6D wrench estimation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forces = asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids, :3].squeeze(1)
    torques = asset.permanent_wrench_composer.composed_torque_as_torch[:, asset_cfg.body_ids, :3].squeeze(1)
    return torch.cat([forces, torques], dim=-1)


def base_external_force(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base")) -> torch.Tensor:
    """External forces applied to the base body."""
    asset: Articulation = env.scene[asset_cfg.name]
    external_force_b = asset._external_force_b[:, asset_cfg.body_ids, :].squeeze(1)

    return external_force_b


def base_external_force_visual(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base")
) -> torch.Tensor:
    """External forces applied to the base body."""
    from isaaclab.utils.math import quat_apply, quat_from_matrix
    import torch
    
    asset: Articulation = env.scene[asset_cfg.name]
    external_force_b = asset._external_force_b[:, asset_cfg.body_ids, :]
    forces = external_force_b.squeeze(1)

    force_norms = torch.norm(forces, dim=1)
    env.extras["force_magnitude_mean"] = force_norms.mean()
    env.extras["force_magnitude_max"] = force_norms.max()
    env.extras["num_envs_with_force"] = (force_norms > 0.1).sum().float()

    # Create markers on first call
    if not hasattr(env, '_force_markers'):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/ForceMarkers",
            markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        env._force_markers = VisualizationMarkers(marker_cfg)
        print("[INFO] Force visualization markers created")
    
    # Visualize forces
    n_vis = min(32, env.num_envs)
    base_pos = asset.data.root_pos_w[:n_vis]
    base_quat = asset.data.root_quat_w[:n_vis]
    
    # Transform forces to world frame
    forces_world = quat_apply(base_quat, forces[:n_vis])
    
    # Calculate arrow orientation (arrow points in force direction)
    # Arrow asset points along +X axis, so we need rotation from X to force direction
    force_orientations = torch.zeros(n_vis, 4, device=env.device)
    for i in range(n_vis):
        force_norm = torch.norm(forces_world[i])
        if force_norm > 0.1:  # Only orient if force exists
            force_dir = forces_world[i] / force_norm  # Normalize
            
            # Create rotation matrix that aligns X-axis with force direction
            # Simple approach: use force as new X-axis
            x_axis = force_dir
            # Choose arbitrary Y-axis perpendicular to X
            if abs(force_dir[2]) < 0.9:
                up = torch.tensor([0.0, 0.0, 1.0], device=env.device)
            else:
                up = torch.tensor([1.0, 0.0, 0.0], device=env.device)
            z_axis = torch.cross(x_axis, up)
            z_axis = z_axis / torch.norm(z_axis)
            y_axis = torch.cross(z_axis, x_axis)
            
            # Build rotation matrix
            rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=1)
            force_orientations[i] = quat_from_matrix(rot_mat)
        # else:
        #     force_orientations[i] = base_quat[i]  # No force, use base orientation
    
    env._force_markers.visualize(base_pos, force_orientations)
    
    return forces