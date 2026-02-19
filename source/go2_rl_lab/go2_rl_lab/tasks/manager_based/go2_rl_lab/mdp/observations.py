from __future__ import annotations
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import torch
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_contact_force_norms(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Norm of net contact forces on foot bodies. Shape: [num_envs, num_feet]."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return torch.norm(forces, dim=-1)


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