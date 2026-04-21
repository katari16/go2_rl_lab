"""Static force evaluation — standing robot with persistent XY forces.

Spawns a Go2 standing on flat ground with zero velocity commands and persistent
XY body forces that re-randomize every 1-3 seconds.

Supports both runner types:
  - ForceOnPolicyRunner   (Go2-Force-Only-v0)
  - CompliantOnPolicyRunner (Go2-Compliant-v0)

4-panel plot (saved next to checkpoint):
  Panel 1 (RED):    Applied force — GT (solid) vs estimated (dashed), Fx & Fy
  Panel 2 (BLUE):   Base velocity — vx & vy (body frame)
  Panel 3 (GREEN):  Normal velocity command — cmd_vx & cmd_vy
  Panel 4 (YELLOW): Adjusted velocity command — v* = v_cmd + k * F_hat

Arrows above the robot (GT always shown, others via flags):
  RED    = ground-truth force        (always)
  BLUE   = NN-estimated force        (--show_est)
  GREEN  = velocity command          (--show_cmd)
  YELLOW = adjusted velocity command (--show_adj)

Usage examples:
    # ForceOnPolicyRunner:
    python scripts/rsl_rl/static_eval.py --task Go2-Force-Only-v0 --num_envs 1

    # CompliantOnPolicyRunner:
    python scripts/rsl_rl/static_eval.py --task Go2-Compliant-v0 --num_envs 1 \
        --checkpoint logs/rsl_rl/go2_compliant/2026-XX-XX/model_XXXX.pt

    # Custom force range + compliance:
    python scripts/rsl_rl/static_eval.py --task Go2-Compliant-v0 \
        --force_min 10.0 --force_max 30.0 --compliance_k 0.02 --real-time
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Static force estimation evaluation with arrow visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1).")
parser.add_argument("--task", type=str, default=None, help="Task name (e.g. Go2-Force-Only-v0 or Go2-Compliant-v0).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--force_min", type=float, default=10.0, help="Minimum force magnitude per axis (N).")
parser.add_argument("--force_max", type=float, default=20.0, help="Maximum force magnitude per axis (N).")
parser.add_argument("--duration", type=float, default=20.0, help="Duration to run in seconds (then save plots and exit).")
parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA smoothing factor for filtered estimate (0=full smooth, 1=no smooth).")
parser.add_argument("--compliance_k", type=float, default=0.0, help="Compliance gain k for v*=v+k*F (0=disabled).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--show_est", action="store_true", default=False, help="Show estimated force arrow (blue).")
parser.add_argument("--show_cmd", action="store_true", default=False, help="Show velocity command arrow (green).")
parser.add_argument("--show_adj", action="store_true", default=False, help="Show adjusted velocity command arrow (yellow).")
parser.add_argument("--flat", action="store_true", default=False, help="Use flat ground instead of training terrain.")
parser.add_argument("--torque_min", type=float, default=3.0, help="Minimum torque magnitude per axis (Nm).")
parser.add_argument("--torque_max", type=float, default=10.0, help="Maximum torque magnitude per axis (Nm).")
parser.add_argument("--torque_only", action="store_true", default=False, help="Apply only torques (no forces).")
parser.add_argument("--compliance_k_yaw", type=float, default=0.0, help="Yaw compliance gain for wz*=wz+k*tau_yaw (0=off).")
parser.add_argument("--show_torque", action="store_true", default=False, help="Show yaw torque as dual tangent arrows (magenta).")
parser.add_argument("--trapezoid_cycle", action="store_true", default=False, help="Multi-cycle trapezoid force profile (ramp/hold/zero repeating within episode).")
parser.add_argument("--trapezoid_paint", action="store_true", default=False, help="PAINT-style trapezoid: one ramp-up/hold/ramp-down per period, Cartesian per-axis sampling.")
parser.add_argument("--paint_period", type=float, default=10.0, help="Period duration (s) for --trapezoid_paint (default: 10.0).")
parser.add_argument("--follow_robot", action="store_true", default=False, help="Camera follows env 0 robot base each step.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_from_matrix
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401


# ── Arrow helpers ─────────────────────────────────────────────────────────────


def _create_arrow_markers(prim_path: str, color: tuple) -> VisualizationMarkers:
    """Create coloured arrow markers for force visualization."""
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.5, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _force_to_quat(force_world: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert world-frame force vectors [N,3] to arrow quaternions [N,4]."""
    n = force_world.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0  # identity (w,x,y,z)

    for i in range(n):
        mag = force_world[i].norm()
        if mag < 0.1:
            continue
        x_ax = force_world[i] / mag
        up = torch.tensor(
            [0.0, 0.0, 1.0] if abs(x_ax[2].item()) < 0.9 else [1.0, 0.0, 0.0],
            device=device,
        )
        z_ax = torch.cross(x_ax, up)
        z_ax = z_ax / z_ax.norm()
        y_ax = torch.cross(z_ax, x_ax)
        rot_mat = torch.stack([x_ax, y_ax, z_ax], dim=1)  # [3, 3]
        quats[i] = quat_from_matrix(rot_mat)
    return quats


# ── Plotting ─────────────────────────────────────────────────────────────────


def generate_plots(
    log: dict,
    force_min: float,
    force_max: float,
    checkpoint_path: str,
    ema_alpha: float,
    compliance_k: float,
) -> None:
    """Generate multi-panel evaluation plot and save next to checkpoint.

    One panel per force/torque component: Fx, Fy, Fz (if 3D+),
    τ_roll, τ_pitch (if 6D), τ_yaw (if 4D+).
    Velocity panels are omitted.
    """
    import matplotlib.pyplot as plt
    from datetime import datetime

    has_fz = "gt_force_z" in log and len(log["gt_force_z"]) > 0
    has_torque_yaw = "gt_torque_yaw" in log and len(log["gt_torque_yaw"]) > 0
    has_torque_rp = "gt_torque_roll" in log and len(log["gt_torque_roll"]) > 0

    t = np.array(log["time_s"])
    gt_fx = np.array(log["gt_force_x"])
    gt_fy = np.array(log["gt_force_y"])
    est_fx = np.array(log["est_force_x"])
    est_fy = np.array(log["est_force_y"])
    rerandom = log["rerandom_steps"]

    # Count panels: Fx, Fy always; Fz if 3D+; τ_roll, τ_pitch if 6D; τ_yaw if 4D+
    n_panels = (
        2
        + (1 if has_fz else 0)
        + (2 if has_torque_rp else 0)
        + (1 if has_torque_yaw else 0)
    )
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    k_str = f"k={compliance_k:.4f}" if compliance_k > 0 else "k=0 (disabled)"
    fig.suptitle(
        f"Static Force Eval — [{force_min:.0f}, {force_max:.0f}] N/axis — EMA \u03b1={ema_alpha} — {k_str}"
        + (" (6D wrench)" if has_torque_rp else " (4D)" if has_torque_yaw else " (XYZ)" if has_fz else " (XY)"),
        fontsize=14,
        fontweight="bold",
    )

    for ax in axes:
        for rs in rerandom:
            if rs < len(t):
                ax.axvline(t[rs], color="gray", alpha=0.3, linewidth=0.5, linestyle="--")

    panel = 0

    # ── Panel: Fx ────────────────────────────────────────────────────────
    ax = axes[panel]
    ax.plot(t, gt_fx, color="tab:red", linewidth=1.2, alpha=0.9, label="GT Fx")
    ax.plot(t, est_fx, color="tab:red", linewidth=0.8, alpha=0.5, linestyle="--", label="Est Fx")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Fx: GT (solid) vs Estimated (dashed)", fontsize=11)
    panel += 1

    # ── Panel: Fy ────────────────────────────────────────────────────────
    ax = axes[panel]
    ax.plot(t, gt_fy, color="darkred", linewidth=1.2, alpha=0.9, label="GT Fy")
    ax.plot(t, est_fy, color="darkred", linewidth=0.8, alpha=0.5, linestyle="--", label="Est Fy")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Fy: GT (solid) vs Estimated (dashed)", fontsize=11)
    panel += 1

    # ── Panel: Fz (only if 3D+) ──────────────────────────────────────────
    if has_fz:
        gt_fz = np.array(log["gt_force_z"])
        est_fz = np.array(log["est_force_z"])
        ax = axes[panel]
        ax.plot(t, gt_fz, color="tab:purple", linewidth=1.2, alpha=0.9, label="GT Fz")
        ax.plot(t, est_fz, color="tab:purple", linewidth=0.8, alpha=0.5, linestyle="--", label="Est Fz")
        ax.set_ylabel("Force (N)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title("Fz: GT (solid) vs Estimated (dashed)", fontsize=11)
        panel += 1

    # ── Panel: τ_roll (only if 6D) ───────────────────────────────────────
    if has_torque_rp:
        gt_troll  = np.array(log["gt_torque_roll"])
        est_troll = np.array(log["est_torque_roll"])
        ax = axes[panel]
        ax.plot(t, gt_troll,  color="tab:orange", linewidth=1.2, alpha=0.9, label="GT τ_roll")
        ax.plot(t, est_troll, color="tab:orange", linewidth=0.8, alpha=0.5, linestyle="--", label="Est τ_roll")
        ax.set_ylabel("Torque (Nm)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title("τ_roll: GT (solid) vs Estimated (dashed)", fontsize=11)
        panel += 1

        # ── Panel: τ_pitch (only if 6D) ──────────────────────────────────
        gt_tpitch  = np.array(log["gt_torque_pitch"])
        est_tpitch = np.array(log["est_torque_pitch"])
        ax = axes[panel]
        ax.plot(t, gt_tpitch,  color="darkorange", linewidth=1.2, alpha=0.9, label="GT τ_pitch")
        ax.plot(t, est_tpitch, color="darkorange", linewidth=0.8, alpha=0.5, linestyle="--", label="Est τ_pitch")
        ax.set_ylabel("Torque (Nm)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title("τ_pitch: GT (solid) vs Estimated (dashed)", fontsize=11)
        panel += 1

    # ── Panel: τ_yaw (only if 4D+) ───────────────────────────────────────
    if has_torque_yaw:
        gt_tyaw  = np.array(log["gt_torque_yaw"])
        est_tyaw = np.array(log["est_torque_yaw"])
        ax = axes[panel]
        ax.plot(t, gt_tyaw,  color="tab:brown", linewidth=1.2, alpha=0.9, label="GT τ_yaw")
        ax.plot(t, est_tyaw, color="tab:brown", linewidth=0.8, alpha=0.5, linestyle="--", label="Est τ_yaw")
        ax.set_ylabel("Torque (Nm)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title("τ_yaw: GT (solid) vs Estimated (dashed)", fontsize=11)
        panel += 1

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    eval_dir = os.path.join(os.path.dirname(checkpoint_path), "force_eval")
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dim_str = "6d" if has_torque_yaw else ("xyz" if has_fz else "xy")
    filename = f"static_eval_{dim_str}_F{force_max:.0f}_k{compliance_k:.4f}_ema{ema_alpha}_{timestamp}.png"
    fig_path = os.path.join(eval_dir, filename)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[static_eval] Plot saved: {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play a trained policy while visualizing GT and estimated forces."""

    # ── Config overrides ──────────────────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Flat ground override
    if args_cli.flat:
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainImporterCfg
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        if hasattr(env_cfg, "curriculum"):
            env_cfg.curriculum = None

    # Standing robot: zero velocity commands, disable built-in debug arrows
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.debug_vis = False

    # Auto-detect force event term name (XY vs XYZ) from agent config
    force_event_name = agent_cfg.to_dict().get("force_event_term_name", "persistent_xy_force")
    is_wrench = "wrench" in force_event_name
    is_xyz = "xyz" in force_event_name or is_wrench

    # Activate forces immediately with user-specified range
    force_event = getattr(env_cfg.events, force_event_name)
    if args_cli.torque_only:
        force_event.params["force_range"] = (0.0, 0.0)
    else:
        force_event.params["force_range"] = (args_cli.force_min, args_cli.force_max)
    if is_wrench and "torque_range" in force_event.params:
        force_event.params["torque_range"] = (args_cli.torque_min, args_cli.torque_max)
    # Trapezoid modes
    if args_cli.trapezoid_cycle and is_wrench:
        # Multi-cycle: apply_trapezoid_wrench fires every step, manages own state
        from go2_rl_lab.tasks.manager_based.go2_rl_lab.mdp.events import apply_trapezoid_wrench
        force_event.func = apply_trapezoid_wrench
        force_event.params.setdefault("ramp_s_range", (0.2, 0.8))
        force_event.params.setdefault("hold_s_range", (1.5, 3.0))
        force_event.params.setdefault("zero_s_range", (0.5, 1.5))
        force_event.params.setdefault("zero_prob", 0.2)
        force_event.params.setdefault("bucket_fracs", ((0.0, 1.0),))
        force_event.interval_range_s = (0.02, 0.02)
    elif args_cli.trapezoid_paint and is_wrench:
        # PAINT-style applied manually in the sim loop — disable the interval event
        force_event.params["force_range"] = (0.0, 0.0)
        force_event.interval_range_s = (1e6, 1e6)
    else:
        # Constant persistent wrench: re-randomize every 1-3s.
        # Always override func so trapezoid-trained tasks (apply_trapezoid_wrench,
        # apply_paint_trapezoid_wrench) get a simple constant force for eval.
        if is_wrench:
            from go2_rl_lab.tasks.manager_based.go2_rl_lab.mdp.events import apply_persistent_wrench
            force_event.func = apply_persistent_wrench
            force_event.params = {
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "force_range": (0.0, 0.0) if args_cli.torque_only else (args_cli.force_min, args_cli.force_max),
                "fz_scale": 0.8,
                "torque_range": (args_cli.torque_min, args_cli.torque_max),
            }
        force_event.interval_range_s = (1.0, 3.0)

    # Override the reset event to also apply forces on episode start.
    if is_wrench:
        from go2_rl_lab.tasks.manager_based.go2_rl_lab.mdp.events import apply_persistent_wrench
        env_cfg.events.base_external_force_torque.func = apply_persistent_wrench
        reset_force_range = (0.0, 0.0) if args_cli.torque_only else (args_cli.force_min, args_cli.force_max)
        env_cfg.events.base_external_force_torque.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": reset_force_range,
            "fz_scale": force_event.params.get("fz_scale", 0.6),
            "torque_range": (args_cli.torque_min, args_cli.torque_max),
        }
    elif is_xyz:
        from go2_rl_lab.tasks.manager_based.go2_rl_lab.mdp.events import apply_persistent_xyz_force
        env_cfg.events.base_external_force_torque.func = apply_persistent_xyz_force
        env_cfg.events.base_external_force_torque.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (args_cli.force_min, args_cli.force_max),
            "fz_scale": force_event.params.get("fz_scale", 0.6),
        }
    else:
        from go2_rl_lab.tasks.manager_based.go2_rl_lab.mdp.events import apply_persistent_xy_force
        env_cfg.events.base_external_force_torque.func = apply_persistent_xy_force
        env_cfg.events.base_external_force_torque.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (args_cli.force_min, args_cli.force_max),
        }

    # ── Resolve checkpoint ────────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[static_eval] Checkpoint: {resume_path}")

    # ── Create env + runner (auto-detect runner class) ──────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class_name = agent_cfg.class_name
    print(f"[static_eval] Runner class: {runner_class_name}")

    if runner_class_name == "CompliantOnPolicyRunner":
        from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner
        train_cfg = agent_cfg.to_dict()
        if args_cli.checkpoint:
            train_cfg["estimator_checkpoint"] = args_cli.checkpoint
        runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
        runner_mode = "compliant"
    elif runner_class_name == "FrozenPolicyEstimatorRunner":
        from go2_rl_lab.estimator.frozen_policy_estimator_runner import FrozenPolicyEstimatorRunner
        train_cfg = agent_cfg.to_dict()
        if args_cli.checkpoint:
            train_cfg["estimator_checkpoint"] = args_cli.checkpoint
        runner = FrozenPolicyEstimatorRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
        runner_mode = "compliant"
    elif runner_class_name == "ForceOnPolicyRunner":
        from go2_rl_lab.estimator.force_runner import ForceOnPolicyRunner
        runner = ForceOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner_mode = "force"
    else:
        raise ValueError(f"Unsupported runner class for static_eval: {runner_class_name}")

    runner.load(resume_path)
    runner.eval_mode()

    # runner.load() may set force_range=(0, max_force) on the live event — restore our eval range.
    if is_wrench and not args_cli.trapezoid_cycle and not args_cli.trapezoid_paint:
        try:
            isaac_env_raw = env.unwrapped
            live_event = isaac_env_raw.event_manager.get_term_cfg(force_event_name)
            live_event.params["force_range"] = (0.0, 0.0) if args_cli.torque_only else (args_cli.force_min, args_cli.force_max)
            if "torque_range" in live_event.params:
                live_event.params["torque_range"] = (args_cli.torque_min, args_cli.torque_max)
        except Exception:
            pass

    # Switch to the estimator-wrapped env so policy gets augmented obs
    if hasattr(runner, "_wrapped_env"):
        env = runner._wrapped_env

    # For CompliantOnPolicyRunner, activate the mapping so compliance is visible
    if runner_mode == "compliant" and hasattr(runner, "_mapping_active"):
        runner._mapping_active = True
        isaac_env_raw = env.unwrapped if hasattr(env, "unwrapped") else env
        isaac_env_raw._mapping_active = True
        isaac_env_raw._compliance_alpha = runner._compliance_alpha
        isaac_env_raw._compliance_beta = runner._compliance_beta
        print(
            f"[static_eval] Compliance mapping forced ON: "
            f"alpha={runner._compliance_alpha:.1f}N  beta={runner._compliance_beta:.1f}"
        )

    # For CompliantOnPolicyRunner, determine the raw obs dim (without force estimate)
    if runner_mode == "compliant":
        compliant_raw_obs_dim = runner._num_one_step_obs

    # Detect force dimension and layout from runner
    force_dim = getattr(runner, "_force_dim", 2)
    force_layout = getattr(runner.estimator, "force_layout", "auto") if hasattr(runner, "estimator") else "auto"
    is_xy_yaw = (force_layout == "xy_yaw" and force_dim == 3)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ── Access robot asset for GT force and pose ──────────────────────────
    isaac_env = env.unwrapped
    device = isaac_env.device
    asset = isaac_env.scene["robot"]
    base_body_ids, _ = asset.find_bodies("base")
    base_idx = base_body_ids[0] if isinstance(base_body_ids, (list, tuple)) else int(base_body_ids)

    # ── Create arrow markers ─────────────────────────────────────────────
    gt_markers = _create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))       # Red = GT force
    est_markers = _create_arrow_markers("/World/Visuals/EstForceArrow", (0.2, 0.4, 1.0)) if args_cli.show_est else None
    cmd_markers = _create_arrow_markers("/World/Visuals/CmdVelArrow", (0.0, 0.8, 0.2)) if args_cli.show_cmd else None
    adj_markers = _create_arrow_markers("/World/Visuals/AdjVelArrow", (1.0, 0.85, 0.0)) if args_cli.show_adj else None

    # Torque arrows: two tangent arrows forming a circle (magenta)
    torque_markers_1 = _create_arrow_markers("/World/Visuals/TorqueArrow1", (1.0, 0.0, 1.0)) if args_cli.show_torque else None
    torque_markers_2 = _create_arrow_markers("/World/Visuals/TorqueArrow2", (1.0, 0.0, 1.0)) if args_cli.show_torque else None

    # ── Run loop ──────────────────────────────────────────────────────────
    dt = isaac_env.step_dt
    n = args_cli.num_envs

    # Initialize force estimate for CompliantOnPolicyRunner
    if runner_mode == "compliant":
        isaac_env._force_estimate_xy = torch.zeros(n, force_dim, device=device)

    # ── PAINT trapezoid state ─────────────────────────────────────────────
    paint_active = args_cli.trapezoid_paint and is_wrench
    if paint_active:
        paint_T = args_cli.paint_period
        paint_t = torch.zeros(n, device=device)
        _paint_body_ids = base_body_ids if isinstance(base_body_ids, list) else [int(base_body_ids)]
        _paint_fz_scale = force_event.params.get("fz_scale", 0.6)
        _paint_f = torch.zeros(n, 1, 3, device=device)   # [n, 1-body, xyz]
        _paint_t = torch.zeros(n, 1, 3, device=device)   # [n, 1-body, xyz]
        _paint_all_ids = torch.arange(n, device=device)

        def _resample_paint(ids):
            ni = len(ids)
            f_max = args_cli.force_max
            t_max = args_cli.torque_max
            _paint_f[ids, 0, 0] = (torch.rand(ni, device=device) * 2 - 1) * f_max
            _paint_f[ids, 0, 1] = (torch.rand(ni, device=device) * 2 - 1) * f_max
            if is_xyz:
                _paint_f[ids, 0, 2] = (torch.rand(ni, device=device) * 2 - 1) * f_max * _paint_fz_scale
            if is_wrench:
                _paint_t[ids, 0, 2] = (torch.rand(ni, device=device) * 2 - 1) * t_max

        _resample_paint(_paint_all_ids)

    obs = env.get_observations()
    step_count = 0
    max_steps = int(args_cli.duration / dt)

    # Compliance (always XY only)
    ema_alpha = args_cli.ema_alpha
    compliance_k = args_cli.compliance_k
    force_ema = torch.zeros(n, 2, device=device)  # EMA-filtered force estimate (XY only)

    # Data collection for plots (env 0 only)
    plot_log: dict[str, list] = {
        "time_s": [],
        "gt_force_x": [],
        "gt_force_y": [],
        "est_force_x": [],
        "est_force_y": [],
        "base_vel_x": [],
        "base_vel_y": [],
        "cmd_vel_x": [],
        "cmd_vel_y": [],
        "adj_vel_x": [],
        "adj_vel_y": [],
        "rerandom_steps": [],
    }
    if force_dim >= 3 and not is_xy_yaw:
        plot_log["gt_force_z"] = []
        plot_log["est_force_z"] = []
    if force_dim >= 4 or is_xy_yaw:
        plot_log["gt_torque_yaw"] = []
        plot_log["est_torque_yaw"] = []
    if force_dim >= 6:
        plot_log["gt_torque_roll"] = []
        plot_log["gt_torque_pitch"] = []
        plot_log["est_torque_roll"] = []
        plot_log["est_torque_pitch"] = []
    prev_gt_xy = None
    fd3 = min(force_dim, 3)
    yaw_idx = 2 if is_xy_yaw else {4: 3, 6: 5}.get(force_dim, None)

    print(f"\n{'=' * 70}")
    print(f"  Runner      : {runner_class_name} ({runner_mode})")
    print(f"  Mode        : STATIC (zero velocity commands)")
    dim_labels = {2: "XY (2D)", 3: "XYZ (3D)", 6: "6D wrench"}
    print(f"  Force dims  : {dim_labels.get(force_dim, f'{force_dim}D')}")
    print(f"  Force range : [{args_cli.force_min:.0f}, {args_cli.force_max:.0f}] N per axis" +
          (f" (fz: x0.6)" if force_dim >= 3 else ""))
    print(f"  EMA alpha   : {ema_alpha}")
    if compliance_k > 0.0:
        print(f"  Compliance  : k={compliance_k:.4f}  (v* = v_cmd + k * F_hat_xy)")
    else:
        print(f"  Compliance  : DISABLED (use --compliance_k to enable)")
    if args_cli.compliance_k_yaw > 0.0:
        print(f"  Yaw comply  : k_yaw={args_cli.compliance_k_yaw:.4f}  (wz* = wz + k_yaw * τ_yaw)")
    if args_cli.torque_only:
        print(f"  Mode        : TORQUE-ONLY (no forces)")
    print(f"  Torque range: [{args_cli.torque_min:.1f}, {args_cli.torque_max:.1f}] Nm")
    if args_cli.trapezoid_paint:
        print(f"  Force profile: PAINT trapezoid (T={args_cli.paint_period:.1f}s, Cartesian per-axis)")
    elif args_cli.trapezoid_cycle:
        print(f"  Force profile: multi-cycle trapezoid (ramp/hold/zero repeating)")
    arrows = ["RED=GT"]
    if args_cli.show_est:
        arrows.append("BLUE=est")
    if args_cli.show_cmd:
        arrows.append("GREEN=cmd")
    if args_cli.show_adj:
        arrows.append("YELLOW=adj")
    if args_cli.show_torque:
        arrows.append("MAGENTA=torque")
    print(f"  Arrows      : {', '.join(arrows)}")
    print(f"  Duration    : {args_cli.duration:.0f}s ({max_steps} steps)")
    print(f"  Envs        : {n}")
    print(f"{'=' * 70}\n")

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # ── CompliantOnPolicyRunner: update estimator BEFORE acting ──
                if runner_mode == "compliant":
                    raw_obs = obs["policy"][:, :compliant_raw_obs_dim]
                    runner._history_buffer.insert(raw_obs)
                    force_hat_pre, _ = runner.estimator.get_latent(
                        runner._history_buffer.get_flattened()
                    )
                    isaac_env._force_estimate_xy = force_hat_pre
                    # EMA filter the estimate
                    force_ema = ema_alpha * force_hat_pre[:, :2] + (1.0 - ema_alpha) * force_ema

                # ── Inject adjusted velocity command into obs ─────────
                # Policy obs layout: [6:8] = velocity_commands (vx, vy)
                # v* = v_cmd + k * EMA(F_hat)
                if compliance_k > 0.0:
                    obs["policy"][:, 6] = obs["policy"][:, 6] + compliance_k * force_ema[:, 0]
                    obs["policy"][:, 7] = obs["policy"][:, 7] + compliance_k * force_ema[:, 1]

                # Yaw compliance: wz* = wz + k_yaw * τ_yaw
                if args_cli.compliance_k_yaw > 0.0 and yaw_idx is not None and runner_mode == "compliant":
                    if yaw_idx < force_hat_pre.shape[1]:
                        obs["policy"][:, 8] = obs["policy"][:, 8] + args_cli.compliance_k_yaw * force_hat_pre[:, yaw_idx]

                # ── Policy step ───────────────────────────────────────
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

                # ── CompliantOnPolicyRunner: reset history for terminated envs ──
                if runner_mode == "compliant":
                    done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids) > 0:
                        runner._history_buffer.reset(done_ids)

                # ── PAINT trapezoid force application ─────────────────
                if paint_active:
                    t_up = 0.1 * paint_T
                    t_hold = 0.8 * paint_T
                    t_down = 0.1 * paint_T
                    alpha = torch.zeros(n, device=device)
                    in_up = paint_t < t_up
                    in_hold = (paint_t >= t_up) & (paint_t < t_up + t_hold)
                    in_dn = paint_t >= t_up + t_hold
                    alpha[in_up] = (paint_t[in_up] / t_up).clamp(0.0, 1.0)
                    alpha[in_hold] = 1.0
                    alpha[in_dn] = ((paint_T - paint_t[in_dn]) / t_down).clamp(0.0, 1.0)
                    a = alpha.view(n, 1, 1)
                    asset.permanent_wrench_composer.set_forces_and_torques(
                        forces=_paint_f * a,
                        torques=_paint_t * a,
                        body_ids=_paint_body_ids,
                        env_ids=_paint_all_ids,
                    )
                    paint_t += dt
                    new_period = paint_t >= paint_T
                    if new_period.any():
                        np_ids = new_period.nonzero(as_tuple=False).squeeze(-1)
                        paint_t[np_ids] = 0.0
                        _resample_paint(np_ids)
                        plot_log["rerandom_steps"].append(step_count)

                # ── GT force (body frame from wrench composer) ─────────
                if is_xy_yaw:
                    gt_f2 = asset.permanent_wrench_composer.composed_force_as_torch[:n, base_idx, :2]
                    gt_tz = asset.permanent_wrench_composer.composed_torque_as_torch[:n, base_idx, 2:3]
                    gt_force_body = torch.cat([gt_f2, gt_tz], dim=-1)
                elif force_dim <= 3:
                    gt_force_body = asset.permanent_wrench_composer.composed_force_as_torch[
                        :n, base_idx, :force_dim
                    ]
                elif force_dim == 4:
                    gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:n, base_idx, :3]
                    gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:n, base_idx, 2:3]
                    gt_force_body = torch.cat([gt_f, gt_t], dim=-1)
                else:
                    gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:n, base_idx, :3]
                    gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:n, base_idx, :3]
                    gt_force_body = torch.cat([gt_f, gt_t], dim=-1)
                if gt_force_body.dim() == 3:
                    gt_force_body = gt_force_body.squeeze(1)

                # ── NN estimated force (body frame) ───────────────────
                force_hat, _ = runner.estimator.get_latent(
                    runner._history_buffer.get_flattened()
                )
                force_hat = force_hat[:n]

                # EMA filter (XY only for compliance)
                if runner_mode != "compliant":
                    force_ema = ema_alpha * force_hat[:, :2] + (1.0 - ema_alpha) * force_ema

                # ── Read base velocity and velocity command ───────────
                base_lin_vel = asset.data.root_lin_vel_b[:n]  # [n, 3]
                cmd_vel = isaac_env.command_manager.get_command("base_velocity")[:n]  # [n, 3+]

                # Adjusted velocity: v* = v_cmd + k * EMA(F_hat) (XY only)
                adj_vel_x = cmd_vel[:, 0] + compliance_k * force_ema[:, 0]
                adj_vel_y = cmd_vel[:, 1] + compliance_k * force_ema[:, 1]

                # ── Transform to world frame for arrow rendering ──────
                base_pos = asset.data.root_pos_w[:n]
                base_quat = asset.data.root_quat_w[:n]

                # ── Camera follow env 0 ───────────────────────────────
                if args_cli.follow_robot:
                    p = base_pos[0].cpu().tolist()
                    sim_utils.SimulationContext.instance().set_camera_view(
                        eye=(p[0] - 2.0, p[1] - 2.0, p[2] + 1.5),
                        target=(p[0], p[1], p[2] + 0.3),
                    )

                # ── Arrow direction vectors (body→world) ──────────────
                fd3 = min(force_dim, 3)
                force_scale = 0.05   # 20N -> length 1.0
                vel_scale = 1.0      # 1 m/s -> length 1.0

                # GT force arrow (RED) — always shown
                gt_3d = torch.zeros(n, 3, device=device)
                gt_3d[:, :fd3] = gt_force_body[:, :fd3]
                gt_world = quat_apply(base_quat, gt_3d)
                gt_pos = base_pos.clone()
                gt_pos[:, 2] += 0.55
                gt_quats = _force_to_quat(gt_world, device)
                gt_mag_w = gt_world.norm(dim=-1, keepdim=True)
                gt_scales = torch.full((n, 3), 0.3, device=device)
                gt_scales[:, 0:1] = (gt_mag_w * force_scale).clamp(min=0.05)
                gt_markers.visualize(gt_pos, gt_quats, gt_scales)

                # Estimated force arrow (BLUE)
                if est_markers is not None:
                    est_3d = torch.zeros(n, 3, device=device)
                    est_3d[:, :fd3] = force_hat[:, :fd3]
                    est_world = quat_apply(base_quat, est_3d)
                    est_pos = base_pos.clone()
                    est_pos[:, 2] += 0.45
                    est_quats = _force_to_quat(est_world, device)
                    est_mag_w = est_world.norm(dim=-1, keepdim=True)
                    est_scales = torch.full((n, 3), 0.3, device=device)
                    est_scales[:, 0:1] = (est_mag_w * force_scale).clamp(min=0.05)
                    est_markers.visualize(est_pos, est_quats, est_scales)

                # Normal cmd vel arrow (GREEN)
                if cmd_markers is not None:
                    cmd_3d = torch.zeros(n, 3, device=device)
                    cmd_3d[:, 0] = cmd_vel[:, 0]
                    cmd_3d[:, 1] = cmd_vel[:, 1]
                    cmd_world = quat_apply(base_quat, cmd_3d)
                    cmd_pos = base_pos.clone()
                    cmd_pos[:, 2] += 0.35
                    cmd_quats = _force_to_quat(cmd_world, device)
                    cmd_mag_w = cmd_world.norm(dim=-1, keepdim=True)
                    cmd_scales = torch.full((n, 3), 0.3, device=device)
                    cmd_scales[:, 0:1] = (cmd_mag_w * vel_scale).clamp(min=0.05)
                    cmd_markers.visualize(cmd_pos, cmd_quats, cmd_scales)

                # Adjusted cmd vel arrow (YELLOW)
                if adj_markers is not None:
                    adj_3d = torch.zeros(n, 3, device=device)
                    adj_3d[:, 0] = adj_vel_x
                    adj_3d[:, 1] = adj_vel_y
                    adj_world = quat_apply(base_quat, adj_3d)
                    adj_pos = base_pos.clone()
                    adj_pos[:, 2] += 0.25
                    adj_quats = _force_to_quat(adj_world, device)
                    adj_mag_w = adj_world.norm(dim=-1, keepdim=True)
                    adj_scales = torch.full((n, 3), 0.3, device=device)
                    adj_scales[:, 0:1] = (adj_mag_w * vel_scale).clamp(min=0.05)
                    adj_markers.visualize(adj_pos, adj_quats, adj_scales)

                # Yaw torque arrows (MAGENTA) — two tangent arrows at ±Y and ∓X
                if torque_markers_1 is not None and yaw_idx is not None:
                    gt_torque_yaw = gt_force_body[:, yaw_idx] if yaw_idx < gt_force_body.shape[1] else torch.zeros(n, device=device)
                    radius = torch.abs(gt_torque_yaw) * 0.15
                    sign = torch.sign(gt_torque_yaw)
                    sign[sign == 0] = 1.0

                    pos1 = base_pos.clone()
                    pos1[:, 1] += radius
                    pos1[:, 2] += 0.5
                    arrow1_dir = sign.unsqueeze(-1) * torch.tensor([[-1.0, 0.0, 0.0]], device=device).expand(n, 3)
                    quat1 = _force_to_quat(arrow1_dir, device)
                    scale1 = torch.full((n, 3), 0.3, device=device)
                    scale1[:, 0] = 0.4

                    pos2 = base_pos.clone()
                    pos2[:, 0] -= radius
                    pos2[:, 2] += 0.5
                    arrow2_dir = sign.unsqueeze(-1) * torch.tensor([[0.0, -1.0, 0.0]], device=device).expand(n, 3)
                    quat2 = _force_to_quat(arrow2_dir, device)
                    scale2 = torch.full((n, 3), 0.3, device=device)
                    scale2[:, 0] = 0.4

                    torque_markers_1.visualize(pos1, quat1, scale1)
                    torque_markers_2.visualize(pos2, quat2, scale2)

                # ── Record data for plots (env 0) ────────────────────
                g0 = gt_force_body[0]
                e0 = force_hat[0]

                plot_log["time_s"].append(step_count * dt)
                plot_log["gt_force_x"].append(g0[0].item())
                plot_log["gt_force_y"].append(g0[1].item())
                plot_log["est_force_x"].append(e0[0].item())
                plot_log["est_force_y"].append(e0[1].item())
                if force_dim >= 3 and not is_xy_yaw:
                    plot_log["gt_force_z"].append(g0[2].item())
                    plot_log["est_force_z"].append(e0[2].item())
                if yaw_idx is not None:
                    plot_log["gt_torque_yaw"].append(g0[yaw_idx].item())
                    plot_log["est_torque_yaw"].append(e0[yaw_idx].item())
                if force_dim >= 6:
                    plot_log["gt_torque_roll"].append(g0[3].item())
                    plot_log["gt_torque_pitch"].append(g0[4].item())
                    plot_log["est_torque_roll"].append(e0[3].item())
                    plot_log["est_torque_pitch"].append(e0[4].item())
                plot_log["base_vel_x"].append(base_lin_vel[0, 0].item())
                plot_log["base_vel_y"].append(base_lin_vel[0, 1].item())
                plot_log["cmd_vel_x"].append(cmd_vel[0, 0].item())
                plot_log["cmd_vel_y"].append(cmd_vel[0, 1].item())
                plot_log["adj_vel_x"].append(adj_vel_x[0].item())
                plot_log["adj_vel_y"].append(adj_vel_y[0].item())

                # Detect force re-randomization (GT force jumped)
                if prev_gt_xy is not None:
                    delta = (g0[:2] - prev_gt_xy).norm().item()
                    if delta > 1.0:
                        plot_log["rerandom_steps"].append(step_count)
                prev_gt_xy = g0[:2].clone()

                # ── Terminal readout (first env, every 10 steps) ──────
                step_count += 1
                if step_count % 10 == 0:
                    gm0 = g0[:fd3].norm().item()
                    em0 = e0[:fd3].norm().item()
                    pct = step_count / max_steps
                    bar_len = 20
                    filled = int(bar_len * pct)
                    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                    elapsed_s = step_count * dt
                    gt_parts = ",".join(f"{g0[i]:+6.1f}" for i in range(fd3))
                    est_parts = ",".join(f"{e0[i]:+6.1f}" for i in range(fd3))
                    tau_str = ""
                    if yaw_idx is not None:
                        tau_str = f"  τyaw:{g0[yaw_idx]:+5.1f}/{e0[yaw_idx]:+5.1f}"
                    print(
                        f"\r  [{bar}] {elapsed_s:.0f}/{args_cli.duration:.0f}s  "
                        f"GT:[{gt_parts}]N |{gm0:5.1f}|  "
                        f"Est:[{est_parts}]N |{em0:5.1f}|{tau_str}  "
                        f"v*:[{adj_vel_x[0]:+5.2f},{adj_vel_y[0]:+5.2f}]",
                        end="",
                        flush=True,
                    )

                if step_count >= max_steps:
                    break

            # Real-time pacing
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        print(f"\n\n[static_eval] Stopping simulation ({type(exc).__name__}).")

    print()

    # Close env first, then plot
    env.close()

    from time import sleep
    sleep(1)

    # ── Save plots ────────────────────────────────────────────────────────
    if len(plot_log["time_s"]) > 10:
        print(f"[static_eval] Collected {len(plot_log['time_s'])} steps. Saving plots...")
        generate_plots(
            plot_log, args_cli.force_min, args_cli.force_max, resume_path, ema_alpha, compliance_k,
        )
    else:
        print("[static_eval] Too few steps collected, skipping plots.")


if __name__ == "__main__":
    main()
    simulation_app.close()
